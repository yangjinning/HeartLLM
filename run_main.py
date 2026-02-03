import os
import argparse
import torch
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate import DistributedDataParallelKwargs
from torch import nn, optim
from tqdm import tqdm
from model import heartllm
from data_provider.data_factory import data_provider
import time
import random
import numpy as np
import yaml
from utils.tools import vali
import shutil
from transformers import AutoModelForCausalLM

os.environ['RDMAV_FORK_SAFE'] = '1'
os.environ['CURL_CA_BUNDLE'] = ''
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

fix_seed = 42
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='heartllm')
parser.add_argument('--train_epochs', type=int, default=1, help='Number of training epochs to run')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='Optimizer (Adam) learning rate')
parser.add_argument('--use_amp', action='store_true', default=False, help='Use automatic mixed precision (AMP) to save memory and speed up training')
parser.add_argument('--tasktype', type=str, default="", required=False, help='Task type: "qa" for ECG Q&A, "report" for ECG report generation')
parser.add_argument('--batch_size', type=int, default=10, required=False, help='Training batch size per GPU')
parser.add_argument('--dataset', type=str, default="", required=True, help='Dataset name: "ptbxl" or "mimic-iv-ecg"')
parser.add_argument('--llm', type=str, default="3b", required=False, help='LLM size identifier: "3b", "7b", or "8b"')
parser.add_argument('--num_train', type=int, default=None, required=False, help='Max number of training samples to use (None = use all)')
parser.add_argument('--num_val', type=int, default=None, required=False, help='Max number of validation samples to use (None = use all)')
parser.add_argument('--stage', type=str, default="pretrain", required=True, help='Training stage: "pretrain" (ECG tokenizer + LLM) or "finetune" (downstream task)')
parser.add_argument('--local_llm_path', type=str, default="", required=False, help='Local path to the base LLM (e.g. Llama-3.2-3B)')
parser.add_argument('--verbose_steps', type=int, default=1, required=False, help='Print training loss every N steps')
parser.add_argument('--eval_steps', type=int, default=2, required=False, help='Run validation every N steps')
parser.add_argument('--pretrain_path', type=str, default="", required=False, help='Path to pretrained heartllm checkpoint (used when stage=finetune)')
parser.add_argument('--root_path_ecg', type=str, default="", required=False, help='Root directory of raw ECG data (wfdb-compatible files)')
parser.add_argument('--root_path_json', type=str, default="", required=False, help='Directory of ECG QA JSON lists (used when tasktype=qa)')
parser.add_argument('--root_report_json', type=str, default="", required=False, help='Directory of report JSON lists (used when tasktype=report)')
parser.add_argument('--tokenizer_path', type=str, default="", required=False, help='Path to ECG tokenizer checkpoint (e.g. ecg_tokenizer/result_tokenzier/best.pt)')

args = parser.parse_args()

args.llm_dim = 3072

ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
deepspeed_plugin = DeepSpeedPlugin(hf_ds_config='./ds_config_zero2.json')
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], deepspeed_plugin=deepspeed_plugin)


accelerator.print(f"==========DATASET:{args.dataset}==============")
accelerator.print(f"=============STAGE:{args.stage}==============")

num_gpu = accelerator.num_processes
num_cpu = os.cpu_count()
args.num_workers = 8

args.ts_dim = 4
args.levels = 6
# args.task="taskA"
model = heartllm(args).float()

if args.stage == "finetune":
    checkpoint = torch.load(args.pretrain_path)
    llama = AutoModelForCausalLM.from_pretrained(args.local_llm_path,
                                                     torch_dtype=torch.float16,
                                                     local_files_only=True,
                                                     )
    checkpoint["llm_model.base_model.model.model.embed_tokens.weight"][:model.ORIGIN_VOCAB_TEXT] = llama.get_input_embeddings().weight[:model.ORIGIN_VOCAB_TEXT]

    model.load_state_dict(checkpoint)


    for idx, (name, param) in enumerate(model.named_parameters()):
        if ("norm" in name) or ("lora" in name):
            param.requires_grad = True
        else:
            param.requires_grad = False
        if ("enc" in name):
            param.requires_grad = False


llm_name = args.local_llm_path.split("/")[-1]
setting = '{}_{}_{}'.format(args.tasktype,args.dataset,args.stage)
args.path = f'./results/{setting}'
model_path = os.path.join(args.path,"checkpoint")

if accelerator.is_local_main_process:
    os.makedirs(args.path, exist_ok=True)
    os.makedirs(os.path.join(args.path, "checkpoint"), exist_ok=True)

time_now = time.time()

total_params,trainable_params,trained_parameters = [],[],[]
for name, p in model.named_parameters():
    total_params.append(p.numel())
    if p.requires_grad is True:
        accelerator.print("Train:", name, " Number of Parameters:", p.numel())
        trained_parameters.append(p)
        trainable_params.append(p.numel())
    else: accelerator.print("Freeze:", name)
accelerator.print(f"Overall Situation: "
      f"Total Parameters: {sum(total_params)} || Trainable Parameters: {sum(trainable_params)} || Trainable Ratio: {sum(trainable_params)/sum(total_params)* 100:.2f}%")


model_optim = optim.Adam(trained_parameters, lr=args.learning_rate)

train_data, train_loader = data_provider(args, 'train', args.num_train)
vali_data, vali_loader = data_provider(args, 'valid', args.num_val)

num_batch = len(train_loader)
train_steps = num_batch // num_gpu

train_loader, vali_loader, model, model_optim = accelerator.prepare(train_loader, vali_loader, model, model_optim)

if args.use_amp:
    scaler = torch.cuda.amp.GradScaler()


for epoch in range(args.train_epochs):
    iter_count = 0

    train_step_list = []
    train_loss_list = []
    val_step_list = []
    val_loss_list = []

    model.train()
    epoch_time = time.time()

    for i, data in tqdm(enumerate(train_loader)):
        iter_count += 1
        model_optim.zero_grad()

        if args.stage == "pretrain":
            batch_ecg_data, _ = data
            batch_ecg_data = batch_ecg_data.to(torch.bfloat16).to(accelerator.device)
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(args.stage, batch_ecg_data)
            else:
                outputs = model(args.stage, batch_ecg_data)

            llm_model, pad_prompt_ids_batch, pad_attention_mask_batch, pad_labels_batch, _, loss = outputs

            outputs = llm_model(input_ids=pad_prompt_ids_batch, attention_mask=pad_attention_mask_batch,
                                labels=pad_labels_batch)
            loss = outputs.loss

        else:
            if args.tasktype == "qa":
                batch_ecg_data, batch_question, batch_answer, batch_background, _ = data
                batch_ecg_data = batch_ecg_data.to(torch.bfloat16).to(accelerator.device)
                if args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = model(args.stage, batch_ecg_data=batch_ecg_data, batch_question=batch_question, batch_answer=batch_answer, batch_background=batch_background)
                else:
                    outputs = model(args.stage, batch_ecg_data=batch_ecg_data, batch_question=batch_question, batch_answer=batch_answer, batch_background=batch_background)
            elif args.tasktype == "report":
                batch_ecg_data, batch_report, batch_background = data
                batch_ecg_data = batch_ecg_data.to(torch.bfloat16).to(accelerator.device)
                if args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = model(args.stage, batch_ecg_data=batch_ecg_data, batch_report=batch_report, batch_background=batch_background)
                else:
                    outputs = model(args.stage, batch_ecg_data=batch_ecg_data, batch_report=batch_report, batch_background=batch_background)

            llm_model, pad_prompt_ids_batch, pad_attention_mask_batch, pad_labels_batch, _, loss = outputs

            outputs = llm_model(input_ids=pad_prompt_ids_batch, attention_mask=pad_attention_mask_batch,
                                labels=pad_labels_batch)

            loss = outputs.loss


        train_loss_list.append(loss.detach().cpu().to(torch.float32).numpy())
        train_step_list.append(i)

        """print training loss of current batch"""
        if (i + 1) % args.verbose_steps == 0:
            accelerator.print(
                "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss))
            speed = (time.time() - time_now) / iter_count
            left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
            accelerator.print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
            iter_count = 0
            time_now = time.time()

        """evaluate"""
        EVAL = False
        if (i + 1) % args.eval_steps == 0:
            """evaluate checkpoint"""
            vali_avg_loss = vali(args, accelerator, model, vali_loader)
            val_loss_list.append(vali_avg_loss)
            val_step_list.append(i)
            accelerator.print(
                "Epoch: {0} | Train Loss: {1:.7f} Vali Loss(average per batch): {2:.7f}".format(
                    epoch + 1, loss, vali_avg_loss))
            EVAL = True

        if args.use_amp:
            scaler.scale(loss).backward()
            scaler.step(model_optim)
            scaler.update()
        else:
            accelerator.backward(loss)
            model_optim.step()


    """save and evaluate"""
    accelerator.print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
    train_avg_loss = np.average(train_loss_list)
    vali_avg_loss = vali(args, accelerator, model, vali_loader)
    accelerator.print(
        "Epoch: {0} | Train Loss(average per batch): {1:.7f} Vali Loss(average per batch): {2:.7f}".format(
            epoch + 1, train_avg_loss, vali_avg_loss))


    if accelerator is not None:
        if accelerator.is_local_main_process:
            model = accelerator.unwrap_model(model)
            torch.save(model.state_dict(),model_path + '/' + f'checkpoint_epoch{epoch}.pth')
            accelerator.print("Save model....")
    else:
        torch.save(model.state_dict(), model_path + '/' + f'checkpoint_epoch{epoch}.pth')
        print("Save model....")
