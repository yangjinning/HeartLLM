#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import argparse
import torch
from accelerate import Accelerator
from model import heartllm
from data_provider.data_factory import data_provider
import random
import numpy as np
from utils.tools import test
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

os.environ['CURL_CA_BUNDLE'] = ''
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


parser = argparse.ArgumentParser(description='heartllm')

fix_seed = 42
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)


parser.add_argument('--num_workers', type=int, default=8, help='data loader num workers')
parser.add_argument('--batch_size', type=int, default=50, required=False,help='model name')
parser.add_argument('--dataset', type=str, default="", required=True,help='dataset name, options:[ptbxl,mimic-iv-ecg]')
parser.add_argument('--num_test', type=int, default=None, required=False,help='number of test samples, default None means all samples')
parser.add_argument('--ckp_path', type=str, default="",help='checkpoint file path followed /results/')
parser.add_argument('--tasktype', type=str, default="report", required=False,help='task name, options:[qa,report]')
parser.add_argument('--pretrain_path', type=str, default="", required=False,help='path of pretrained heartllm')
parser.add_argument('--stage', type=str, default="test", required=False,help='testing stage')
parser.add_argument('--local_llm_path', type=str, default="", required=False,help='path of LLM')
parser.add_argument('--llm', type=str, default="3b", required=False,help='3b,7b,8b')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--root_path_ecg', type=str, default="", required=False,help='path of original dataset')
parser.add_argument('--root_path_json', type=str, default="", required=False, help='Directory of ECG QA JSON lists (used when tasktype=qa)')
parser.add_argument('--root_report_json', type=str, default="", required=False, help='Directory of report JSON lists (used when tasktype=report)')
parser.add_argument('--tokenizer_path', type=str, default="", required=False, help='Path to ECG tokenizer checkpoint (e.g. ecg_tokenizer/result_tokenzier/best.pt)')

args = parser.parse_args()

args.shuffle_flag = True

args.llm_dim = 3072

model_name = args.ckp_path.split("/")[-3]

if args.tasktype == "qa":
    args.max_new_tokens = 170
else:
    args.max_new_tokens = 200

accelerator = Accelerator()

accelerator.print(f"==========DATASET:{args.dataset}==============")


args.ts_dim = 4
args.levels = 6
model = heartllm(args).to(accelerator.device)
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

test_data, test_loader = data_provider(args, 'test',args.num_test)

checkpoint = torch.load(args.ckp_path, map_location=accelerator.device)

model.load_state_dict(checkpoint)

model.to(torch.bfloat16)

test_loader, model = accelerator.prepare(test_loader, model)

tokenizer = AutoTokenizer.from_pretrained(args.local_llm_path, local_files_only=True)

tokenizer.padding_side = "left"

model.eval()


with torch.no_grad():
    model = accelerator.unwrap_model(model)
    results = test(args, accelerator, model, test_loader)