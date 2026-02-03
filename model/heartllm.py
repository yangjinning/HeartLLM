import torch.nn.functional as F
import torch
import torch.nn as nn
from transformers import LlamaConfig, LlamaModel, LlamaTokenizer,BitsAndBytesConfig, AutoModelForCausalLM,AutoTokenizer
import transformers
from transformers.models.llama.modeling_llama import LlamaRMSNorm
from peft import LoraConfig, get_peft_model
from ecg_tokenizer.tokenizer import FSQ_AE
import random


transformers.logging.set_verbosity_error()


class LinearProjection(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=2048, output_dim=4096, rms_norm_eps=1e-06):
        super().__init__()

        self.linear1 = nn.Linear(input_dim, hidden_dim, bias=False)  # 768->1024
        self.bn1 = LlamaRMSNorm(hidden_dim, eps=rms_norm_eps)

        self.linear2 = nn.Linear(hidden_dim, output_dim, bias=False)  # 1024->1024*32*2
        self.bn2 = LlamaRMSNorm(output_dim, eps=rms_norm_eps)

    def forward(self, x):
        out = self.linear1(x)
        out = F.gelu(out)
        out = self.bn1(out)

        out = self.linear2(out)
        out = self.bn2(out)
        return out


class heartllm(nn.Module):

    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            configs.local_llm_path,
            attn_implementation="flash_attention_2",
            local_files_only=True,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4'
            )
        )
        config = LlamaConfig.from_pretrained(configs.local_llm_path, local_files_only=True)
        self.d_llm = config.hidden_size
        self.levels = configs.levels

        self.tokenizer = AutoTokenizer.from_pretrained(configs.local_llm_path, local_files_only=True)
        self.tokenizer.padding_side = "left"

        self.ORIGIN_VOCAB_TEXT = len(self.tokenizer)
        configs.ORIGIN_VOCAB_TEXT = self.ORIGIN_VOCAB_TEXT

        self.tokenizer.pad_token = '<|PAD|>'
        num_special_tokens = self.tokenizer.add_special_tokens({'pad_token': '<|PAD|>'})
        new_tokens = ["<|start_ecg|>", "<|end_ecg|>"]
        num_normal_tokens = self.tokenizer.add_tokens(new_tokens)

        VOCAB_ECG = self.levels ** configs.ts_dim #1296
        self.MASK_ID = VOCAB_ECG  # 1297
        self.INFILL_ID = VOCAB_ECG + 1  # 1298
        self.VOCAB_TEXT = len(self.tokenizer)

        self.llm_model.resize_token_embeddings(self.VOCAB_TEXT + VOCAB_ECG + 2) #129259 + 1298 = 129557

        fsqpath = configs.tokenizer_path

        checkpoint = torch.load(fsqpath)
        fsqmodel = FSQ_AE(levels=self.levels)
        fsqmodel.load_state_dict(checkpoint["model"], strict=True)
        self.enc = fsqmodel.enc
        self.quant = fsqmodel.quant
        for name, param in self.enc.named_parameters():
            param.requires_grad = False
        for name, param in self.quant.named_parameters():
            param.requires_grad = False


        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
        lora_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.05,
            r=64,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )
        self.llm_model = get_peft_model(self.llm_model, lora_config)

        if configs.stage == "pretrain":
            for idx, (name, param) in enumerate(self.llm_model.named_parameters()):
                if ("norm" in name) or ("embed_tokens" in name) :
                    param.requires_grad = True
        else:
            for idx, (name, param) in enumerate(self.llm_model.named_parameters()):
                if "norm" in name:
                    param.requires_grad = True

        self.llm_model.print_trainable_parameters()

    def q2idx(self, code_disc: torch.Tensor, levels, ts_dim) -> torch.LongTensor:
        n = levels-1
        disc = torch.round(code_disc * n) / (levels-1)
        v = (disc * n).long()
        base = torch.tensor([levels ** i for i in range(ts_dim)], device=v.device)
        token_id = (v * base).sum(dim=-1)
        return token_id  # (B, T)

    def shift(self,seq):
        out = seq.clone()
        mask = out != -100  # -100 represents ignore_index
        out[mask] += self.VOCAB_TEXT
        return out

    def build_window_step_batch(self,seq, window_size, step_size, mask_label_id=-100):
        T = seq.shape[0]
        L = window_size + step_size
        max_start = T - L
        s = random.choice(range(0, max_start + 1, step_size))

        window_tokens = seq[s: s + window_size]
        step_tokens = seq[s + window_size: s + L]

        input_tokens = torch.cat([window_tokens, step_tokens], dim=0)
        label_tokens = torch.cat([
            torch.full((window_size,), mask_label_id, dtype=torch.long, device=seq.device),
            step_tokens
        ], dim=0)
        input_tokens = self.shift(input_tokens)
        label_tokens = self.shift(label_tokens)
        return input_tokens, label_tokens


    def predict_ecg_token(self,batch_ecg_data,ecg_idx, loss):
        B, L, _ = batch_ecg_data.shape
        prompt_prefix = []
        prompt_suffix = []
        ans = []
        for batch_idx in range(B):

            prompt_prefix_ = (
                "<|start_ecg|>"
            )
            prompt_suffix_ = ("<|end_ecg|>")

            prompt_prefix.append(prompt_prefix_)
            prompt_suffix.append(prompt_suffix_)
            ans.append(self.tokenizer.eos_token)

        prompt_prefix = self.tokenizer(prompt_prefix, return_tensors="pt", padding=True, truncation=True,max_length=4096).input_ids
        prompt_suffix = self.tokenizer(prompt_suffix, return_tensors="pt", padding=True, truncation=True,max_length=4096).input_ids
        ans = self.tokenizer(ans, return_tensors="pt", padding=True, truncation=True, max_length=4096).input_ids

        prompt_prefix_startidx = (prompt_prefix == self.tokenizer.pad_token_id).int().sum(dim=1)
        _, len_prompt_prefix = prompt_prefix.shape

        prompt_suffix_startidx = (prompt_suffix == self.tokenizer.pad_token_id).int().sum(dim=1)
        prompt_suffix_startidx += len_prompt_prefix
        _, len_prompt_suffix = prompt_suffix.shape

        ans_startidx = (ans == self.tokenizer.pad_token_id).int().sum(dim=1)
        ans_startidx += len_prompt_prefix + len_prompt_suffix

        all_input_ids = torch.cat((prompt_prefix, prompt_suffix, ans), dim=1).to(self.llm_model.device)

        prompt_ids_batch = []
        labels_batch = []
        MAX_LENGTH = 0
        window_sizes = [283, 142, 70, 35, 17]
        step_sizes = [30, 15, 7, 4, 2]
        for window_size, step_size in zip(window_sizes,step_sizes):
            for batch_idx in range(B):
                prompt_prefix_each = all_input_ids[batch_idx][prompt_prefix_startidx[batch_idx]:len_prompt_prefix]
                ecg_idx_each = ecg_idx[batch_idx]
                masked_ecg_idx_each, label_each = self.build_window_step_batch(ecg_idx_each, window_size, step_size)
                prompt_suffix_each = all_input_ids[batch_idx][prompt_suffix_startidx[batch_idx] + 1:(
                            len_prompt_prefix + len_prompt_suffix)]

                ans_each = all_input_ids[batch_idx][ans_startidx[batch_idx] + 1:]
                prompt_ids = torch.cat((prompt_prefix_each, masked_ecg_idx_each, prompt_suffix_each, ans_each), dim=0)
                MAX_LENGTH = max(prompt_ids.shape[0], MAX_LENGTH)
                labels = [-100] * len(prompt_prefix_each) + label_each.tolist() + [-100] * len(prompt_suffix_each) + [-100] * len(ans_each)
                labels_batch.append(labels)
                prompt_ids_batch.append(prompt_ids)

        pad_prompt_ids_batch = []
        pad_attention_mask_batch = []
        pad_labels_batch = []


        for prompt_ids, labels in zip(prompt_ids_batch, labels_batch):
            l = prompt_ids.shape[0]
            pad_len = MAX_LENGTH - prompt_ids.size(0)
            pad_ids = torch.full(
                (pad_len,),
                self.tokenizer.pad_token_id,
                dtype=torch.long,
                device=prompt_ids.device
            )
            pad_prompt_ids = torch.cat((pad_ids, prompt_ids), dim=0)
            pad_labels = F.pad(torch.tensor(labels), (MAX_LENGTH - len(labels), 0), value=-100)
            pad_attention_mask = F.pad(torch.tensor([1] * l), (MAX_LENGTH - l, 0), value=0)
            pad_prompt_ids_batch.append(pad_prompt_ids)
            pad_attention_mask_batch.append(pad_attention_mask)
            pad_labels_batch.append(pad_labels)

        pad_prompt_ids_batch = torch.stack(pad_prompt_ids_batch).to(self.llm_model.device)  # batchsize, MAX_LENGTH, 4096
        pad_attention_mask_batch = torch.stack(pad_attention_mask_batch).to(self.llm_model.device)
        pad_labels_batch = torch.stack(pad_labels_batch).to(self.llm_model.device)

        return self.llm_model, pad_prompt_ids_batch, pad_attention_mask_batch, pad_labels_batch, self.tokenizer, loss


    def pretrain(self, batch_ecg_data):
        sig = batch_ecg_data.permute(0, 2, 1)

        with torch.no_grad():
            z = self.enc(sig)  # 200,12,5000
            _, q = self.quant(z)  # q 313,384
            loss = 0

        batch_ecg_idx = self.q2idx(q, self.levels, self.configs.ts_dim)

        return self.predict_ecg_token(batch_ecg_data, batch_ecg_idx, loss)


    def finetune(self, batch_ecg_data, **kwargs):

        if self.configs.tasktype == "qa":
            batch_question = kwargs["batch_question"]
            batch_answer = kwargs["batch_answer"]
            batch_background = kwargs["batch_background"]
        elif self.configs.tasktype == "report":
            batch_background = kwargs["batch_background"]
            batch_report = kwargs["batch_report"]

        B, _, _ = batch_ecg_data.shape
        sig = batch_ecg_data.permute(0, 2, 1)

        with torch.no_grad():
            z = self.enc(sig)  # 200,12,5000
            _, q = self.quant(z)  # q 313,384
            loss = 0

        batch_ecg_idx = self.q2idx(q, self.levels, self.configs.ts_dim)

        prompt_prefix = []
        prompt_suffix = []
        ans = []
        for batch_idx in range(B):
            if self.configs.tasktype in ["qa","qa_analysis","qa_mask"]:
                question = batch_question[batch_idx]
                ans_ = batch_answer[batch_idx]
            elif self.configs.tasktype == "report":
                report = batch_report[batch_idx]
            background = batch_background[batch_idx]
            background = background.split("Recording Details")[0]

            if self.configs.tasktype == "qa":
                prompt_prefix_ = (
                    f"<|start_prompt|> Dataset description: The dataset contains electrocardiogram (ECG) time-series data. "
                    f"Task description: Based on a patient's 12-lead ECG signal (500 Hz) and background information, answer a clinical question related to cardiac health. "
                    f"Question:{question} "
                    "<|end_prompt|> "
                    "<|start_ecg|> "
                )
                prompt_suffix_ = ("<|end_ecg|>Answer:")
                prompt_prefix.append(prompt_prefix_)
                prompt_suffix.append(prompt_suffix_)
                ans.append(ans_ + self.tokenizer.eos_token)

            elif self.configs.tasktype == "report":
                prompt_prefix_ = (
                    f"<|start_prompt|>Task description: Based on the given ECG signal embeddings, generate a ECG diagnostic report."
                    f"Background:{background} "
                    "<|end_prompt|>"
                    "<|start_ecg|>"
                )
                prompt_suffix_ = ("<|end_ecg|> Report:")
                prompt_prefix.append(prompt_prefix_)
                prompt_suffix.append(prompt_suffix_)
                ans.append(report + self.tokenizer.eos_token)

        prompt_prefix = self.tokenizer(prompt_prefix, return_tensors="pt", padding=True, truncation=True,max_length=4096).input_ids
        prompt_suffix = self.tokenizer(prompt_suffix, return_tensors="pt", padding=True, truncation=True,max_length=4096).input_ids
        ans = self.tokenizer(ans, return_tensors="pt", padding=True, truncation=True, max_length=4096).input_ids

        prompt_prefix_startidx = (prompt_prefix == self.tokenizer.pad_token_id).int().sum(dim=1)
        _, len_prompt_prefix = prompt_prefix.shape

        prompt_suffix_startidx = (prompt_suffix == self.tokenizer.pad_token_id).int().sum(dim=1)
        prompt_suffix_startidx += len_prompt_prefix
        _, len_prompt_suffix = prompt_suffix.shape

        ans_startidx = (ans == self.tokenizer.pad_token_id).int().sum(dim=1)
        ans_startidx += len_prompt_prefix + len_prompt_suffix

        all_input_ids = torch.cat((prompt_prefix, prompt_suffix, ans), dim=1).to(self.llm_model.device)


        prompt_ids_batch = []
        labels_batch = []
        MAX_LENGTH = 0


        for batch_idx in range(B):
            prompt_prefix_each = all_input_ids[batch_idx][prompt_prefix_startidx[batch_idx]:len_prompt_prefix]
            ecg_idx_each = batch_ecg_idx[batch_idx]
            ecg_idx_each = self.shift(ecg_idx_each)
            prompt_suffix_each = all_input_ids[batch_idx][prompt_suffix_startidx[batch_idx] + 1:(len_prompt_prefix + len_prompt_suffix)]
            ans_each = all_input_ids[batch_idx][ans_startidx[batch_idx] + 1:]
            prompt_ids = torch.cat((prompt_prefix_each, ecg_idx_each, prompt_suffix_each, ans_each), dim=0)
            MAX_LENGTH = max(prompt_ids.shape[0], MAX_LENGTH)
            labels = [-100] * (len(prompt_prefix_each) + len(ecg_idx_each) + len(prompt_suffix_each)) + ans_each.tolist()
            labels_batch.append(labels)
            prompt_ids_batch.append(prompt_ids)

        pad_prompt_ids_batch = []
        pad_attention_mask_batch = []
        pad_labels_batch = []

        for idx,(prompt_ids, labels) in enumerate(zip(prompt_ids_batch, labels_batch)):
            l = prompt_ids.shape[0]
            pad_len = MAX_LENGTH - prompt_ids.size(0)
            pad_ids = torch.full(
                (pad_len,),
                self.tokenizer.pad_token_id,
                dtype=torch.long,
                device=prompt_ids.device
            )
            pad_prompt_ids = torch.cat((pad_ids, prompt_ids), dim=0)
            pad_labels = F.pad(torch.tensor(labels), (MAX_LENGTH - len(labels), 0), value=-100)
            pad_attention_mask = F.pad(torch.tensor([1] * l), (MAX_LENGTH - l, 0), value=0)
            pad_prompt_ids_batch.append(pad_prompt_ids)
            pad_attention_mask_batch.append(pad_attention_mask)
            pad_labels_batch.append(pad_labels)

        pad_prompt_ids_batch = torch.stack(pad_prompt_ids_batch).to(self.llm_model.device)
        pad_attention_mask_batch = torch.stack(pad_attention_mask_batch).to(self.llm_model.device)
        pad_labels_batch = torch.stack(pad_labels_batch).to(self.llm_model.device)

        return self.llm_model, pad_prompt_ids_batch, pad_attention_mask_batch, pad_labels_batch, self.tokenizer, loss


    def forward(self, stage, batch_ecg_data, **kwargs):
        if stage in ["finetune","test"]:
            if self.configs.tasktype == "qa":
                batch_question = kwargs["batch_question"]
                batch_answer = kwargs["batch_answer"]
                batch_background = kwargs["batch_background"]
                return self.finetune(batch_ecg_data=batch_ecg_data, batch_question=batch_question, batch_answer=batch_answer, batch_background=batch_background)
            elif self.configs.tasktype == "report":
                batch_background = kwargs["batch_background"]
                batch_report = kwargs["batch_report"]
                return self.finetune(batch_ecg_data=batch_ecg_data, batch_report=batch_report, batch_background=batch_background)

        elif stage == "pretrain":
            return self.pretrain(batch_ecg_data=batch_ecg_data)
        else:
            raise ValueError("stage must be finetune or pretrain, but got {}".format(stage))


