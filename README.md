# HeartLLM: Discretized ECG Tokenization for LLM-Based Diagnostic Reasoning

## 1. Environment Setup
```bash
conda create -n heartllm python=3.10
conda activate heartllm
pip install -r requirements.txt
```

## 2. Download checkpoints, datasets, LLM:

Please download files from 🔗 Google Drive:

https://drive.google.com/drive/folders/1y6oedzwgoyPkI1DeM5FzDTDGnxL4aBFu?usp=drive_link

You also need to manually download the following ECG datasets from PhysioNet:

MIMIC-IV-ECG: https://physionet.org/content/mimic-iv-ecg/1.0/

PTB-XL: https://physionet.org/content/ptb-xl/1.0.1/

## 3. Project Structure
After downloading the files from Google Drive, replace the existing files in the repository or create new directories as needed to match the following project structure.
All dataset paths and checkpoint locations are configured via .env files. Please modify these .env files according to your local file system.

File Structure
```bash
HeartLLM
├── corpora
│   └── wordnet.zip
├── data_provider
│   ├── data_factory.py
│   └── data_loader.py
├── dataset
│   ├── ecgqa
│   │   ├── mimic-iv-ecg
│   │   │   ├── template_test_background.json
│   │   │   ├── template_train_background.json
│   │   │   └── template_valid_background.json
│   │   └── ptbxl
│   │       ├── template_test_background.json
│   │       ├── template_train_background.json
│   │       └── template_valid_background.json
│   └── report
│       ├── mimic-iv-ecg
│       │   ├── test.json
│       │   ├── train.json
│       │   └── valid.json
│       └── ptbxl
│           ├── test.json
│           ├── train.json
│           └── valid.json
├── ecg_tokenizer
│   ├── result_tokenzier
│   │   └── best.pt
│   ├── config.env
│   ├── run_tokenizer.sh
│   └── tokenizer.py
├── env
│   ├── ft_qa_mimic.env
│   ├── ft_qa_ptbxl.env
│   ├── ft_report_mimic.env
│   ├── ft_report_ptbxl.env
│   ├── pretrain.env
│   ├── test_qa_mimic.env
│   ├── test_qa_ptbxl.env
│   ├── test_report_mimic.env
│   └── test_report_ptbxl.env
├── model
│   ├── __init__.py
│   └── heartllm.py
├── results
│   └── previous
│   │   ├── _mimic-iv-ecg_pretrain
│   │   │   └── checkpoint
│   │   │       └── checkpoint_epoch0.pth
│   │   ├── qa_mimic-iv-ecg_finetune
│   │   │   └── checkpoint
│   │   │       └── checkpoint_epoch0.pth
│   │   ├── qa_ptbxl_finetune
│   │   │   └── checkpoint
│   │   │       ├── checkpoint_epoch0.pth
│   │   ├── report_mimic-iv-ecg_finetune
│   │   │   └── checkpoint
│   │   │       └── checkpoint_epoch0.pth
│   │   ├── report_ptbxl_finetune
│   │   │   └── checkpoint
│   │   │       └── checkpoint_epoch0.pth
├── utils
│   ├── evaluation.py
│   └── tools.py
├── ds_config_zero2.json
├── eval.sh
├── finetune.sh
├── pretrain.sh
├── PROJECT_TREE.txt
├── requirements.txt
├── run_eval.py
└── run_main.py
```

## 4. Training Pipeline

HeartLLM follows a three-stage training pipeline:

Stage 1: ECG Tokenizer Training

Train the discretized ECG tokenizer:

```bash
bash ecg_tokenizer/run_tokenizer.sh
```

Stage 2: Pretraining

Pretrain the HeartLLM model with aligned ECG tokens and text:

```bash
bash pretrain.sh
```

Stage 3: Downstream Instruction Fine-Tuning

Perform instruction fine-tuning for downstream tasks (ECG-QA and report generation):

```bash
bash finetune.sh
```

## 5. Evaluation
The Google Drive folder provides pretrained model checkpoints. By running following command, you can directly load the pretrained models and perform evaluation without additional training.
```bash
bash eval.sh
```
