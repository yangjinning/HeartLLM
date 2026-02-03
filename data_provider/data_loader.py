import os
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import warnings
import json
import random
import wfdb


warnings.filterwarnings('ignore')

class Dataset_ECG_Background(Dataset):
    def __init__(self, root_path_ecg, path_json,flag,shuffle_flag,num_data=None):

        self.root_path_ecg = root_path_ecg
        assert flag in ['train', 'test', 'valid']
        type_map = {'train': 0, 'valid': 1, 'test': 2}
        self.set_type = type_map[flag]

        if shuffle_flag:
            random.seed(42)

        if num_data:
            self.json_data = self.__load_json__(path_json,shuffle_flag)[:num_data]
        else:
            self.json_data = self.__load_json__(path_json,shuffle_flag)

        self.scaler = StandardScaler()

    def __load_json__(self,path_json, shuffle_flag):
        with open(path_json, "r") as f:
            records = json.load(f)
        if shuffle_flag:
            random.shuffle(records)
        return records

    def __load_ecg_data__(self, root_path_ecg, filepath):
        data = wfdb.rdsamp(os.path.join(root_path_ecg, filepath))
        signal, meta = data
        return np.array(signal)

    def __getitem__(self, index):
        example = self.json_data[index]
        example_str = json.dumps(example)
        ecg_data = self.__load_ecg_data__(self.root_path_ecg, example["ecg_path"])

        self.scaler.fit(ecg_data)
        ecg_data = self.scaler.transform(ecg_data)

        question = example["question"]
        background = example["background"]
        answer_text = ".".join(example["answer"])

        return ecg_data,question,answer_text,background,example_str

    def __len__(self):
        return  len(self.json_data)

class Dataset_ECG_Report(Dataset):
    def __init__(self, configs,root_path_ecg, path_report_json, mode,shuffle,num_data=None):
        type_map = {'train': 0, 'valid': 1, 'test': 2}
        self.set_type = type_map[mode]
        self.path_json = path_report_json
        self.root_path_ecg = root_path_ecg
        self.configs = configs

        if shuffle:
            random.seed(42)

        if num_data:
            self.json_data = self.__load_json__(self.path_json,shuffle)[:num_data]
        else:
            self.json_data = self.__load_json__(self.path_json,shuffle)

        self.scaler = StandardScaler()

    def __load_json__(self,path_json, shuffle_flag):
        with open(path_json, "r") as f:
            records = json.load(f)
        if shuffle_flag:
            random.shuffle(records)
        return records

    def __load_ecg_data__(self, root_path_ecg, filepath):
        data = wfdb.rdsamp(os.path.join(root_path_ecg, filepath))
        signal, meta = data
        return np.array(signal)

    def __getitem__(self, index):
        example = self.json_data[index]
        ecg_data = self.__load_ecg_data__(self.root_path_ecg, example["filepath"])
        self.scaler.fit(ecg_data)
        ecg_data = self.scaler.transform(ecg_data)
        report = example["report"]
        return ecg_data,report

    def __len__(self):
        return  len(self.json_data)

class Dataset_ECG_Report_BG(Dataset):
    def __init__(self, root_path_ecg, path_report_json, mode,shuffle,num_data=None):
        type_map = {'train': 0, 'valid': 1, 'test': 2}
        self.set_type = type_map[mode]
        self.path_json = path_report_json
        self.root_path_ecg = root_path_ecg

        if shuffle:
            random.seed(42)

        if num_data:
            self.json_data = self.__load_json__(self.path_json,shuffle)[:num_data]
        else:
            self.json_data = self.__load_json__(self.path_json,shuffle)

        self.scaler = StandardScaler()  # 使其均值为 0，标准差为 1

    def __load_json__(self,path_json, shuffle_flag):
        with open(path_json, "r") as f:
            records = json.load(f)
        if shuffle_flag:
            random.shuffle(records)
        return records

    def __load_ecg_data__(self, root_path_ecg, filepath):
        data = wfdb.rdsamp(os.path.join(root_path_ecg, filepath))
        signal, meta = data
        return np.array(signal)

    def __getitem__(self, index):
        example = self.json_data[index]
        ecg_data = self.__load_ecg_data__(self.root_path_ecg, example["filepath"])
        self.scaler.fit(ecg_data) #归一化没有问题，均值逼近0，标准差为1
        ecg_data = self.scaler.transform(ecg_data)
        report = example["report"]
        background = example["background"]
        return ecg_data,report,background

    def __len__(self):
        return  len(self.json_data)
