from data_provider.data_loader import Dataset_ECG_Report,Dataset_ECG_Background,Dataset_ECG_Report_BG
from torch.utils.data import DataLoader
import os

def data_provider(args, flag, num_data=None):

    root_path_ecg = args.root_path_ecg

    if args.tasktype == "qa":
        root_path_json = args.root_path_json

    if flag == "test":
        batch_size = args.batch_size
    else:
        batch_size = args.batch_size

    if args.stage == "pretrain":
        path_report_json = os.path.join(args.root_report_json, f"{flag}.json")
        data_set = Dataset_ECG_Report(
                                      configs=args,
                                      root_path_ecg=root_path_ecg,
                                      path_report_json=path_report_json,
                                      mode=flag,
                                      shuffle=True,
                                      num_data=num_data)
    elif args.stage in ["finetune","test"]:
        if args.tasktype == "qa":
            path_json = os.path.join(root_path_json, f"template_{flag}_background.json")
            data_set = Dataset_ECG_Background(root_path_ecg=root_path_ecg,
                                            path_json=path_json,
                                            flag=flag,  # train,valid,test
                                            shuffle_flag=True, #shuffle the data
                                            num_data=num_data)
        elif args.tasktype == "report":
            path_report_json = os.path.join(args.root_report_json, f"{flag}.json")
            data_set = Dataset_ECG_Report_BG(root_path_ecg=root_path_ecg,
                                          path_report_json=path_report_json,
                                          mode=flag,
                                          shuffle=True,
                                          num_data=num_data)

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        num_workers=args.num_workers,
        drop_last=False)

    return data_set, data_loader




