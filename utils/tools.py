from tqdm import tqdm
import json
import torch, numpy as np
import os
from utils.evaluation import compute_overall_metrics


def vali(args, accelerator, model, vali_loader):
    total_loss = []
    model.eval()
    with torch.no_grad():
        if args.stage == "pretrain":
            for i, (batch_ecg_data, batch_report) in tqdm(enumerate(vali_loader)):
                batch_ecg_data = batch_ecg_data.to(torch.bfloat16).to(accelerator.device)
                if args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = model(args.stage, batch_ecg_data, batch_report=batch_report)
                else:
                    outputs = model(args.stage, batch_ecg_data, batch_report=batch_report)

                llm_model, pad_prompt_ids_batch, pad_attention_mask_batch, pad_labels_batch, _, loss = outputs

                outputs = llm_model(input_ids=pad_prompt_ids_batch, attention_mask=pad_attention_mask_batch,
                                    labels=pad_labels_batch)

                loss = outputs.loss

                total_loss.append(loss.detach().cpu().to(torch.float32).numpy())

        elif args.stage == "finetune":
            if args.tasktype == "qa":
                for i, (batch_ecg_data, batch_question, batch_answer, batch_background, _) in tqdm(enumerate(vali_loader)):
                    batch_ecg_data = batch_ecg_data.to(torch.bfloat16).to(accelerator.device)
                    if args.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = model(args.stage, batch_ecg_data, batch_question=batch_question, batch_answer=batch_answer, batch_background=batch_background)
                    else:
                        outputs = model(args.stage, batch_ecg_data, batch_question=batch_question, batch_answer=batch_answer, batch_background=batch_background)

                    llm_model, pad_prompt_ids_batch, pad_attention_mask_batch, pad_labels_batch, _, loss = outputs

                    outputs = llm_model(input_ids=pad_prompt_ids_batch, attention_mask=pad_attention_mask_batch, labels=pad_labels_batch)

                    loss = outputs.loss

                    total_loss.append(loss.detach().cpu().to(torch.float32).numpy())
            elif args.tasktype == "report":
                for i, (batch_ecg_data, batch_report, batch_background) in tqdm(
                        enumerate(vali_loader)):
                    batch_ecg_data = batch_ecg_data.to(torch.bfloat16).to(accelerator.device)
                    if args.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = model(args.stage, batch_ecg_data, batch_report=batch_report, batch_background=batch_background)
                    else:
                        outputs = model(args.stage, batch_ecg_data, batch_report=batch_report,
                                            batch_background=batch_background)

                    llm_model, pad_prompt_ids_batch, pad_attention_mask_batch, pad_labels_batch, _, loss = outputs

                    outputs = llm_model(input_ids=pad_prompt_ids_batch, attention_mask=pad_attention_mask_batch,
                                        labels=pad_labels_batch)

                    loss = outputs.loss

                    total_loss.append(loss.detach().cpu().to(torch.float32).numpy())

    total_loss = np.average(total_loss)
    model.train()
    return total_loss


def test(args, accelerator, model, test_loader):

    if args.tasktype == "report":
        batch_threshold = 10
        partial_predictions = []
        partial_gts = []
        output_file = os.path.join(os.path.dirname(args.ckp_path), f"predictions_report_{args.dataset}.json")
        with open(output_file, "w", encoding="utf-8") as f:
            pass

    results = []
    model.eval()
    with torch.no_grad():

        for i, data in tqdm(enumerate(test_loader)):
            if args.tasktype == "qa":
                batch_ecg_data, batch_question, batch_answer, batch_background, example_str = data
                batch_wo_answer = tuple("" for i in batch_answer)
                batch_ecg_data = batch_ecg_data.to(torch.bfloat16).to(accelerator.device)
                if args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = model(args.stage, batch_ecg_data, batch_question=batch_question, batch_answer=batch_wo_answer, batch_background=batch_background)
                else:
                    outputs = model(args.stage, batch_ecg_data, batch_question=batch_question, batch_answer=batch_wo_answer, batch_background=batch_background)

                llm_model, pad_prompt_ids_batch, pad_attention_mask_batch, pad_labels_batch, tokenizer, loss = outputs


                generate_ids = llm_model.generate(input_ids=pad_prompt_ids_batch[:, :-1],
                                                  attention_mask=pad_attention_mask_batch[:, :-1],
                                                  do_sample=False,
                                                  num_beams=1,
                                                  max_new_tokens=args.max_new_tokens)

                answer_only_ids = [
                    generate_ids[i, pad_prompt_ids_batch.shape[1]-1:] for i in range(generate_ids.shape[0])
                ]

                prediction_list = tokenizer.batch_decode(answer_only_ids,
                                                         skip_special_tokens=True,
                                                         clean_up_tokenization_spaces=False
                                                         )
                results.extend(post_proc(example_str, prediction_list))

            elif args.tasktype == "report":
                batch_ecg_data, batch_report, batch_background = data
                batch_wo_answer = tuple("" for i in batch_report)
                batch_ecg_data = batch_ecg_data.to(torch.bfloat16).to(accelerator.device)
                if args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = model(args.stage, batch_ecg_data, batch_report=batch_wo_answer,batch_background=batch_background)
                else:
                    outputs = model(args.stage, batch_ecg_data, batch_report=batch_wo_answer,batch_background=batch_background)

                llm_model, pad_prompt_ids_batch, pad_attention_mask_batch, pad_labels_batch, tokenizer, loss = outputs

                generate_ids = llm_model.generate(input_ids=pad_prompt_ids_batch[:, :-1],
                                                  attention_mask=pad_attention_mask_batch[:, :-1],
                                                  do_sample=False,
                                                  num_beams=1,
                                                  max_new_tokens=args.max_new_tokens)


                answer_only_ids = generate_ids[:, pad_prompt_ids_batch.shape[1] - 1:]  # [B, L]
                gathered_preds = accelerator.gather_for_metrics(answer_only_ids)  # [B_total, L]

                gathered_gts = list(batch_report)


                if accelerator.is_main_process:
                    prediction_list = tokenizer.batch_decode(
                        gathered_preds,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False
                    )

                    partial_predictions.extend(prediction_list)
                    partial_gts.extend(gathered_gts)

                    if (i + 1) % batch_threshold == 0:
                        with open(output_file, "a", encoding="utf-8") as f:
                            for pred, gt in zip(partial_predictions, partial_gts):
                                json.dump({"prediction": pred, "gt": gt}, f, ensure_ascii=False)
                                f.write("\n")
                        partial_predictions.clear()
                        partial_gts.clear()


    if args.tasktype == "report":
        if partial_predictions:
            if accelerator.is_main_process:
                with open(output_file, "a", encoding="utf-8") as f:
                    for pred, gt in zip(partial_predictions, partial_gts):
                        record = {"prediction": pred, "gt": gt}
                        json.dump(record, f, ensure_ascii=False)
                        f.write("\n")

        results = []
        if accelerator.is_main_process:
            with open(output_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        l = json.loads(line)
                        results.append((l["prediction"], l["gt"]))
            metrics = compute_overall_metrics(results, num_workers=args.num_workers, batch_size=1000)
            print("Overall Metrics:")
            print("BLEU:", metrics["bleu"])
            print("Meteor:", metrics["meteor"])
            print("Rouge:", metrics["rouge"])

    elif args.tasktype == "qa":

        correct,total,em_acc = cal_acc_qtype(results)
        tasks = ['single-verify', "single-query",  "single-choose"]
        num_task = len(tasks)
        correct_numerical = [0]*num_task
        total_numerical = [0]*num_task

        for taskid,task in enumerate(tasks):
            correct_numerical[taskid] += correct[task]
            total_numerical[taskid] += total[task]

        output_file = os.path.join(os.path.dirname(args.ckp_path), f"predictions_qa_{args.dataset}.json")

        num_gpus = accelerator.num_processes
        if num_gpus > 1:
            gathered_correct = accelerator.gather(torch.tensor(correct_numerical, device=accelerator.device))
            gathered_total = accelerator.gather(torch.tensor(total_numerical, device=accelerator.device))

            if accelerator.is_main_process:
                sum_correct = gathered_correct.view(num_gpus, -1).sum(dim=0)
                sum_total = gathered_total.view(num_gpus, -1).sum(dim=0)

                for idx, (c,t) in enumerate(zip(sum_correct,sum_total)):
                    tasktype = tasks[idx]
                    print(f"{tasktype} correct num: {c}, total num: {t}, ","em_acc rate:{:.2f}".format(c/t*100))

                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
        else:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

    model.train()
    return results


def post_proc_cls(example_str, prediction_list,batch_labels):
    results = []
    for idx,item in enumerate(example_str):
        item = json.loads(f"[{item}]")[0]
        item["prediction"] = prediction_list[idx]
        item["label"] = batch_labels[idx]
        results.append(item)
    return results

def cal_acc_qtype_cls(j):
    correct = {'single-verify': 0,"single-query": 0,  "single-choose": 0, "comparison_irrelevant-verify": 0, "comparison_consecutive-verify": 0}
    total = {'single-verify': 0,"single-query": 0,  "single-choose": 0, "comparison_irrelevant-verify": 0, "comparison_consecutive-verify": 0}
    em_acc = {'single-verify': 0, "single-query": 0, "single-choose": 0, "comparison_irrelevant-verify": 0,"comparison_consecutive-verify": 0}
    for i in j:
        if set(i["prediction"]) == set(i["label"]):
            correct[i["question_type"]] += 1
        total[i["question_type"]] += 1
    print("Correct Stat:", correct)
    print("Total Stat:", total)
    print("Acc Ratio:")
    for k in total:
        if total[k] !=0:
            ratio = correct[k]/total[k] * 100
            print(k,":{:.2f}".format(ratio))
            em_acc[k] = ratio
    return correct,total,em_acc


def post_proc(example_str, prediction_list):
    results = []
    for item, prediction in zip(example_str, prediction_list):
        ans = prediction.strip()
        item = json.loads(f"[{item}]")[0]
        item["prediction"] = ans.split(".")
        results.append(item)
    return results

def cal_acc_qtype(j):
    correct = {'single-verify': 0,"single-query": 0,  "single-choose": 0, "comparison_irrelevant-verify": 0, "comparison_consecutive-verify": 0}
    total = {'single-verify': 0,"single-query": 0,  "single-choose": 0, "comparison_irrelevant-verify": 0, "comparison_consecutive-verify": 0}
    em_acc = {'single-verify': 0, "single-query": 0, "single-choose": 0, "comparison_irrelevant-verify": 0, "comparison_consecutive-verify": 0}
    for i in j:
        if set(i["prediction"]) == set(i["answer"]):
            correct[i["question_type"]] += 1
        total[i["question_type"]] += 1
    print("Correct Stat:", correct)
    print("Total Stat:", total)
    print("Acc Ratio:")
    for k in total:
        if total[k] !=0:
            ratio = correct[k]/total[k] * 100
            print(k,":{:.2f}".format(ratio))
            em_acc[k] = ratio
    return correct,total,em_acc

def cal_acc_atype(j):
    correct = {"scp_code":0,"numeric_feature":0,"heart_axis":0,"stage_of_infarction":0,"noise":0,"extra_systole":0}
    total = {"scp_code":0,"numeric_feature":0,"heart_axis":0,"stage_of_infarction":0,"noise":0,"extra_systole":0}
    for i in j:
        if set(i["prediction"]) == set(i["answer"]):
            correct[i["attribute_type"]] += 1
        total[i["attribute_type"]] += 1
    print(correct)
    print(total)
    for k in total:
        if total[k] != 0:
            print(k,":{:.2f}".format(correct[k]/total[k] * 100))

def cal_detail_acc(j):
    correct = {'single-verify': {"scp_code":0,"numeric_feature":0,"heart_axis":0,"stage_of_infarction":0,"noise":0,"extra_systole":0},
               'single-query': {"scp_code":0,"numeric_feature":0,"heart_axis":0,"stage_of_infarction":0,"noise":0,"extra_systole":0},
               "single-choose": {"scp_code":0,"numeric_feature":0,"heart_axis":0,"stage_of_infarction":0,"noise":0,"extra_systole":0}}
    total = {'single-verify': {"scp_code":0,"numeric_feature":0,"heart_axis":0,"stage_of_infarction":0,"noise":0,"extra_systole":0},
               'single-query': {"scp_code":0,"numeric_feature":0,"heart_axis":0,"stage_of_infarction":0,"noise":0,"extra_systole":0},
               "single-choose": {"scp_code":0,"numeric_feature":0,"heart_axis":0,"stage_of_infarction":0,"noise":0,"extra_systole":0}}
    for i in j:
        if set(i["prediction"]) == set(i["answer"]):
            correct[i["question_type"]][i["attribute_type"]] += 1
        total[i["question_type"]][i["attribute_type"]] += 1
    print("=" * 50)
    print(correct)
    print(total)
    for k1 in total:
        print("="*50)
        print(k1)
        for k2 in total[k1]:
            if total[k1][k2] != 0:
                print(k2,":{:.2f}".format(correct[k1][k2]/total[k1][k2] * 100))
            else:
                print("None")