import nltk
from nltk.translate.bleu_score import SmoothingFunction,sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
import concurrent.futures
nltk.data.path.append('.')

def compute_batch_meteor_rouge(data_batch):
    meteor_scores = []
    rouge_scores_list = []
    bleu1_scores = []
    bleu2_scores = []
    bleu3_scores = []
    bleu4_scores = []
    rouge = Rouge()
    for pred, gt in data_batch:
        # 计算每个句子的 Meteor 得分
        m_score = meteor_score([gt.split()], pred.split())
        meteor_scores.append(m_score)
        smoothing_fn = SmoothingFunction().method1
        reference = [gt.split()]
        hypothesis = pred.split()
        bleu1 = sentence_bleu(reference, hypothesis, weights=(1, 0, 0, 0), smoothing_function=smoothing_fn)
        bleu2 = sentence_bleu(reference, hypothesis, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing_fn)
        bleu3 = sentence_bleu(reference, hypothesis, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing_fn)
        bleu4 = sentence_bleu(reference, hypothesis, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing_fn)
        bleu1_scores.append(bleu1)
        bleu2_scores.append(bleu2)
        bleu3_scores.append(bleu3)
        bleu4_scores.append(bleu4)

        try:
            r_score = rouge.get_scores(pred, gt)[0]
        except ValueError as e:
            r_score = {
                "rouge-1": {"p": 0.0, "r": 0.0, "f": 0.0},
                "rouge-2": {"p": 0.0, "r": 0.0, "f": 0.0},
                "rouge-l": {"p": 0.0, "r": 0.0, "f": 0.0}
            }
        rouge_scores_list.append(r_score)
    return meteor_scores, rouge_scores_list,bleu1_scores,bleu2_scores,bleu3_scores,bleu4_scores

def compute_overall_metrics(data, num_workers=8, batch_size=1000):
    batches = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
    all_meteor_scores = []
    all_rouge_scores_list = []
    all_bleu1_scores = []
    all_bleu2_scores = []
    all_bleu3_scores = []
    all_bleu4_scores = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = executor.map(compute_batch_meteor_rouge, batches)

    for meteor_scores, rouge_scores_list, bleu1_scores,bleu2_scores,bleu3_scores,bleu4_scores in results:
        all_meteor_scores.extend(meteor_scores)
        all_rouge_scores_list.extend(rouge_scores_list)
        all_bleu1_scores.extend(bleu1_scores)
        all_bleu2_scores.extend(bleu2_scores)
        all_bleu3_scores.extend(bleu3_scores)
        all_bleu4_scores.extend(bleu4_scores)

    overall_meteor = sum(all_meteor_scores) / len(all_meteor_scores) if all_meteor_scores else 0.0

    overall_rouge = {}
    if all_rouge_scores_list:
        keys = all_rouge_scores_list[0].keys()  # "rouge-1", "rouge-2", "rouge-l"
        for key in keys:
            p_sum = sum(score[key]['p'] for score in all_rouge_scores_list)
            r_sum = sum(score[key]['r'] for score in all_rouge_scores_list)
            f_sum = sum(score[key]['f'] for score in all_rouge_scores_list)
            count = len(all_rouge_scores_list)
            overall_rouge[key] = {"p": p_sum / count, "r": r_sum / count, "f": f_sum / count}

    overall_bleu1 = sum(all_bleu1_scores) / len(all_bleu1_scores) if all_bleu1_scores else 0
    overall_bleu2 = sum(all_bleu2_scores) / len(all_bleu2_scores) if all_bleu2_scores else 0
    overall_bleu3 = sum(all_bleu3_scores) / len(all_bleu3_scores) if all_bleu3_scores else 0
    overall_bleu4 = sum(all_bleu4_scores) / len(all_bleu4_scores) if all_bleu4_scores else 0

    overall_metrics = {
        "bleu": {"bleu1": overall_bleu1, "bleu2": overall_bleu2, "bleu3": overall_bleu3, "bleu4": overall_bleu4},
        "meteor": overall_meteor,
        "rouge": overall_rouge
    }
    return overall_metrics


if __name__ == "__main__":
    data = [
        ("The cat is on the mat", "The cat is sitting on the mat") for i in range(10)
    ]

    metrics = compute_overall_metrics(data, num_workers=8, batch_size=1000)
    print("Overall Metrics:")
    print("BLEU:", metrics["bleu"])
    print("Meteor:", metrics["meteor"])
    print("Rouge:", metrics["rouge"])