import pdb
import re
import os

import nlp
import argparse
import numpy as np
import jsonlines, json
from tqdm import tqdm
from collections import defaultdict


def break_at_periods(s):
    lines = []
    prev = 0
    period_indices = [i for (i,ch) in enumerate(s) if ch=="."]
    for elem in period_indices:
        lines.append(s[prev:elem+1])
        prev=elem+1
    return lines


match_pattern = re.compile(r'.*target_\d+(_\d+).*')
match_pattern_compiled = re.compile(match_pattern)
def paraphrase_reduce(s):
    words = s.split(" ")
    new_words = []
    for w in words:
        res = match_pattern_compiled.match(w)
        if res:
            start_ind, end_ind = res.regs[1]
            new_word = w[:start_ind]+w[end_ind:]
            new_words.append(new_word)
        else:
            new_words.append(w)
    return " ".join(new_words)


def evaluate_rouge_using_huggingface(test_preds):
    rouge = nlp.load_metric('rouge')
    # all_scores = {"rouge1":[],"rouge2":[],"rougel":[], "exactmatch":[], "unordered_exactmatch":[]}
    all_scores = defaultdict(list)
    for elem in tqdm(test_preds):
        generated = elem["prediction"].lower()
        reference = elem["ground_truth"].lower()

        generated = paraphrase_reduce(generated)
        reference = paraphrase_reduce(reference)

    #     second one is ground truth
        rouge.add( prediction=generated, reference=reference)
        all_scores["exactmatch"].append(generated==reference)

        # unordered exact match
        try:
            generated_lines = break_at_periods(generated)
            generated_lines = [x.strip() for x in generated_lines]
            generated_lines = sorted(generated_lines)
            reference_lines = break_at_periods(reference)
            reference_lines = [x.strip() for x in reference_lines]
            reference_lines = sorted(reference_lines)
            all_scores["unordered_exactmatch"].append(generated_lines==reference_lines)
        except ValueError:  # can happen if there is no period generated
            all_scores["unordered_exactmatch"].append(0)


    score = rouge.compute(rouge_types=["rouge1", "rouge2", "rougeL"])

    all_scores["rouge1"].append( score['rouge1'].mid.fmeasure )
    all_scores["rouge2"].append( score['rouge2'].mid.fmeasure )
    all_scores["rougel"].append( score['rougeL'].mid.fmeasure )
    for (k,v) in all_scores.items():
        all_scores[k]=np.mean(v)

    return all_scores


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='calculate rouge scores from output file')
    parser.add_argument(
        '-prediction_file',
        dest='pred_file',
        help='Prediction file',
        type=str,
        required=True,
    )

    parser.add_argument(
        '-output_file',
        dest='out_file',
        help='Output file',
        type=str,
        required=True,
    )

    parser.add_argument(
        '-rouge_impl',
        dest='rouge_impl',
        help='implementation of rouge to use - one of  pyrouge or sumeval',
        default='huggingface',
        type=str
    )

    args = parser.parse_args()
    pred_file = args.pred_file
    out_file = args.out_file
    rouge_impl = args.rouge_impl


    exact, rouge1, rouge2, rougel, num_read = 0,0,0,0,0

    test_preds = list(jsonlines.open(pred_file))

    df_dict = evaluate_rouge_using_huggingface(test_preds)

    json.dump(df_dict, open(out_file, "w"))






