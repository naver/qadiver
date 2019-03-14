# Copyright 2019-present NAVER Corp.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import sys
sys.path.insert(0, "./")

import argparse, json, collections
import re, string
import numpy as np
from tqdm import tqdm

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s):
    if not s: return []
    return normalize_answer(s).split()

def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))

def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def get_raw_scores(dataset, preds, no_prob, threshold):
    exact_scores = {}
    f1_scores = {}
    noans_scores = {}
    for article in dataset:
        for p in article['paragraphs']:
            for qa in p['qas']:
                qid = qa['id']
                gold_answers = [a['text'] for a in qa['answers'] if normalize_answer(a['text'])]
                unanswerable = qa['is_impossible']
                if not gold_answers:
                    # For unanswerable questions, only correct answer is empty string
                    gold_answers = ['']
                if qid not in preds or qid not in no_prob:
                    print('Missing prediction for %s' % qid)
                    continue
                a_pred = preds[qid]
                a_noans = int(no_prob[qid] > threshold)
                a_pred = "" if a_noans else preds[qid]

                # Take max over all gold answers
                exact_scores[qid] = max(compute_exact(a, a_pred) for a in gold_answers)
                f1_scores[qid] = max(compute_f1(a, a_pred) for a in gold_answers)
                noans_scores[qid] = int(a_noans == unanswerable)
    return exact_scores, f1_scores, noans_scores


## Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default=None)

    parser.add_argument('--data_path', type=str, default="data/dev-v2.0.json")
    parser.add_argument('--pred_path', type=str, default="toy_model/checkpoint/pred.json")
    parser.add_argument('--no_prob_path', type=str, default="toy_model/checkpoint/no_prob.json")
    parser.add_argument('--na_threshold', type=int, default=0.5)

    parser.add_argument('--output', type=str, default="toy_model/eval_result.json")

    args, unknown = parser.parse_known_args()
    parser = argparse.ArgumentParser(parents=[parser], add_help=False)

    if args.config_file is not None:
        if ".json" in args.config_file:
            with open(args.config_file) as f:
                config = json.load(f)
                print("load from file:")
                print(config)
                parser.set_defaults(**config)
            [parser.add_argument(arg)
                 for arg in [arg for arg in unknown if arg.startswith('--')]
                 if arg.split('--')[-1] in config]

    config = parser.parse_args()
    print(config)

    with open(config.data_path) as f:
        data = json.load(f)['data']

    with open(config.pred_path) as f:
        pred = json.load(f)

    with open(config.no_prob_path) as f:
        no_prob = json.load(f)

    e_result, f_result, na_result = get_raw_scores(data, pred, no_prob, config.na_threshold)

    e_mean = np.mean([e_result[v] for v in e_result])
    f_mean = np.mean([f_result[v] for v in f_result])
    na_mean = np.mean([na_result[v] for v in na_result])

    eval_result = {"exact": e_mean, "f1": f_mean, "noans_acc": na_mean, "na_threshold": config.na_threshold}
    print(eval_result)
    with open(config.output, "w") as f:
        json.dump(eval_result, f)
