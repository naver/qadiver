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


import argparse, json, random
import numpy as np
import numexpr as ne

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

from tornado.wsgi import WSGIContainer
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop 

from hashlib import md5
from inference import QAModel
from tqdm import tqdm
from copy import deepcopy
from random import shuffle

from tools.emb_faiss import Embedding
from tools.sentence.compare import SentenceLoader
from tools.apply_rule import ApplyRule

from toy_model.evaluate_v2 import normalize_answer, compute_exact, compute_f1
from cache.cache_db import QACache 

app = Flask(__name__, static_url_path='/static')

app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False
CORS(app)

vocab = {}
model_info = {}
qa_model = None
cache = QACache() 

na_threshold = 0.0
model_eval_info = {}

def cal_em_f1(answers, predict, uans):
    gold_answers = [a['text'] for a in answers if normalize_answer(a['text'])]
    if not gold_answers or uans == 1:
        gold_answers = ['']

    em = max(compute_exact(a, predict) for a in gold_answers)
    f1 = max(compute_f1(a, predict) for a in gold_answers)
    return [em, f1]

def get_ap(id_, context, question):
    c_data = cache.get_answer_prob(id_) if len(id_) > 0 else None
    if c_data is None:
        a, p = qa_model(context, question)
        if len(id_) > 0:
            cache.put_answer_prob(id_, question, context, a, p)
        return (a, p)
    else:
        return c_data

def gen_qid(q_id, context, question):
    new_id = "6" + q_id[:4] + md5((context + " " + question).encode("utf-8")).hexdigest()[:18]
    return new_id

@app.route('/')
def index():
    #return app.send_static_file('index.html')
    m = model_eval_info
    model_info = "EM {:.1f} / F1 {:.1f} / NoAns ACC {:.3f} (thres {:.3f})" \
            .format(m["exact"] * 100, m["f1"] * 100, m["noans_acc"], na_threshold)
    if int(config.use_tools) > 0:
        return render_template('index.html', model_info=model_info)
    else:
        return render_template('index_nt.html', model_info=model_info)

@app.route('/files/<path:path>')
def static_files(path):
    return app.send_static_file('files/'+path)

def retrieve_data(id_list, filter_data, use_filter=True):
    if use_filter:
        limit = filter_data["limit"]
        filter_label = filter_data["label"]
        filter_pred = filter_data["pred"]
        filter_prob_expr = filter_data["prob_expr"]
        filter_shuffle = filter_data["shuffle"]

        if limit == 0:
            return []
        elif limit < 0:
            limit = len(id_list)

    prob_cache = cache.get_uans_data()

    id_items = []
    for id_ in id_list:
        data = get_qa_data(id_)
        if id_[0] == "6":
            continue
        ua = data["unanswerable"]

        if use_filter:
            if filter_label[0] == 0 and ua == 0:
                continue
            if filter_label[1] == 0 and ua == 1:
                continue
            if id_ in prob_cache:
                prediction = prob_cache[id_] > na_threshold
                if filter_pred[0] == 0 and prediction == ua:
                    continue
                if filter_pred[1] == 0 and prediction != ua:
                    continue

        context, question = data["context"], data["question"]
        answer, na_prob = get_ap(id_, context, question)
        prediction = na_prob > na_threshold 
        name = data["name"]

        if use_filter:
            if filter_pred[0] == 0 and prediction == ua:
                continue
            if filter_pred[1] == 0 and prediction != ua:
                continue
                    
        X = np.array([na_prob])
        try: 
            if use_filter and len(filter_prob_expr) > 0 and not ne.evaluate(filter_prob_expr)[0]:
                continue
        except:
            return []

        id_items.append({"key": id_, "name": name, "q": question, "uans": data["unanswerable"], "pred": na_prob})

    if use_filter:
        if filter_shuffle:
            shuffle(id_items)
        id_items = id_items[:limit]
    return id_items

def get_qa_data(qid):
    if qid[0] == "6" and qid in squad_data_add:
        return squad_data_add[qid]
    elif qid[0] == "5" and qid in squad_data:
        return squad_data[qid]
    else:
        return None

@app.route('/ids', methods = ['POST'])
def retrieve_ids():
    data = request.json
    filter_data = data["filter"]
    limit = filter_data["limit"]
    key_list = list(squad_data.keys())
    id_items = retrieve_data(key_list, filter_data)

    result = {"data": id_items, "ratio": min(100, round(limit * 100 / len(squad_data), 2))}
    return jsonify(result)

@app.route('/ids_list', methods = ['POST'])
def retrieve_ids_range():
    data = request.json
    key_list = data["ids"]
    id_items = retrieve_data(key_list, None, False)

    result = {"data": id_items, "ratio": min(100, round(len(key_list) * 100 / len(squad_data), 2))} 
    return jsonify(result)

@app.route('/search', methods = ['POST'])
def search_ids():
    data = request.json
    query = data["query"]
    filter_data = data["filter"]
    limit = filter_data['limit']
    id_list = cache.search_ids(query, limit)
    id_items = retrieve_data(id_list, filter_data)

    result = {"data": id_items, "ratio": min(100, round(limit * 100 / len(squad_data), 2))}
    return jsonify(result)


@app.route('/qa', methods = ['GET'])
def reterieve_qa():
    result = {"error": "id not found."}
    q_id = request.args['id']
    q_data = get_qa_data(q_id)
    if q_data is not None:
        result = deepcopy(q_data)
        context, question = result["context"], result["question"]
        answer, na_prob = get_ap(q_id, context, question)
        context_oov = [i for i, word in enumerate(result["context"].split(" ")) if word.lower() not in vocab]
        context_p_oov = [i for i, word in enumerate(result["context_p"].split(" ")) if word.lower() not in vocab]
        question_oov = [i for i, word in enumerate(result["question"].split(" ")) if word.lower() not in vocab]
        question_p_oov = [i for i, word in enumerate(result["question_p"].split(" ")) if word.lower() not in vocab]
        result["pred"] = [answer,na_prob]
        result["c_oov"], result["c_oov_p"] = context_oov, context_p_oov
        result["q_oov"], result["q_oov_p"] = question_oov, question_p_oov
        result["em_f1"] = cal_em_f1(result["answers"], "" if na_prob > na_threshold else answer, result["unanswerable"])

    return jsonify(result)

@app.route('/eval', methods = ['POST'])
def evaluate_qa():
    data = request.json
    q_id = data["id"]
    context = data["context"]
    question = data["question"]
    vocab = qa_model.vocab

    q_data = get_qa_data(q_id)
    original_id = q_id if "original_id" not in q_data else q_data["original_id"]

    result = deepcopy(q_data)
    new_id = gen_qid(original_id, context, question)

    result["id"] = new_id
    result["context"] = context
    result["question"] = question
    result["context_p"] = " ".join(qa_model.tokenize(context))
    result["question_p"] = " ".join(qa_model.tokenize(question))
    result["original_id"] = original_id

    new_answers = []
    
    original_lens = [len(v) for v in q_data["context"].split(" ")]
    new_words = result["context"].split(" ")
    new_lens = [len(v) for v in new_words]
    word_diff = len(new_lens) - len(original_lens)

    for i in range(len(result["answers"])):
        answer_start = result["answers"][i]["answer_start"]
        answer_text = result["answers"][i]["text"]
        if answer_text not in result["context"]:
            continue

        len_sum = [0, 0]
        for j, ol in enumerate(original_lens):
            len_sum = [len_sum[0] + ol + 1, len_sum[1] + new_lens[j] + 1]
            if len_sum[0] >= answer_start:
                break
        for k in range(j + 1, len(new_lens)):
            if new_words[k] not in answer_text:
                len_sum[1] += new_lens[k] + 1
            else:
                break
    
        len_diff = len_sum[1] - len_sum[0]
        if len_diff != 0:
            answer_start += (len_sum[1] - len_sum[0])
            result["answer_p"] = [result["answer_p"][0] + word_diff, result["answer_p"][1] + word_diff]
        new_answers.append({"answer_start": answer_start, "text": answer_text})

    result["answers"] = new_answers
    

    orig_answer, orig_na_prob = get_ap(original_id, q_data["context"], q_data["question"])
    orig_em_f1 = cal_em_f1(result["answers"], "" if orig_na_prob > na_threshold else orig_answer, result["unanswerable"])

    result["original_em_f1"] = orig_em_f1
    squad_data_add[new_id] = result

    answer, na_prob = get_ap(new_id, context, question)  

    context_oov = [i for i, word in enumerate(result["context"].split(" ")) if word.lower() not in vocab]
    context_p_oov = [i for i, word in enumerate(result["context_p"].split(" ")) if word.lower() not in vocab]
    question_oov = [i for i, word in enumerate(result["question"].split(" ")) if word.lower() not in vocab]
    question_p_oov = [i for i, word in enumerate(result["question_p"].split(" ")) if word.lower() not in vocab]
    result["pred"] = [answer,na_prob]
    result["c_oov"], result["c_oov_p"] = context_oov, context_p_oov
    result["q_oov"], result["q_oov_p"] = question_oov, question_p_oov
    result["em_f1"] = cal_em_f1(result["answers"], "" if na_prob > na_threshold else answer, result["unanswerable"])
    
    return jsonify(result)

@app.route('/emb', methods = ['POST'])
def retrieve_emb():
    K = 30
    data = request.json
    if data["conly"]:
        domain = data["context"].split(" ")
    else:
        domain = None

    if "word1" not in data:
        return jsonify({"error": "no word info"})
    elif "word2" not in data:
        return jsonify(emb_store.get_word_info(data["word1"], domain, K))
    else:
        return jsonify(emb_store.get_pair_info(data["word1"], data["word2"], domain, K))

@app.route('/model', methods = ['POST'])
def retrieve_model():
    data = request.json
    if "q" not in data or "c" not in data:
        return jsonfiy({"error": "missing key"})

    q = data["q"]
    c = data["c"]

    p1, p2 = qa_model.forward(c,q)
    p1 = p1.tolist()
    p2 = p2.tolist()

    result = {"start_prob": p1, "end_prob": p2}

    return jsonify(result)

@app.route('/att', methods = ['POST'])
def retrieve_att():
    data = request.json
    if "q" not in data or "c" not in data:
        return jsonfiy({"error": "missing key"})

    q = data["q"]
    c = data["c"]

    attention_info = qa_model.attention(c,q)
    return jsonify(attention_info)

@app.route('/sent', methods = ['POST'])
def retrieve_sim_sent():
    data = request.json
    question, context, answer = data["question"], data["context"], data["answer"]
    result = sent_loader.find_similar_sentence(question, context, answer, config.sim_sent_num)
    scores = []
    noans_result = []
    for item in result:
        pred_a, pred_noans = get_ap(item["id"], item["context"], item["question"])
        if pred_noans > na_threshold:
            pred_a = ''

        gt = get_qa_data(item["id"])
        gt_ans, gt_noans = gt["answers"], gt["unanswerable"]
        scores.append(cal_em_f1(gt_ans, pred_a, gt_noans))
        label = int(gt_noans == int(pred_noans > na_threshold))
        noans_result.append(label)
        item["name"] = gt["name"]
        item["label"] = label
        item["answer"] = "" if gt_noans == 1 else item["answer"]
        item["pred_answer"] = "" if pred_noans > na_threshold else pred_a
        item["ua"] = gt_noans

    scores = np.mean(np.asarray(scores), axis=0).tolist()
    scores.append(np.mean(noans_result))

    sent_info = {"stat": scores, "data": result}
    return jsonify(sent_info)

@app.route('/applyrule', methods = ['POST'])
def apply_adversarial_rules():
    data = request.json
    qid, question, context, rules = data["id"], data["question"], data["context"], data["rules"]

    q_data = deepcopy(get_qa_data(qid))
    gold_answer, gold_ua = q_data["answers"], q_data["unanswerable"]
    pred_a, pred_noans = get_ap(qid, context, question)
    q_data["em_f1"] = cal_em_f1(gold_answer, "" if pred_noans > na_threshold else pred_a, gold_ua)

    a_rule.init_rule()
    use_custom = data["customrule"]
    if use_custom:
        a_rule.insert_rule(rules)
    else:
        a_rule.set_sear_rule()
    
    #print(a_rule.as_text())

    result = {"original_result": q_data, "rule_result": []}
    apply_result = a_rule.apply_rules(question)
    total_rules = len(a_rule.rules)
    applied_rules = len(apply_result)

    result["meta"] = {"total": total_rules, "matched": applied_rules}
    
    for rule, p_question in apply_result:
        new_id = gen_qid(qid, context, p_question)
        ans, na_prob = get_ap(new_id, context, p_question)
        item = {"rule": rule, "question": p_question, "answer": ans, "na_prob": na_prob, \
                "em_f1": cal_em_f1(gold_answer, "" if na_prob > na_threshold else ans, gold_ua)}
        result["rule_result"].append(item)

    return jsonify(result)


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default="config/server.json")
    
    args, unknown = parser.parse_known_args()
    parser = argparse.ArgumentParser(parents=[parser], add_help=False)

    if args.config_file is not None:
        if ".json" in args.config_file:
            with open(args.config_file) as f:
                config = json.load(f)
                parser.set_defaults(**config)
            [parser.add_argument(arg)
                 for arg in [arg for arg in unknown if arg.startswith('--')]
                 if arg.split('--')[-1] in config]

    config = parser.parse_args()
    print(config)

    squad_data = {}
    squad_data_add = {}
    with open(config.data_path) as f:
        squad_data = json.load(f)

    emb_store = Embedding(config.embedding_path)

    with open(config.vocab_path) as f:
        vocab = json.load(f)
 
    with open(config.evaluation_result) as f:
        model_eval_info = json.load(f)
        na_threshold = model_eval_info["na_threshold"]

    print("load model")
    qa_model = QAModel(config.checkpoint_path, config.model_name)
    emb_store.set_embed_function(qa_model.embedding)

    sent_loader = SentenceLoader(config.data_path)
    a_rule = ApplyRule()

    print("start server on 0.0.0.0:%d" % int(config.port))
    http_server = HTTPServer(WSGIContainer(app))
    http_server.listen(config.port)
    IOLoop.instance().start()
