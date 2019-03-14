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


import os, sys
sys.path.insert(0, "./")

import json
from collections import Counter, defaultdict
import numpy as np
from nltk.tokenize import sent_tokenize
from tools.sentence.vector import QuestionFeature
from tools.sentence.encode import QuestionVector

dir_path = os.path.dirname(os.path.realpath(__file__))

if __name__ == "__main__":
    word_count = Counter()
    qa_info = {}

    prefix_count = Counter()
    first_count = Counter()

    vector_sum = defaultdict(lambda: [])
    first_sum = defaultdict(lambda: [])

    with open("data/dev-v2.0.json") as f:
        data = json.load(f)["data"]
        for para in data:
            p = para["paragraphs"]
            for ctx in p:
                context = ctx["context"]
                qas = ctx["qas"]
                for qa in qas:
                    qid = qa["id"]
                    if "plausible_answers" in qa:
                        ans_key = "plausible_answers"
                        ua = 1
                    else:
                        ans_key = "answers"
                        ua = 0

                    question = qa["question"].strip().rstrip("?")
                    qa_info[qid] = {"context": context, "question": question}

                    if len(qa[ans_key]) > 0:
                        answer_text = qa[ans_key][0]['text']
                        answer_loc = qa[ans_key][0]['answer_start']
                    else:
                        answer_text = ""
                        answer_loc = 0

                    q_word_set = set(question.split(" "))
                    for word in q_word_set:
                        word = word.lower()
                        if len(word) > 0:
                            word_count[word] += 1

                    qa_info[qid]["answer"] = answer_text
                    qa_info[qid]["answer_l"] = answer_text.lower()
                    qa_info[qid]["answer_loc"] = answer_loc
                    qa_info[qid]["ua"] = ua

    print(len(qa_info))
    
    question_common_word = [v[0] for v in word_count.most_common(60)]
    
    with open("tools/sentence/qdata/common_word.json", "w") as f:
        json.dump(question_common_word, f)

    feature_data = {}
    qf = QuestionFeature()    
    for n, key in enumerate(qa_info):
        
        item = qa_info[key]
        question = item["question"]
        context = item["context"]
        answer = item["answer"]
        answer_l = item["answer_l"]
        # answer_loc = item["answer_loc"]
        ua = item["ua"]
        
        features = qf.question_features(question, context, answer)

        feature_data[key] = {}
        feature_data[key]["question"] = question
        feature_data[key]["feature"] = features

        prefix = "_".join(question.split(" ")[:2]).strip(",")
        first = prefix.split("_")[0]

        prefix_count[prefix] += 1
        first_count[first] += 1
        vector_sum[prefix].append(features[:4])
        first_sum[first].append(features[:4])

        if n % 1000 == 0:
            print(n)

    common_prefix = [v[0] for v in prefix_count.most_common(1000)]
    common_first = [v[0] for v in first_count.most_common(300)]

    common_prefix_vect = {}
    common_first_vect = {}

    for item in common_first:
        common_first_vect[item] = np.mean(first_sum[item], 0).tolist()

    for item in common_prefix:
        common_prefix_vect[item] = np.mean(vector_sum[item], 0).tolist()

    with open("tools/sentence/qdata/common_prefix_vector.json", "w") as f:
        json.dump(common_prefix_vect, f)

    with open("tools/sentence/qdata/common_first_vector.json", "w") as f:
        json.dump(common_first_vect, f)

    # generate sentence vector
    print("generate sentence vector")
    qv = QuestionVector("tools/sentence")
    qv_size = qv.get_vector_size()

    with open("tools/sentence/question_vector.txt", "w") as f:            
        for n, key in enumerate(qa_info):
            item = qa_info[key]
            question = item["question"]
            context = item["context"]
            answer = item["answer"]
            answer_l = item["answer_l"]
            ua = item["ua"]
            
            features = qv.get_question_vector(question, context, answer)
            f.write("{}\t{}\n".format(key, " ".join([str(round(v, 5)) for v in features])))
                        
            if n % 1000 == 0:
                print(n)
