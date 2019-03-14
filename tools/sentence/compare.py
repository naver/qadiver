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
dir_path = os.path.dirname(os.path.realpath(__file__))

import json
import numpy as np
import faiss
from sentence.encode import QuestionVector #get_question_vector
from operator import itemgetter

import os
dir_path = os.path.dirname(os.path.realpath(__file__))

class SentenceLoader:
    def __init__(self, qa_list_txt, feature_txt=dir_path + "/question_vector.txt"):
        self.qa_data = {}
        self.qid_list = []
        self.features = []
        self.qv = QuestionVector(dir_path)
        self.d = self.qv.get_vector_size()

        with open(qa_list_txt) as f:
            data = json.load(f)
            for qid in data:
                item = data[qid]
                q, c = item["question"], item["context"]
                a = item["answers"][0]["text"] if len(item["answers"]) > 0 else ""
                self.qa_data[qid] = {"id":qid, "question": q, "context": c, "answer": a}

        with open(feature_txt) as f:
            for i, line in enumerate(f):
                qid, feature = line.strip().split("\t", 1)
                self.qid_list.append(qid)
                feature = np.fromstring(feature, sep=' ')
                feature = feature[:self.d] 
                self.features.append(feature)

        self.features = np.asarray(self.features).astype(np.float32)
        self.qid_list = np.asarray(self.qid_list)

        self.index = faiss.IndexFlatL2(self.d)
        self.index.add(self.features)

    def find_similar_sentence(self, question, context, answer, K=10):
        question = question.rstrip("?")
        question_vector = self.qv.get_question_vector(question, context, answer)
        q_input = np.asarray([question_vector[:self.d]]).astype("float32")

        D, I = self.index.search(q_input, K)
        D, I = D[0], I[0]

        ret_list = []
        retrieved = self.qid_list[I]
        for i, ret in enumerate(retrieved):
            sim_qa = self.qa_data[ret]
            ret_list.append((self.qa_data[ret], i)) 
        ret_list = [v[0] for v in sorted(ret_list, key=itemgetter(1), reverse=False)]

        return ret_list

if __name__ == "__main__":
    question = "What is one example of an instance that the qualitative answer to the traveling salesman fails to answer?"
    context = "To further highlight the difference between a problem and an instance, consider the following instance of the decision version of the traveling salesman problem: Is there a route of at most 2000 kilometres passing through all of Germany's 15 largest cities? The quantitative answer to this particular problem instance is of little use for solving other instances of the problem, such as asking for a round trip through all sites in Milan whose total length is at most 10 km. For this reason, complexity theory addresses computational problems and not particular problem instances."
    answer = "round trip through all sites in Milan"

    sent_loader = SentenceLoader(70)
    result = sent_loader.find_similar_sentence(question, context, answer, 20)
    print(question)
    for ret in result:
        print(ret)

