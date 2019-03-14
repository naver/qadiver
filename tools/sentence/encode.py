# -*- coding: utf-8 -*-
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
import numpy as np
import json
from sentence.vector import QuestionFeature

dir_path = os.path.dirname(os.path.realpath(__file__))

class QuestionVector:
    def __init__(self, dir_path):
        with open(dir_path + "/qdata/common_prefix_vector.json") as f:
            self.prefix = json.load(f)
            
        with open(dir_path + "/qdata/common_first_vector.json") as f:
            self.first = json.load(f)

        self.qf = QuestionFeature(dir_path + "/qdata/common_word.json")

    def get_question_stat_vector(self, question):
        words = question.split(" ")
        if words[0] == "In":
            words.pop(0)
            words[0] = words[0].title()

        prefix = "_".join(words[:2]).strip(",")
        first = prefix.split("_")[0]    
        if prefix in self.prefix:
            return self.prefix[prefix]
        elif first in self.first:
            return self.first[first]    
        else:
            return np.zeros(4)
        
    def get_question_vector(self, question, context, answer):
        prefix_v = self.get_question_stat_vector(question)
        question_v = self.qf.question_features(question, context, answer)
        question_v = np.asarray(question_v) * (0.25)
        return np.concatenate((prefix_v, question_v), 0)

    def get_vector_size(self):
        return 13

