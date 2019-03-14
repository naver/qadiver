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


import json
from collections import Counter
import numpy as np
from nltk.tokenize import sent_tokenize

class QuestionFeature:
    def __init__(self, word_path=None):
        if word_path:
            with open(word_path) as f:
                self.question_common_word = json.load(f)
        
    def number_exist(self, s):
        return any(i.isdigit() for i in s)
    
    def captial_exist(self, s):
        return any(i.isupper() for i in s)

    def question_features(self, question, context, answer):
        answer_loc = context.index(answer)
        return self.question_features_(question, context, answer, answer_loc)
        
    def question_features_(self, question, context, answer, answer_loc):
        answer_span = (answer_loc, answer_loc + len(answer))
        
        has_number = int(self.number_exist(answer)) # number exist
        has_entity = int(self.captial_exist(answer)) # entity exist
        
        question_word = list(question.split(" "))
        context_word = list(context.split(" "))

        question_len = min(1.0, len(question_word)/15)
        answer_len = min(1.0, len(answer.split(" "))/10)
        
        matched_words = [int(word.lower() in context.lower()) for word in question_word]
        word_ratio1 = np.mean(matched_words) 
    
        matched_words2 = [int(word in context_word) for word in question_word]
        word_ratio2 = np.mean(matched_words2)

        word_distance_list = []
        word_location_list = []
        for word in question_word:
            word = word.lower()
            if word not in context.lower():
                continue
            else:  
                min_distance = len(" ".join(context_word))
                # print("c", min_distance)
                for i, c in enumerate(context_word):
                    if c.lower() == word:
                        c_index = len(" ".join(context_word[:i+1]))
                        #print(c_index, answer_span)
                        if c_index >= answer_span[0] and c_index < answer_span[1]:
                            min_distance = 0
                            break
                        else:
                            distance = min(abs(answer_span[0]-c_index), abs(answer_span[1]-c_index))
                            if min_distance > distance:
                                min_distance = distance
                                                            
                word_distance_list.append(min_distance)

        for word in question_word:
            word = word.lower()
            if word not in context.lower():
                continue
            else:
                c_index = context.lower().index(word)
                if c_index >= answer_span[0] and c_index < answer_span[1]:
                    word_location_list.append(0.5)
                elif c_index < answer_span[0]:
                    word_location_list.append(1)
                else:
                    word_location_list.append(0)

        # word level distance
        if len(word_distance_list) == 0:
            word_distance = 0
        else:
            word_distance = 1 - np.mean(word_distance_list) / len(" ".join(context_word))

        if len(word_location_list) == 0:
            word_location = 0.5
        else:
            word_location = np.mean(word_location_list)

        ## Sentence Nearest Answer Distance
        
        sent_tokenize_list = sent_tokenize(context)
        #print("\n".join(sent_tokenize_list))
        sent_lens = [len(" ".join(sent_tokenize_list[:i+1])) for i in range(len(sent_tokenize_list))]
        answer_sent_loc = sum([int(v < answer_loc) for v in sent_lens])
        
        sent_distance_list = []
        for word in question_word:
            word = word.lower()
            if word not in context.lower():
                continue
            else:
                min_distance = len(sent_tokenize_list)           
                for i, c in enumerate(sent_tokenize_list):
                    if word in c.lower():
                        if answer_sent_loc == i:
                            min_distance = 0
                            break
                        else:
                            distance = abs(answer_sent_loc - i)
                            if min_distance > distance:
                                min_distance = distance
                    
                            
                sent_distance_list.append(min_distance)
    
        # word level distance
        if len(sent_distance_list) == 0:
            sent_distance = 0
        else:
            sent_distance = 1 - np.mean(sent_distance_list) / len(sent_tokenize_list)
        #print("sent_distance", sent_distance, len(sent_tokenize_list))
        
        # common_match = [int(v in question.lower()) for v in self.question_common_word]
        
        features = [has_number, has_entity, answer_len, word_ratio1, word_ratio2, question_len, word_distance, word_location, sent_distance] # + common_match
        return features

