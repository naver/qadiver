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


import nltk
from collections import Counter, defaultdict

class ApplyRule:
    def __init__(self, rule_path="data/sear_rules.txt"):
        with open(rule_path) as f:
            self.predefined_list = [v.strip() for v in f]
        self.replacement = [["tag:NOUN", "tag:NN"], ["tag:VERB", "tag:VB"]]
        self.init_rule()

    def set_sear_rule(self):
        self.rules = self.predefined_list

    def insert_rule(self, text):
        self.rules = [v for v in text.split("\n") if len(v) > 0]

    def init_rule(self):
        self.rules = []
        self.prep_rule = {}

    def as_text(self):
        return "\n".join(self.rules)

    def match_pattern(self, text_pos, pattern):
        for (t, p), z in zip(text_pos, pattern[0]):
            if "tag:" in z and z not in p:
                return False
            if "tag:" not in z and t != z:
                return False
        return True

    def replace_pattern(self, text_pos, pattern):
        tag_count = Counter()
        tag_replace_table = defaultdict(lambda: "<unk>")
        word_tag_table = {}

        # tag <-> word mapping
        for t, p in zip(text_pos, pattern[0]):
            if "tag:" in p:
                if t[0] not in word_tag_table:
                    tag_count[p] += 1
                    p_id = p + str(tag_count[p])
                    tag_replace_table[p_id] = t[0]
                    word_tag_table[t[0]] = p_id
        new_text = []
        new_tag_count = Counter()
        for word in pattern[1]:
            if "tag:" in word:
                new_tag_count[word] += 1
                tag_n = min(new_tag_count[word], tag_count[word])
                p_id = word + str(tag_n)
                new_text.append(tag_replace_table[p_id])
            else:
                if word != "[empty]":
                    new_text.append(word)
        return new_text

    def apply_rules(self, text):
        candidates = []
        pattern_list = []
        for i, r in enumerate(self.rules):
            if r not in self.prep_rule:
                for item in self.replacement:
                    r = r.replace(item[0], item[1])

                row = [v.strip().split(" ") for v in r.split("=>")]
                if len(row) != 2:
                    continue
                before, after = row
                self.prep_rule[r] = [before, after]
            pattern_list.append([i, self.prep_rule[r]])

        text_token = nltk.word_tokenize(text)
        text_pos = [[v[0], "tag:"+v[1]]for v in nltk.pos_tag(text_token)]
        for i, (n, pattern) in enumerate(pattern_list):
            # print("pattern", pattern)
            flag = False
            new_text = []
            c, p_len = (0, len(pattern[0]))
            while c < (len(text_pos) - p_len + 1):
                text_span = text_pos[c:c+p_len]
                match_result = self.match_pattern(text_span, pattern)
                if match_result:
                    flag = True
                    new_text.extend(self.replace_pattern(text_span, pattern))
                    c += p_len
                else:
                    new_text.append(text_pos[c][0])
                    c += 1
            new_text.extend([v[0] for v in text_pos[len(text_pos)-p_len+1:]])
            if flag:
                candidates.append([self.rules[n], " ".join(new_text).strip()])

        return candidates


if __name__ == "__main__":
    a_rule = ApplyRule()
    a_rule.set_sear_rule()

    #text = "What is the critical path of this question ?"
    text = "What is a critical path of this question ?"
    result = a_rule.apply_rules(text)

    print(text)
    print(a_rule.as_text())
    for item in result:
        print("=>", item)

