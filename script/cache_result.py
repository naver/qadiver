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

import argparse, json
from tqdm import tqdm
from cache.cache_db import QACache

parser = argparse.ArgumentParser()
parser.add_argument('--config_file', type=str, default=None)

parser.add_argument('--cache_path', type=str, default="cache/qa_cache.db")
parser.add_argument('--data_path', type=str, default="data/processed_data.json")
parser.add_argument('--pred_path', type=str, default="toy_model/pred.json")
parser.add_argument('--no_prob_path', type=str, default="toy_model/no_prob.json")

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


cache = QACache(config.cache_path)
cache.clear_cache() # force to remove all items in the db cache

with open(config.data_path) as f:
    data = json.load(f)

with open(config.pred_path) as f:
    pred_ = json.load(f)

with open(config.no_prob_path) as f:
    no_prob_ = json.load(f)

insert_items = []

for key in tqdm(data):
    item = data[key]
    question, context = item["question"], item["context"]
    if key not in pred_:
        print(key, "not in pred_")
    if key not in no_prob_:
        print(key, "not in no_prob_")

    insert_items.append([key, question, context, pred_[key], no_prob_[key]])

print(len(insert_items))
cache.put_answer_batch(insert_items)
