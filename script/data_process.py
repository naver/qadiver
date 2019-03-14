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
from tqdm import tqdm
import argparse, sys

sys.path.insert(0, "./")
from inference import QAModel

word_idx = {}
def get_words(tokens):
    for t in tokens:
        t = t.strip()
        if len(t) > 0 and t not in word_idx:
            word_idx[t] = len(word_idx)

def convert_squad_to_list(qa_model, filename="./dev-v2.0.json"):
    squad_data = {}
    word_data = {}
    tokenize = qa_model.tokenize
    word_answer = qa_model.word_answer
   
    with open(filename) as f:
        raw_data = json.load(f)
    for article in tqdm(raw_data['data'], desc=None):
        doc_name = article["title"]
        context_count = 1
        for paragraph in article['paragraphs']:
            context = paragraph['context'].replace("\n", " ")
            context_tokens = tokenize(context)

            for qa in paragraph['qas']:
                qid = qa['id']
                squad_data[qid] = {}
                squad_data[qid]['question'] = qa['question']
                question_tokens = tokenize(qa['question'])
                squad_data[qid]['question_p'] = " ".join(question_tokens)
                squad_data[qid]['context'] = context
                squad_data[qid]['context_p'] = " ".join(context_tokens)
                squad_data[qid]['name'] = doc_name + "_" + str(context_count)
                impossible = qa['is_impossible']
                squad_data[qid]['unanswerable'] = int(impossible)
                if impossible:
                    squad_data[qid]['answers'] = qa['plausible_answers']
                else:
                    squad_data[qid]['answers'] = qa['answers']

                squad_data[qid]["answer_p"] = word_answer(context, context_tokens, squad_data[qid]['answers']) 

                get_words(qa['question'].split(" "))
                get_words(context.split(" ")) 
                get_words(question_tokens)
                get_words(context_tokens)

                context_count += 1

    return squad_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default=None)

    parser.add_argument('--filename', type=str, default='data/dev-v2.0.json')
    parser.add_argument('--output_path', type=str, default='data/processed_data.json')
    parser.add_argument('--model_name', type=str, default='DocQA')
    parser.add_argument('--checkpoint_path', type=str, default='toy_model/checkpoint/docqa/model')
    parser.add_argument('--extract_embedding', type=int, default=1)
    parser.add_argument('--embedding_output', type=str, default='data/embedding.txt')

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

    # generate preprocess data
    qa_model = QAModel(config.checkpoint_path, config.model_name)
    result = convert_squad_to_list(qa_model, config.filename)
    with open(config.output_path, "w", encoding='utf-8') as f:
        json.dump(result, f)

    if config.extract_embedding:
        # build vocab
        print("vocab:", len(word_idx))
        embedding = qa_model.embedding

        with open(config.embedding_output, "w", encoding='utf-8') as f:
            for word in tqdm(word_idx):
                if len(word.strip()) == 0:
                    continue
                emb_in = embedding(word, sparse=False)
                f.write(word + " " + " ".join([str(round(v, 5)) for v in emb_in]) + "\n")

