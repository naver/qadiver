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


import numpy as np
import torch
import torch.nn.functional as F
from copy import deepcopy
from collections import Counter
import nltk
from toy_model.model import DocQA

class QAModel:
    def __init__(self, checkpoint, model="DocQA"):
        checkpoint = torch.load(checkpoint,map_location=lambda storage, loc: storage)
        self.checkpoint = checkpoint
        config = checkpoint['config']
        vocab = checkpoint['vocab']
        char_vocab = checkpoint['char_vocab']
        self.model = DocQA(len(vocab),config.w_embed,len(char_vocab),
                           config.c_embed,config.d_model,
                           config.dropout,vocab['<pad>'])
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()
        self.vocab = vocab
        self.char_vocab = char_vocab

    def _normalize(self, text):
        replace_pair = [('``', '"'), ("''", '"'), ('\t', ' '), ("‘", "'"), ("’", "'"), ("''", '"')]
        for pair in replace_pair:
            text = text.replace(pair[0], pair[1])
        return text

    def tokenize(self,text):
        def word_tokenize(text):
            tokens = [token.replace("``", '"').replace("''", '"') for token in nltk.word_tokenize(text)]
            return tokens
        text = self._normalize(text)
        return word_tokenize(text)

    def answer_select(self, answer_data):
        if len(answer_data) == 0:
            return 0, 0, 0
        
        answer_spans = [(a['answer_start'], a['answer_start'] + len(a['text'])) for a in answer_data]
        answer_counter = Counter(answer_spans)
        value = answer_counter.most_common(1)[0][0]
        index = answer_spans.index(value)
        return value[0], value[1], index
   
    def word_answer(self, context, context_tokens, answer_data):
        def _tokens2idxs(text, tokens):
            idxs = []
            idx = 0
            for token in tokens:
                idx = text.find(token, idx)
                assert idx >= 0, (text, tokens, token)
                idxs.append(idx)
                idx += len(token)
            return idxs

        def answer_select(answer_data):
            if len(answer_data) == 0:
                return 0, 0, 0
            
            answer_spans = [(a['answer_start'], a['answer_start'] + len(a['text'])) for a in answer_data]
            answer_counter = Counter(answer_spans)
            value = answer_counter.most_common(1)[0][0]
            index = answer_spans.index(value)
            return value[0], value[1], index
     
        context = self._normalize(context)
        char_idxs = _tokens2idxs(context, context_tokens)
        answer_start, answer_end, _ = answer_select(answer_data)
        word_answer_start, word_answer_end = 0, 0

        for word_idx, char_idx in enumerate(char_idxs):
            if char_idx <= answer_start:
                word_answer_start = word_idx
            if char_idx < answer_end:
                word_answer_end = word_idx
        return word_answer_start, word_answer_end
 
    def prepare_input(self, context, query):
        def numericalize(seq,vocab,lower=False,max_len=None):
            if lower: seq = [s.lower() for s in seq]
            idx = [vocab[s] if vocab.get(s) else vocab['<unk>'] for s in seq]
            length = len(idx)
            if max_len:
                if length < max_len:
                    idx.extend([vocab['<pad>']]*(max_len-length))
                else:
                    idx = idx[:max_len]
            return torch.LongTensor(idx)
     
        context = self.tokenize(context)
        query = self.tokenize(query)
        
        context_ = numericalize(context,self.vocab,True).unsqueeze(0)
        query_ = numericalize(query,self.vocab,True).unsqueeze(0)
        s = [numericalize(c,self.char_vocab,max_len=16).view(1,-1)\
             for c in context]
        context_c = torch.cat(s).unsqueeze(0)
        s = [numericalize(q,self.char_vocab,max_len=16).view(1,-1)\
             for q in query]
        query_c = torch.cat(s).unsqueeze(0)
        return context_,query_,context_c,query_c

    def embedding(self, word, char_max=16, sparse=False):
        def check_vocab(word, vocab, lower=True):
            if lower:
                word = word.lower()
            if word in vocab:
                return vocab[word]
            else:
                return vocab["<unk>"]

        word_v = torch.tensor([[check_vocab(word, self.vocab)]])
        char_ = np.array([check_vocab(c, self.char_vocab, False) for c in word])
        char__ = np.zeros(char_max, dtype=int)
        char__[:min(char_max, char_.shape[0])] = char_[:char_max]
        #print(char__)
        char_v = torch.tensor([[char__]])
        sparse = torch.tensor([[[float(sparse)]]]).float()

        embedding = self.model.embedding(word_v, char_v)
        #print(word, embedding.size())
        return embedding[0][0].detach().numpy()

    def attention(self,context,query):
        C,Q,Cc,Qc = self.prepare_input(context,query)
        with torch.no_grad():
            C2Q, Q2C = self.model.attention(C,Q,Cc,Qc)

        return C2Q[0].tolist() #, Q2C[0][0].tolist()

    def forward(self,context,query):
        C,Q,Cc,Qc = self.prepare_input(context,query)
        Cr,Qr = self.tokenize(context), self.tokenize(query)
        with torch.no_grad():
            p1,p2 = self.model(C,Q,Cc,Qc)
            p1 = F.softmax(p1,-1)
            p2 = F.softmax(p2,-1)
        p1_prob = p1.detach().numpy()[0]
        p2_prob = p2.detach().numpy()[0]
        return p1_prob, p2_prob

    def __call__(self,context,query):
        def make_answer(context,start,end,max_len=17):
            scores = torch.ger(start[0], end[0])
            scores.triu_().tril_(max_len - 1)
            # Take argmax or top n
            scores = scores.numpy()
            scores_flat = scores.flatten()
            idx_sort = [np.argmax(scores_flat)]
            s_idx, e_idx = np.unravel_index(idx_sort, scores.shape)
            s_idx, e_idx = s_idx[0], e_idx[0]
            answer = context[0][s_idx:e_idx+1]
            return answer

        C,Q,Cc,Qc = self.prepare_input(context,query)
        Cr,Qr = self.tokenize(context),self.tokenize(query)
        with torch.no_grad():
            p1,p2 = self.model(C,Q,Cc,Qc)
            p1 = F.softmax(p1,-1)
            p2 = F.softmax(p2,-1)
            p_s = p1[:,:-1]
            p_e = p2[:,:-1]
            na = p1[:,-1]*p2[:,-1]
        
        answer = make_answer(Cr,p_s,p_e)
        return answer, na[0].item() # return answerable
    
if __name__ == "__main__":
    context = "A reusable launch system (RLS, or reusable launch vehicle, RLV) is a launch system which is capable of launching a payload into space more than once. This contrasts with expendable launch systems, where each launch vehicle is launched once and then discarded. No completely reusable orbital launch system has ever been created. Two partially reusable launch systems were developed, the Space Shuttle and Falcon 9. The Space Shuttle was partially reusable: the orbiter (which included the Space Shuttle main engines and the Orbital Maneuvering System engines), and the two solid rocket boosters were reused after several months of refitting work for each launch. The external tank was discarded after each flight."
    question = "How many partially reusable launch systems were developed?"

    #model = QAModel('toy_model/checkpoint/bidaf/model', "BidAF")
    model = QAModel('toy_model/checkpoint/docqa/model', "DocQA")
    print("model is loaded")

    answer,na_prob = model(context, question)
    print(answer, na_prob)

    print(model.embedding("test").shape)
    print(model.attention(context, question)[0].shape)
