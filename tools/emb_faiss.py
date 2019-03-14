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


import faiss
import numpy as np
from tqdm import tqdm
from numpy.linalg import norm
from sklearn.decomposition import PCA

class Embedding:
    def __init__(self, embed_path = "data/embedding.txt"):
        self.word_idx = {}
        self.idx_word = {}
        self.data = []
        self.embedding = None
        
        with open(embed_path,'r',encoding='utf-8') as f:
            for i, line in tqdm(enumerate(f)):
                line = line.strip().split(" ", 1)
                #print(line)
                word, vec = line
                vec = np.fromstring(vec, sep=' ')
                self.data.append(vec / norm(vec))
                self.word_idx[word] = i
                self.idx_word[i] = word
            self.data = np.asarray(self.data).astype('float32')

        self.nb, self.d = self.data.shape
        self.xb = np.ascontiguousarray(self.data[:,:self.d])
        self.index = faiss.IndexFlatIP(self.d)
        self.index.add(self.xb)

    def set_embed_function(self, func):
        self.embedding = func

    def get_emb(self, word):
        start_idx = 0
        if word in self.word_idx:
            emb = self.data[self.word_idx[word],:self.d] 
            start_idx = 1
        else:
            emb = self.embedding(word)[:self.d]
            emb = emb / norm(emb)
        return emb, start_idx

    def get_sim_words(self, word, domain=None, k=15):
        if domain is not None:
            d_words = {}
            d_idx_word = {}
            word_emb_list = []
            for i, dw in enumerate(domain):
                if dw in d_words: # or dw == word:
                    continue
                d_words[dw] = True
                d_idx_word[len(d_idx_word)] = dw
                word_emb_list.append(self.get_emb(dw)[0])
            word_emb_list = np.asarray(word_emb_list)
            d_index = faiss.IndexFlatIP(self.d)
            d_index.add(word_emb_list)
        else:
            word_emb_list = self.xb
            d_idx_word = self.idx_word
            d_index = self.index

        emb, start_idx = self.get_emb(word)
        emb = np.expand_dims(emb, axis=0)
        D, I = d_index.search(emb, k+start_idx)
        D, I = D[0], I[0]
        W = [d_idx_word[v] for v in I]

        word_list = W[start_idx:][:k]
        dist_list = D[start_idx:].tolist()[:k]

        label_list = W if start_idx == 1 else [word] + W
        _vec_sub = np.array([word_emb_list[v] for v in I])
        vector_list = _vec_sub if start_idx == 1 else np.concatenate((emb, _vec_sub), axis=0)
        vector_list = np.array(vector_list)

        pca = PCA(n_components=2)
        reduced = pca.fit_transform(vector_list).tolist()
        pca_result = {}
        pca_result["label"] = label_list
        pca_result["plot"] = reduced

        return {"word_list": word_list, "dist_list": dist_list, "pca_result": pca_result}

    def get_sim_words_extra(self, word, word2, domain=None, k=15):
        if domain is not None:
            d_words = {}
            d_idx_word = {}
            word_emb_list = []
            for i, dw in enumerate(domain):
                if dw in d_words: # or dw == word or dw == word2:
                    continue
                d_words[dw] = True
                d_idx_word[len(d_idx_word)] = dw
                word_emb_list.append(get_emb(dw)[0])
            word_emb_list = np.asarray(word_emb_list)
            d_index = faiss.IndexFlatIP(self.d)
            d_index.add(word_emb_list)
        else:
            word_emb_list = self.xb
            d_idx_word = self.idx_word
            d_index = self.index

        start_idx = 0
        emb, start_idx = self.get_emb(word)
        emb2, _ = self.get_emb(word2)

        emb = np.expand_dims(emb, axis=0)
        emb2 = np.expand_dims(emb2, axis=0)
        D, I = d_index.search(emb, k+start_idx)
        D, I = D[0], I[0]
        W = [d_idx_word[v] for v in I]

        word_list = W[start_idx:][:k]
        dist_list = D[start_idx:].tolist()[:k]

        pca = PCA(n_components=2)

        is_included = word2 in word_list
        _vec_sub = np.array([word_emb_list[v] for v in I])
        label_list = W if start_idx == 1 else [word] + W
        if not is_included:
            _vec_sub = np.concatenate((emb2, _vec_sub), axis=0)
            label_list = [word2] + label_list

        vector_list = _vec_sub if start_idx == 1 else np.concatenate((emb, _vec_sub), axis=0)
        vector_list = np.array(vector_list)

        pca = PCA(n_components=2)
        reduced = pca.fit_transform(vector_list).tolist()
        pca_result = {}
        pca_result["label"] = label_list
        pca_result["plot"] = reduced

        return {"pca_result": pca_result}

    def get_embedding(self, word):
        if word in self.word_idx:
            emb = self.data[self.word_idx[word],:self.d]
        else:
            emb = self.embedding(word)[:self.d]
        return emb

    def cosine_sim(self, a, b):
        return np.dot(a, b)/(norm(a)*norm(b))

    def get_word_info(self, word, domain=None, K=15):
        emb = self.get_embedding(word).tolist()
        sim_words = self.get_sim_words(word, domain, K)
        result = sim_words
        result["word"] = [word]
        result["embed"] = [emb]
        return result

    def get_pair_info(self, word1, word2, domain=None, K=15):
        emb1 = self.get_embedding(word1).tolist()
        emb2 = self.get_embedding(word2).tolist()
       
        result = self.get_sim_words_extra(word1, word2, domain, K)
        result["word"] = [word1, word2]
        result["embed"] = [emb1, emb2]
        result["sim"] = self.cosine_sim(emb1, emb2)
        return result


if __name__ == "__main__":
    ev = EmbeddingVector()
    print(ev.d, ev.nb)
    result = ev.get_pair_info("!!", "4!", 15)
    print(result)

    exit()

