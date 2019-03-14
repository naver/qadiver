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


import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import sublayer
except:
    from . import sublayer

    
class QADiverModel(nn.Module):
    def __init__(self):
        super(QADiverModel,self).__init__()
    
    def embedding(self, *args):
        raise NotImplementedError('this function should return embedding used in the model for single word')
        
    def attention(self, *args):
        raise NotImplementedError('this function should return attention matrix from the model')
    
    
class InputEmbedding(nn.Module):
    def __init__(self,w_size,w_embed,
                      c_size,c_embed,
                      hidden_dim,dropout = 0.2, pad_idx=1):
        super(InputEmbedding,self).__init__()

        self.pad_idx = pad_idx
        self.c_embed = sublayer.CharCNN(c_size,20,5,c_embed,dropout,pad_idx)
        self.w_embed = nn.Embedding(w_size,w_embed,padding_idx=pad_idx)
        embed_dim = c_embed + w_embed
            
        self.contextual = nn.GRU(embed_dim,hidden_dim,1,\
                                  batch_first=True,bidirectional=True)
        self.dropout = nn.Dropout(dropout)
       
    def single(self,words,chars):
        mask = words.eq(self.pad_idx).unsqueeze(-1) # B,T,1
        e_w = self.w_embed(words) # B,T,E
        e_w = e_w.detach() # fix word embedding
        e_w = self.dropout(e_w)
        e_c = self.c_embed(chars) # B,T,C
        e_c = self.dropout(e_c)
        embed = torch.cat([e_w,e_c],-1)
        return embed 

    def forward(self,words,chars):
        mask = words.eq(self.pad_idx).unsqueeze(-1) # B,T,1
        e_w = self.w_embed(words) # B,T,E
        e_w = e_w.detach() # fix word embedding
        e_w = self.dropout(e_w)
        e_c = self.c_embed(chars) # B,T,C
        e_c = self.dropout(e_c)
        embed = torch.cat([e_w,e_c],-1)
        embed , _ = self.contextual(embed)
        embed = embed.masked_fill(mask,0.)
        embed = self.dropout(embed)  
        return embed
    
class DocQA(QADiverModel):
    def __init__(self,w_size,w_embed,
                      c_size,c_embed,
                      hidden_dim,
                      dropout=0.2,pad_idx=1):
        super(DocQA,self).__init__()
        
        self.pad_idx = pad_idx
        self.embed = InputEmbedding(w_size,w_embed,c_size,c_embed,hidden_dim,\
                                    dropout,pad_idx)
        self.attn = sublayer.AttentionFlow(hidden_dim*2)
        self.linear1 = nn.Sequential(nn.Linear(hidden_dim*8,hidden_dim*2),
                                    nn.ReLU())
        self.sgru = nn.GRU(hidden_dim*2,hidden_dim,batch_first=True,bidirectional=True)
        self.sattn = sublayer.AttentionFlow(hidden_dim*2,True)
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim*6,hidden_dim*2),
                                    nn.ReLU())
        
        self.dropout = nn.Dropout(dropout)
        self.decoder = sublayer.SpanDecoder(hidden_dim,dropout)
        self.noanswer = sublayer.NoAnswer(hidden_dim)        

    def embedding(self,C,Cc):
        return self.embed.single(C,Cc)

    def attention(self,C,Q,Cc,Qc):
        c_mask = C.eq(self.pad_idx)
        q_mask = Q.eq(self.pad_idx)
        
        context_embed = self.embed(C,Cc) # B,T,2H
        question_embed = self.embed(Q,Qc) # B,T,2H

        C2Q, Q2C = self.attn.attention(context_embed, question_embed,\
                                       c_mask.unsqueeze(1),q_mask.unsqueeze(1)) # B,T,8H

        return C2Q, Q2C


    def forward(self,C,Q,Cc,Qc):
        """
        C : word ids of context
        Q : word ids of question
        Cc : character ids of context
        Qc : character ids of question        
        """
        c_mask = C.eq(self.pad_idx)
        q_mask = Q.eq(self.pad_idx)
        
        context_embed = self.embed(C,Cc) # B,T,2H
        question_embed = self.embed(Q,Qc) # B,T,2H
        
        # bidirectional attention
        G = self.attn(context_embed,question_embed,\
                      c_mask.unsqueeze(1),q_mask.unsqueeze(1)) # B,T,8H
        G = self.linear1(G) # B,T,2H
        
        # self attention
        S,_ = self.sgru(self.dropout(G))
        S = S.masked_fill(c_mask.unsqueeze(-1),0.)
        S = self.dropout(S)
        S = self.sattn(S,S,c_mask.unsqueeze(1),c_mask.unsqueeze(1))
        S = self.linear2(S)
        
        M = self.dropout(G + S) # Residual
        
        start_logit, end_logit = self.decoder(M,c_mask)
        noanswer_bias = self.noanswer(context_embed,start_logit,end_logit)
        p_1 = torch.cat([start_logit.squeeze(2),noanswer_bias],1)
        p_2 = torch.cat([end_logit.squeeze(2),noanswer_bias],1)

        return p_1,p_2
