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
    
class CharCNN(nn.Module):
    def __init__(self,vocab_size,embedding_dim,kernel_size,kernel_dim,dropout = 0.2,pad_idx = 1):
        super(CharCNN,self).__init__()
        
        self.char_embed = nn.Embedding(vocab_size,embedding_dim,padding_idx = pad_idx)
        self.conv = nn.Conv1d(1,kernel_dim,embedding_dim*kernel_size,stride=embedding_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self,x):
        b,t_w,t_c = x.size()
        embed = self.char_embed(x)
        embed = embed.view(b*t_w,1,-1)
        embed = self.dropout(embed)
        conved = F.relu(self.conv(embed))
        pooled = F.max_pool1d(conved,conved.size(2)).squeeze(2)
        return pooled.view(b,t_w,-1)

    
class Highway(nn.Module):
    def __init__(self, size, num_layers):
        super(Highway, self).__init__()
        self.num_layers = num_layers
        self.Hx = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.gate = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.relu = nn.ReLU()
    def forward(self, x):
        for layer in range(self.num_layers):
            gate = F.sigmoid(self.gate[layer](x))
            nonlinear = self.relu(self.Hx[layer](x))
            x = gate * nonlinear + (1 - gate) * x
        return x
    
    
    
class AttentionFlow(nn.Module):
    def __init__(self,dim,self_attn=False):
        super(AttentionFlow,self).__init__()
        self.w_1 = nn.Linear(dim,1,bias=False)
        self.w_2 = nn.Linear(dim,1,bias=False)
        self.w_3 = nn.Parameter(torch.randn(1,1,dim))
        self.self_attn = self_attn
        self.bias = nn.Parameter(torch.FloatTensor([[-1]]))
        
    def trilinear(self,C,Q):
        nbatch,n,m = C.size(0),C.size(1),Q.size(1) # B,N,D
        shape = (nbatch,n,m)
        c_logit = self.w_1(C).expand(shape)
        q_logit = self.w_2(Q).transpose(1,2).expand(shape)
        dot_logit = torch.mul(C,self.w_3)
        dot_logit = torch.matmul(dot_logit,Q.transpose(1,2))
        
        return c_logit + q_logit + dot_logit
        
    def masking(self,S,C_mask,Q_mask):
        # C_mask = B,1,N
        # Q_mask = B,1,M
        S = S.masked_fill(Q_mask,-1e9)
        S = S.masked_fill(C_mask.transpose(1,2),-1e9)
        return S
        
    def self_masking(self,S):
        nbatch,n,_ = S.size()
        eye = next(self.parameters())
        eye = eye.new_tensor(torch.randn(n,n))
        nn.init.eye_(eye)
        eye = eye.expand(nbatch,n,n)
        eye = eye.byte()
        return S.masked_fill(eye,-float('inf'))

    def attention(self,C,Q,c_mask,q_mask):
        S = self.trilinear(C,Q) # B,N,M
        
        if self.self_attn:
            S = self.masking(S,c_mask,q_mask)
            S = self.self_masking(S)
            bias = torch.exp(self.bias)
            C2Q = torch.exp(S)
            C2Q = C2Q / (C2Q.sum(dim=-1, keepdim=True).expand(C2Q.size()) + bias.expand(C2Q.size()))
            return C2Q, None
        else:
            C2Q = self.masking(S,c_mask,q_mask)
            C2Q = F.softmax(C2Q,2) # B,N,M
            Q2C = S.transpose(1,2).masked_fill(c_mask,-1e9) # B,M,N
            Q2C = F.softmax(Q2C.max(1,keepdim=True)[0],2) # B,1,N
            return C2Q, Q2C

    def forward(self,C,Q,c_mask,q_mask):
        nbatch,n,dim = C.size() # B,N,D
        C2Q, Q2C = self.attention(C,Q,c_mask,q_mask)
        if self.self_attn:
            A = torch.bmm(C2Q,Q)
            out = torch.cat([C,A,C*A],2)
        else:
            A = torch.bmm(C2Q,Q)
            Q2C = Q2C.expand(nbatch,n,n) # B,N,N
            B = torch.bmm(Q2C,C)
            out = torch.cat([C,A,C*A,C*B], 2)
        return out

class SpanDecoder(nn.Module):
    def __init__(self,hidden_dim,dropout=0.2):
        super(SpanDecoder,self).__init__()
        self.start_gru = nn.GRU(hidden_dim*2,hidden_dim,batch_first=True,bidirectional=True)
        self.end_gru = nn.GRU(hidden_dim*4,hidden_dim,1,batch_first=True,bidirectional=True)
        self.output_1 = nn.Linear(hidden_dim*2,1,bias=False)
        self.output_2 = nn.Linear(hidden_dim*2,1,bias=False)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self,M,c_mask):
        M_1,_ = self.start_gru(M)
        M_1 = M_1.masked_fill(c_mask.unsqueeze(-1),0.)
        M_1 = self.dropout(M_1)
        p_1 = self.output_1(M_1)
        M_2,_ = self.end_gru(torch.cat([M,M_1],2))
        M_2 = M_2.masked_fill(c_mask.unsqueeze(-1),0.)
        M_2 = self.dropout(M_2)
        p_2 = self.output_2(M_2) # B,T,1
        p_1 = p_1.masked_fill(c_mask.unsqueeze(-1),-1e7)
        p_2 = p_2.masked_fill(c_mask.unsqueeze(-1),-1e7)
        
        return p_1,p_2
    
class NoAnswer(nn.Module):
    def __init__(self,hidden_dim):
        super(NoAnswer,self).__init__()
        self.W_no = nn.Linear(hidden_dim*2,1,bias=False)
        self.no_answer = nn.Sequential(nn.Linear(hidden_dim*6,hidden_dim*2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_dim*2,1))
        
    def forward(self,context_embed,start_logit,end_logit):
        
        # No-answer option
        pa_1 = F.softmax(start_logit.transpose(1,2),-1) # B,1,T
        v1 = torch.bmm(pa_1,context_embed).squeeze(1) # B,2H
        pa_2 = F.softmax(end_logit.transpose(1,2),-1) # B,1,T
        v2 = torch.bmm(pa_2,context_embed).squeeze(1) # B,2H
        pa_3 = self.W_no(context_embed)
        pa_3 = F.softmax(pa_3.transpose(1,2),-1) # B,1,T
        v3 = torch.bmm(pa_3,context_embed).squeeze(1) # B,2H
        
        noanswer_bias = self.no_answer(torch.cat([v1,v2,v3],-1)) # B,1
        
        return noanswer_bias