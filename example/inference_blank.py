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


'''
QAModel class: interface for model implementation
'''

class QAModel:
    r"""Interface for model implementation

    Arguments:
        checkpoint (str): checkpoint path for model
        model (str, optional): model identifier
    """

    def __init__(self, checkpoint, model=""):
        pass

    def __call__(self, context, question):
        r"""Answer and no-answer probability from the model

        Arguments:
            context (string): original context 
            question (string): original question 
        
        Returns:
            1) answer span and 2) no-answer probability (string, float)
        """
        return "", 0.0

    def tokenize(self, text):
        r"""Tokenize text into tokens
        
        Arguments:
            text (string): text to tokenize
        
        Returns:
            list: tokens 
        """
        return text.split(" ")

    def word_answer(self, context, context_tokens, answer_data):
        r"""Word-level answer location used for training model
        
        Arguments:
            context (string): original document
            context_tokens (list): tokenized context (by self.tokenize)
            answer_data (list of dict): SQuAD-format answer data
        
        Returns:
            start, end index of the answer span basd on token list
        """
        return -1, -1
 
    def prepare_input(self, context, question):
        r"""Convert question/context pair to model input

        Arguments:
            context (string): original context 
            question (string): original question 
        
        Returns:
            input data for the model
        """
        return None

    def embedding(self, word):
        r"""Embedding used in the model for single word

        Arguments:
            word (string): input word
        
        Returns:
            vector: embedding for the given word, likely numpy array
        """
        return None

    def attention(self, context, question):
        r"""Attention matrix from the model

        Arguments:
            context (string): original context 
            question (string): original question 
 
        Returns:
            2d attention matrix extracted from the model
            (size: [# of tokens in the question] x [# of tokens in the context])

        """
        return None

    def forward(self, context, question):
        r"""Retrieve start and end probabiltiy from the model

        Arguments:
            context (string): original context 
            question (string): original question 
 
        Returns:
            start_prob, end_prob (float, float): model output (probability of each tokens)
        """
        return 0, 0

   
