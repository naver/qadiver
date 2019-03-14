# QADiver: Interactive Framework for Diagnosing QA Models

QADiver is the web-based interactive framework for diagonising the RC model for SQuAD 2.0 dataset.  
[[Paper]](https://arxiv.org/abs/1812.00161) [[Demo Video]](https://youtu.be/V6c8nls6Qcc)

## Requirements and Dependencies
This demo runs on Python 3. (Recommend: 3.6+)  
You need to install below packages to use this framework.
```
nltk
numexpr
numpy
Flask_Cors
Flask
tqdm
faiss==1.4.0
tornado
scikit_learn>=0.19
```

To install ``faiss``, see this [link](https://github.com/facebookresearch/faiss/blob/master/INSTALL.md).

Also, these packages are required to use the toy model.
```
torch>=0.4.0
matplotlib
```

## Run the demo with the toy model
```
$ ./prepare_toy_model.sh
$ python server.py --config config/server.json
```

## Using Docker Environment

### 1. Pull Docker image and Run Container

```
$ pip install docker-compose
$ docker-compose up -d
```

### 2. Attach the container (default name is qadiver_app_1)

```
$ docker exec -it qadiver_app_1 bash
```

### 3. Run server on the container

```
root@xxxx: cd app/QADiver/
root@xxxx: ./prepare_toy_model.sh   # if you want to start with toy model
root@xxxx: python server.py --config config/server.json
```




## Using the framework for you model

### 1. Writing inference.py
To use the tools in the framework, you must complete methods in ``QAModel`` class in ``inference.py``.  
This class connects your model implementation with the framework. 

```
- __init__: load model params
- __call__: returns answer and no ans prob from the model
- tokenize: tokenize text into tokens
- word_answer: return word-level answer location (start, end) used for training model
- prepare_input: convert question/context pair to model input
- embedding: return embedding used in the model for single word
- attention: return attention matrix from the model
- forward: get start and end probabiltiy from the model
```

You can see the example code in the ``example`` folder, which is written for the toy model.  
※ If you want to only visualize the result of the model (with ``--use_tools 0`` option), you don't need to fill above methods.

### 2. Preprocessing
#### 2-1. Preparing model evaluation info
You have to prepare json file containing the evaluation result of the model for given no-answer threshold.  
To obtain such information, you need two json file that contains answer and no-answer probability for given question id.  
(see `example/pred_example.json` and `example/no_prob_example.json`)  
Threshold can be determined by evaluating your result with the [official evaluation script](https://rajpurkar.github.io/SQuAD-explorer/).

When these files are ready, you can generate evaluation information by run this line:  
``python script/evaluate_result.py --config_file config/eval.json``  

You can configure related settings on `config/eval.json` file.
```
- data_path: path of the SQuAD dataset file (default: data/dev-v2.0.json)
- pred_path: path of json file contains prediction result from the model
- no_prob_path: path of json file contains no-answer probability from the model
- na_threshold: no-answer probability threshold to determine the question is answerable or not (example: 0.5)
- output: path of json file to export the evaluation result 
```

#### 2-2. Caching model result
We recommend to cache the evaluation result, especially when you run the framework on CPU-only device.  
Run this line: ``python script/cache_result.py --config_file config/cache.json``  

You can configure related settings on `config/cache.json` file.  
```
- cache_path: path to cache the evaluation result (default: cache/qa_cache.db)
- data_path: path of preprocessed data (see 2-3)
- pred_path: path of json file contains prediction result from the model
- no_prob_path: path of json file contains no-answer probability from the model
```

#### 2-3. Preparing data and embedding
This script will convert SQuAD dataset to the format used in the framework, and extract embedding vector from the model.  
You must implement `embedding` in `inference.py` when you want to extract embedding vector.  

Run this line: ``python script/data_process.py --config_file config/data_process.json``  

You can configure related settings on `config/data_process.json` file.  
```
- filename: path of the SQuAD dataset file (default: data/dev-v2.0.json)
- output_path: path to export preprocessed data (default: data/processed_data.json)
- model_name: identifier for the model (example: DocQA)
- checkpoint_path: path of model checkpoint file
- extract_embedding: whether to extract embedding vector from the model or not (1 for true, 0 for false)
- embedding_output: path to export embedding vector
```

Also, you have to generate vocabulary dictionary containing the word when used for the model training.  
One common way to get this information is dumping the word2idx variable saved in the model checkpoint.

#### 2-4. Generating question vector
If you are using other dataset than dev set (like `train-v2.0.json`), you must update the question vector.  
For SQuAD dataset, you can just fix the dataset filename in `script/prepare_sentence_features.py` and run it.  
Else, you should reimplement `find_similar_sentence` from `SentLoader` class in `tools/sentence/compare.py`.  
(this function returns the list of question ids, sorted with the question similarity score.)


### 3. Run the server!
Run this line: ``python server.py --config config/server.json --port <PORT>``  

You can configure related settings on `config/server.json` file:
```
- data_path: path of preprocessed data (see 2-3)
- vocab_path: path of vocabulary dictionary (word2idx) used in the model (see 2-3)
- embedding_path: path of extracted embedding vector from the model (see 2-3)
- evaluation_result: path of json file containing evaluation result (see 2-1)

- model_name: identifier for the model (example: DocQA)
- checkpoint_path: path of model checkpoint file
- cache_path: path of the cache (see 2-2, default: cache/qa_cache.db)

- use_tools: whether to use diagonising tools or not (1 for true, 0 for false)
- sim_sent_num: number of (default: 50)
- adversarial_rule_path: path for predefined rules on adversarial test (default: data/sear_rules.txt)

- port: port to serve QADiver on (default: 8080)
```

## Misc
- This framework works on full-screen mode. Unexpected behavior can be appeared when using on the small window.
- `script/evaluate_result.py` is written based on [official evaluation script of SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/).

## License

```
Copyright 2019-present NAVER Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
