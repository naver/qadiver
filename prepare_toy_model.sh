#!/bin/bash

c_dir=$(pwd)
checkpoint_path="https://drive.google.com/uc?export=view&id=1iAskLYcA2erCqxFW0L-AvApF1E6_XH_F"

# Download toy model checkpoints
cd toy_model/
if [ ! -f checkpoint.tar.gz ]; then
    wget --no-check-certificate "$checkpoint_path" -O checkpoint.tar.gz
fi
tar xzvf checkpoint.tar.gz && rm -r checkpoint.tar.gz

# Run preprocess script
cd $c_dir

cp example/inference.py .
cp -r example/config/ .

python script/evaluate_result.py --config_file config/eval.json
python script/cache_result.py --config_file config/cache.json
python script/data_process.py --config_file config/data_process.json

