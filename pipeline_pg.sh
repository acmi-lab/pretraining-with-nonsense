#!/bin/bash
test_file_path=`python -c 'import json; print(json.load(open("'$1'.jsonnet"))["test_data_path"])'`
val_file_path=`python -c 'import json; print(json.load(open("'$1'.jsonnet"))["validation_data_path"])'`
train_file_path=`python -c 'import json; print(json.load(open("'$1'.jsonnet"))["train_data_path"])'`


allennlp train $1.jsonnet --include-package pointergen -s $1
allennlp evaluate $1/model.tar.gz $test_file_path --include-package pointergen --cuda-device 0 --output-file $1/test_metrics.json
allennlp predict $1/model.tar.gz --include-package pointergen --cuda-device 0 --predictor beamsearch_wordpiece $test_file_path --output-file $1/test_outputs.jsonl
python calculate_rouge.py -prediction_file $1/test_outputs.jsonl -output_file $1/test_genmetrics.json
sha256sum ${train_file_path} >> $1/checksums.txt
sha256sum ${val_file_path} >> $1/checksums.txt
sha256sum ${test_file_path} >> $1/checksums.txt
