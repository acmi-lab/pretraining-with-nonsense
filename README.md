## Does Pretraining for Summarization Reuqire Knowledge Transfer?
This repository is the official implementation of the work in the paper
`Does Pretraining for Summarization Reuqire Knowledge Transfer?` to appear 
in `Findings of EMNLP 2021.`  
You can find the paper on arXiv here: https://arxiv.org/abs/2109.04953


### Requirements
This code requires Python 3 (tested using version 3.6)

To install requirements, run:
```
pip install -r requirements.txt
```

### Preparing finetuning datasets

To prepare a summarization dataset for finetuning, run the corresponding script in the `finetuning_datasetgen`
 folder. For example, to prepare the cnn-dailymail dataset run:
```
cd finetuning_datasetgen
python cnndm.py
```

### Running finetuning experiment

We show here how to run training, prediction and  evaluation steps for 
a finetuning experiment.
We assume that you have downloaded the pretrained
models in the `pretrained_models`
folder from the provided Google Drive link (see `pretrained_models/README.md`)
If you want to pretrain models yourself, see latter part of this readme for the instructions.

All models in our work are trained using allennlp config files which are in `.jsonnet` format. 
To run a finetuning experiment, simply run
```
# for t5-like models
./pipeline_t5.sh <experiment_config_path>

# for pointer-generator models
./pipeline_pg.sh <experiment_config_path>
```

For example, for finetuning a T5 model on cnndailymail dataset, starting from 
a model pretrained with ourtasks-nonsense pretraining dataset, run
```bash
./pipeline_t5.sh finetuning_experiments/cnndm/t5-ourtasks-nonsense
```

Similarly, for finetuning a randomly-initialized pointer-generator model, run
```bash
./pipeline_pg.sh finetuning_experiments/cnndm/pg-randominit
```

The trained model and output files would be available in the `<experiment_config_path>` folder 
that would be created by the script.

`model.tar.gz` contains the trained (finetuned) model

`test_outputs.jsonl` contains the outputs of the model on the test split.

`test_genmetrics.json` contains the ROUGE scores of the output

### Creating pretraining datasets

We have provided the nonsense pretraining datasets used in our work via Google Drive
(see dataset_root/pretraining_datasets/README.md for instructions)

However, if you want to generate your own pretraining corpus, you can run
```commandline
cd pretraining_datasetgen
# for generating dataset using pretraining tasks
python ourtasks.py
# for generating dataset using STEP pretraining tasks
python steptasks.py
```
These commands would create pretraining datasets using nonsense.
If you want to create datasets starting from wikipedia documents 
please look into the two scripts which guide you how to do that by
commenting/uncommenting two blocks of code.


### Pretraining models

Although we provide you the pretrained model checkpoints via GoogleDrive, if you
want to pretrain your own models, you can do that by using the corresponding pretraining config file.
As an example, we have provided a config file which pretrains on ourtasks-nonsense dataset.
Make sure that the pretraining dataset files exist (either created by you or downloaded from GoogleDrive)
before running the pretraining command. The pretraining is also done using the same shell scripts used for the finetuning experiments.
For example, to pretrain a model on the `ourtasks-nonsense` dataset, simply run :

```commandline
./pipeline_t5.sh pretraining_experiments/pretraining_t5_ourtasks_nonsense
```


