import os.path
import pdb
import tensorflow_datasets as tfds
import nltk, jsonlines
from tqdm import tqdm

def get_dataset_reformed(ds):
    alls=[]
    for dp in tqdm(ds):
        article = dp["article"].numpy().decode("utf-8")
        summary_lines = dp["highlights"].numpy().decode("utf-8").split("\n")
        alls.append({
            "article_lines": [article],
            "summary_lines": summary_lines
        })
    return alls

# actually we loaded the dataset by using shuffle_files=True, but for reproducibility,
# we will load the exact datapoint indices that we had got when we created the dataset earlier
# and then pick out those datapoints from the full dataset
train_ds = tfds.load('cnn_dailymail', split='train', shuffle_files=False)
val_ds = tfds.load('cnn_dailymail', split='validation', shuffle_files=False)
test_ds = tfds.load('cnn_dailymail', split='test', shuffle_files=False)

train_indices = open("./cnndm_train_indices.txt","r").read().strip().split("\n")
train_indices = [int(x) for x in train_indices]
val_indices = open("./cnndm_val_indices.txt","r").read().strip().split("\n")
val_indices = [int(x) for x in val_indices]


train_ds = list(train_ds)
train_ds_subset = [train_ds[i] for i in train_indices]

val_ds = list(val_ds)
val_ds_subset = [val_ds[i] for i in val_indices]


train_ds = train_ds_subset
val_ds = val_ds_subset

train_ds2 = get_dataset_reformed(train_ds)
val_ds2 = get_dataset_reformed(val_ds)
test_ds2 = get_dataset_reformed(test_ds)



if not os.path.exists("../dataset_root/finetuning_datasets"):
    os.mkdir("../dataset_root/finetuning_datasets")

os.mkdir("../dataset_root/finetuning_datasets/cnndm")


with jsonlines.open("../dataset_root/finetuning_datasets/cnndm/train.jsonl", "w") as w:
    for dp in train_ds2:
        w.write(dp)

with jsonlines.open("../dataset_root/finetuning_datasets/cnndm/val.jsonl", "w") as w:
    for dp in val_ds2:
        w.write(dp)

with jsonlines.open("../dataset_root/finetuning_datasets/cnndm/test.jsonl", "w") as w:
    for dp in test_ds2:
        w.write(dp)

