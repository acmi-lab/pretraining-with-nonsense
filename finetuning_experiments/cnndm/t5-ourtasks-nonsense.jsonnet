{
    "dataset_reader": {
        "type": "pretrained_cnndmail_dataset_reader",
        "lazy": true,
        "lowercase_tokens": true,
        "max_source_wpiece_length": 512,
        "max_target_wpiece_length": 256,
        "max_to_read": 999999999
    },
    "iterator": {
        "type": "basic",
        "batch_size": 16
    },
    "model": {
        "type": "pretrained_t5",
        "max_decode_length": 148,
        "min_decode_length": 44,
        "model_weights_file": "./pretrained_models/pretrained_ourtasks_nonsense/best.th",
        "pretrained_model_name": "t5-small"
    },
    "train_data_path": "dataset_root/finetuning_datasets/cnndm/train.jsonl",
    "validation_data_path": "dataset_root/finetuning_datasets/cnndm/val.jsonl",
    "test_data_path": "dataset_root/finetuning_datasets/cnndm/test.jsonl",
    "trainer": {
        "cuda_device": 0,
        "grad_clipping": 0.5,
        "grad_norm": 2,
        "num_epochs": 100,
        "num_serialized_models_to_keep": 1,
        "optimizer": {
            "type": "bert_adam",
            "lr": 0.0001,
            "max_grad_norm": 1
        },
        "patience": 5,
        "validation_metric": "+accuracy"
    },
    "vocabulary": {
        "directory_path": "dataset_root/pretrained_t5_vocabulary",
        "extend": false
    },
    "validation_dataset_reader": {
        "type": "pretrained_cnndmail_dataset_reader",
        "lazy": true,
        "lowercase_tokens": true,
        "max_source_wpiece_length": 512,
        "max_target_wpiece_length": 256,
        "max_to_read": 999999999
    }
}
