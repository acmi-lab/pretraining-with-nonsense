{
    "dataset_reader": {
        "type": "pretrained_cnndmail_dataset_reader",
        "lazy": false,
        "lowercase_tokens": true,
        "max_source_length": 999999,
        "max_target_length": 999999,
        "max_to_read": 100000,
        "random_seed": 0
    },
    "iterator": {
        "type": "bucket",
        "batch_size": 8,
        "sorting_keys": [
            [
                "source_tokens",
                "num_tokens"
            ]
        ]
    },
    "model": {
        "type": "randominit_t5",
        "max_decode_length": 300,
        "min_decode_length": 0,
        "pretrained_model_name": "t5-small"
    },
    "train_data_path": "dataset_root/pretraining_datasets/ourtasks-nonsense/train.jsonl",
    "validation_data_path": "dataset_root/pretraining_datasets/ourtasks-nonsense/val.jsonl",
    "test_data_path": "dataset_root/pretraining_datasets/ourtasks-nonsense/test.jsonl",
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
        "patience": 10,
        "validation_metric": "+accuracy"
    },
    "vocabulary": {
        "directory_path": "dataset_root/pretrained_t5_vocabulary",
        "extend": false
    },
    "datasets_for_vocab_creation": [
        "train"
    ],
    "validation_dataset_reader": {
        "type": "pretrained_cnndmail_dataset_reader",
        "lazy": false,
        "lowercase_tokens": true,
        "max_source_length": 999999,
        "max_target_length": 999999,
        "max_to_read": 999999
    }
}
