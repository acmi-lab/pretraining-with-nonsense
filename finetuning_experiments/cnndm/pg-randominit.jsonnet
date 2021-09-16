{
    "dataset_reader": {
        "type": "pretrained_wordpiece_dataset_reader",
        "lazy": false,
        "lowercase_tokens": true,
        "max_source_wpiece_length": 512,
        "max_target_wpiece_length": 256,
        "max_to_read": 999999999
    },
    "iterator": {
        "type": "bucket",
        "batch_size": 16,
        "sorting_keys": [
            [
                "source_tokens",
                "num_tokens"
            ]
        ]
    },
    "model": {
        "type": "pointer_generator",
        "emb_size": 128,
        "hidden_size": 256,
        "max_decode_length": 148,
        "min_decode_length": 44,
        "use_copy_mech": true
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
            "type": "adam",
            "lr": 0.0001
        },
        "patience": 5,
        "validation_metric": "+accuracy"
    },
    "vocabulary": {
        "directory_path": "dataset_root/pretrained_t5_vocabulary",
        "extend": false
    },
    "validation_dataset_reader": {
        "type": "pretrained_wordpiece_dataset_reader",
        "lazy": false,
        "lowercase_tokens": true,
        "max_source_wpiece_length": 512,
        "max_target_wpiece_length": 256,
        "max_to_read": 999999999
    }
}
