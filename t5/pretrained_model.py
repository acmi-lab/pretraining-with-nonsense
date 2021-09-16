import pdb

import torch
import torch.nn as nn
import numpy as np
import sys

from allennlp.models.model import Model
from typing import Dict
from overrides import overrides
from allennlp.data.dataset import Batch
from allennlp.nn import util
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.training.metrics import CategoricalAccuracy
from torch.nn import CrossEntropyLoss
from transformers import T5Tokenizer, T5Model, T5ForConditionalGeneration, T5Config
from allennlp.data.instance import Instance


EPS = 1e-8


@Model.register("pretrained_t5")
class PretrainedT5(Model):
    def __init__(self,
                 vocab,
                 pretrained_model_name='t5-base',
                 model_weights_file=None,
                 min_decode_length=0,
                 max_decode_length=99,
                 softmax_temperature=1.0):
        super().__init__(vocab)

        ## vocab related setup begins
        assert "tokens" in vocab._token_to_index and len(vocab._token_to_index.keys())==1, "Vocabulary must have tokens as the only namespace"
        self.vocab_size=vocab.get_vocab_size()
        self.PAD_ID = vocab.get_token_index(vocab._padding_token)
        self.OOV_ID = vocab.get_token_index(vocab._oov_token)
        self.START_ID = vocab.get_token_index(START_SYMBOL)
        self.END_ID = vocab.get_token_index(END_SYMBOL)
        ## vocab related setup ends

        # buffers because these dont need grads. These are placed here because they will be replicated across gpus
        self.register_buffer("true_rep", torch.tensor(1.0))
        self.register_buffer("false_rep", torch.tensor(0.0))

        self.min_decode_length = min_decode_length
        self.max_decode_length = max_decode_length

        self.metrics = {
            "accuracy" : CategoricalAccuracy()
                        }

        self.softmax = nn.Softmax(dim=-1)
        self.softmax_temperature = softmax_temperature  # WARNING: only used for calculating the crossentropy metric

        if model_weights_file!=None:
            base_config = T5Config.from_pretrained(pretrained_model_name)
            model: torch.nn.Module = T5ForConditionalGeneration(base_config)
            # for loading pretrained weights saved by allennlp
            x = torch.load(model_weights_file)
            new_dict = {}
            for (k,v) in x.items():
                if k[:8]=="T5Model.":
                    new_dict[k[8:]] = v
                else:
                    continue

            model.load_state_dict(new_dict)
            self.T5Model = model
        else:
            T5Model = T5ForConditionalGeneration.from_pretrained(pretrained_model_name)
            self.T5Model = T5Model


    def forward(self, source_tokens, target_tokens, meta=None, only_predict_probs=False, return_pgen=False):
        inp_with_unks = source_tokens["tokens"]

        feed_tensor = target_tokens["tokens"][:, :-1]
        target_tensor = target_tokens["tokens"].detach().clone()[:, 1:]

        batch_size = inp_with_unks.size(0)
        input_pad_mask=torch.where(inp_with_unks!=self.PAD_ID, self.true_rep, self.false_rep)

        # output_pad_mask is not needed rather the lm labels have to be set to -100
        output_pad_mask=torch.where(target_tensor!=self.PAD_ID, self.true_rep, self.false_rep)
        target_tensor[target_tensor==self.PAD_ID]=-100         #hardcoded value courtesy of transformers people

        loss, logits, _, probably_encoder_output = self.T5Model(
                                                        input_ids=inp_with_unks,
                                                        attention_mask=input_pad_mask,
                                                        decoder_input_ids=feed_tensor,
                                                        use_cache=None,
                                                        labels=target_tensor)

        predicted_seqfirst = logits.permute(1,0,2)
        true_labels_seqfirst = target_tensor.permute(1,0)
        mask_seqfirst = output_pad_mask.permute(1,0)

        for (p, t, m) in zip(predicted_seqfirst, true_labels_seqfirst, mask_seqfirst):
            self.metrics["accuracy"](p, t, m)

        return {
            "loss": loss,
            "logits": logits,
        }


    @overrides
    def forward_on_instance(self, instance: Instance, decode_strategy) -> Dict[str, str]:
        cuda_device = self._get_prediction_device()
        dataset = Batch([instance])
        dataset.index_instances(self.vocab)
        model_input = util.move_to_device(dataset.as_tensor_dict(), cuda_device)
        if decode_strategy=='greedy':
            output_ids = self.common_decode(**model_input, min_length=self.min_decode_length, max_length=self.max_decode_length, num_beams=1)
        elif decode_strategy=='beamsearch':
            output_ids = self.common_decode(**model_input, min_length=self.min_decode_length, max_length=self.max_decode_length, num_beams=4)
        else:
            raise NotImplementedError

        output_words = []

        for _id in output_ids:
            output_words.append(self.vocab.get_token_from_index(_id))

        assert output_words[0]==START_SYMBOL, "somehow the first symbol is not the START symbol. might be a bug"
        output_words=output_words[1:]

        if output_words[-1]==END_SYMBOL:
            output_words = output_words[:-1]

        return {"tokens": output_words}


    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics_to_return = {
            metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()
        }
        return metrics_to_return


    def common_decode(self, source_tokens, target_tokens=None, meta=None, min_length=35, max_length=120, num_beams=1):
        inp_with_unks = source_tokens["tokens"]
        input_pad_mask=torch.where(inp_with_unks!=self.PAD_ID, self.true_rep, self.false_rep)

        generated_ids = self.T5Model.generate(
            input_ids = inp_with_unks,
            attention_mask = input_pad_mask,
            min_length = min_length,
            max_length = max_length,
            decoder_start_token_id = self.START_ID,
            bos_token_id = self.START_ID,
            pad_token_id = self.PAD_ID,
            eos_token_id = self.END_ID,
            num_beams = num_beams
        )

        return generated_ids.detach().cpu().numpy()[0]




@Model.register("randominit_t5")
class RandomInitT5(PretrainedT5):
    def __init__(self,
                 vocab,
                 pretrained_model_name='t5-base',
                 min_decode_length=0,
                 max_decode_length=99,
                 softmax_temperature=1.0):
        super().__init__(vocab, softmax_temperature=softmax_temperature)

        ## vocab related setup begins
        assert "tokens" in vocab._token_to_index and len(vocab._token_to_index.keys())==1, "Vocabulary must have tokens as the only namespace"
        self.vocab_size=vocab.get_vocab_size()
        self.PAD_ID = vocab.get_token_index(vocab._padding_token)
        self.OOV_ID = vocab.get_token_index(vocab._oov_token)
        self.START_ID = vocab.get_token_index(START_SYMBOL)
        self.END_ID = vocab.get_token_index(END_SYMBOL)
        ## vocab related setup ends


        self.min_decode_length = min_decode_length
        self.max_decode_length = max_decode_length

        self.softmax = nn.Softmax(dim=-1)

        base_config = T5Config.from_pretrained(pretrained_model_name)
        self.T5Model = T5ForConditionalGeneration(base_config)

