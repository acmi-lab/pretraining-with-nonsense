import pdb

import torch
import torch.nn as nn
import numpy as np
import sys

from allennlp.models.model import Model
from pointergen.custom_instance import SyncedFieldsInstance
from typing import Dict
from overrides import overrides
from allennlp.data.dataset import Batch
from allennlp.nn import util
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.training.metrics import CategoricalAccuracy

EPS=1e-8

class Attention(nn.Module):
    def __init__(self, total_encoder_hidden_size, total_decoder_hidden_size, attn_vec_size):
        super(Attention, self).__init__()
        self.total_encoder_hidden_size=total_encoder_hidden_size
        self.total_decoder_hidden_size=total_decoder_hidden_size
        self.attn_vec_size=attn_vec_size
        
        self.Wh_layer=nn.Linear(total_encoder_hidden_size, attn_vec_size, bias=False)
        self.Ws_layer=nn.Linear(total_decoder_hidden_size, attn_vec_size, bias=True)
        self.selector_vector_layer=nn.Linear(attn_vec_size, 1, bias=False)       # called 'v' in see et al


    def forward(self, encoded_seq, decoder_state, input_pad_mask):
        '''
        encoded seq is batchsizexenc_seqlenxtotal_encoder_hidden_size
        decoder_state is batchsizexdec_seqlenxtotal_decoder_hidden_size
        '''

        projected_decstates = self.Ws_layer(decoder_state)
        projected_encstates = self.Wh_layer(encoded_seq)

        added_projections=projected_decstates.unsqueeze(2)+projected_encstates.unsqueeze(1)   #batchsizeXdeclenXenclenXattnvecsize
        added_projections=torch.tanh(added_projections)
         
        attn_logits=self.selector_vector_layer(added_projections)
        attn_logits=attn_logits.squeeze(3)

        attn_weights = torch.softmax(attn_logits, dim=-1)  # shape=batchXdec_lenXenc_len
        attn_weights2 = attn_weights*input_pad_mask.unsqueeze(1)
        attn_weights_renormalized = attn_weights2/torch.sum(attn_weights2, dim=-1, keepdim=True)  # shape=batchx1x1

        context_vector = torch.sum(encoded_seq.unsqueeze(1)*attn_weights_renormalized.unsqueeze(-1) , dim=-2)
        # shape batchXdec_seqlenXhiddensize
        # print(context_vector)

        return context_vector, attn_weights_renormalized
        
        
    
        
class CopyMechanism(nn.Module):
    def __init__(
            self, encoder_hidden_size, decoder_hidden_size, decoder_input_size):
        super(CopyMechanism, self).__init__()
        self.pgen=nn.Sequential(
            nn.Linear(encoder_hidden_size+2*decoder_hidden_size+decoder_input_size, 1),
            nn.Sigmoid()
        )
        self.output_probs=nn.Softmax(dim=-1)

    def forward(
            self, output_logits, attn_weights, decoder_hidden_state, decoder_input,
            context_vector, encoder_input, max_oovs):
        '''output_logits = batchXseqlenXoutvocab
            attn_weights = batchXseqlenXenc_len
            decoder_hidden_state = batchXseqlenXdecoder_hidden_size
            context_vector = batchXseqlenXencoder_hidden_dim
            encoder_input = batchxenc_len'''             
        output_probabilities=self.output_probs(output_logits)

        batch_size = output_probabilities.size(0)
        output_len = output_probabilities.size(1)
        append_for_copy = torch.zeros((batch_size, output_len, max_oovs)).cuda()
        output_probabilities=torch.cat([output_probabilities, append_for_copy], dim=-1)
        
        pre_pgen_tensor=torch.cat([context_vector, decoder_hidden_state, decoder_input], dim=-1)
        pgen=self.pgen(pre_pgen_tensor)    # batchsizeXseqlenX1
        pcopy=1.0-pgen   

        encoder_input=encoder_input.unsqueeze(1).expand(-1, output_len , -1)    # batchXseqlenXenc_len

#         Note that padding words donot get any attention because the attention is a masked attention

        copy_probabilities=torch.zeros_like(output_probabilities)  # batchXseqlenXoutvocab
        copy_probabilities.scatter_add_(2, encoder_input, attn_weights)        

        total_probabilities=pgen*output_probabilities+pcopy*copy_probabilities
        return total_probabilities, pgen  # batchXseqlenXoutvocab , batchsizeXseqlenX1
        
@Model.register("pointer_generator")
class Seq2Seq(Model):
    def __init__(self,
                 vocab,
                 hidden_size=256,
                 emb_size=128,
                 num_encoder_layers=1,
                 num_decoder_layers=1,
                 min_decode_length=0,
                 max_decode_length=9999,
                 use_copy_mech=True,
                 model_weights_file=None):
        super().__init__(vocab)
        # self.vocab=vocab

        ## vocab related setup begins
        assert "tokens" in vocab._token_to_index and len(vocab._token_to_index.keys())==1, "Vocabulary must have tokens as the only namespace"
        self.vocab_size=vocab.get_vocab_size()
        self.PAD_ID = vocab.get_token_index(vocab._padding_token)
        self.OOV_ID = vocab.get_token_index(vocab._oov_token)
        self.START_ID = vocab.get_token_index(START_SYMBOL)
        self.END_ID = vocab.get_token_index(END_SYMBOL)
        ## vocab related setup ends

        self.emb_size=emb_size
        self.hidden_size=hidden_size
        self.num_encoder_layers=num_encoder_layers
        self.num_decoder_layers=num_decoder_layers
        self.crossentropy=nn.CrossEntropyLoss()

        self.min_decode_length = min_decode_length
        self.max_decode_length = max_decode_length

        self.metrics = {
            "accuracy" : CategoricalAccuracy(),
                        }
        
        self.register_buffer("true_rep", torch.tensor(1.0))
        self.register_buffer("false_rep", torch.tensor(0.0))

        self.pre_output_dim=hidden_size

        self.use_copy_mech=use_copy_mech
        
        self.output_embedder = nn.Sequential(
            nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.emb_size)
        )              
        
        self.input_encoder = nn.Sequential(
            self.output_embedder,
            torch.nn.LSTM(input_size=self.emb_size, hidden_size=self.hidden_size, num_layers=self.num_encoder_layers, batch_first=True, bidirectional=True),
        )
        
        self.fuse_h_layer= nn.Sequential(
            nn.Linear(2*hidden_size, hidden_size),
            nn.ReLU()
        )
        
        self.fuse_c_layer= nn.Sequential(
            nn.Linear(2*hidden_size, hidden_size),
            nn.ReLU()
        )
        
        self.attention_layer=Attention(2*hidden_size, 2*hidden_size, 2*hidden_size)
        
        if self.use_copy_mech:
            self.copymech=CopyMechanism(2*self.hidden_size, self.hidden_size, self.emb_size)
        
        self.decoder_rnn=torch.nn.LSTM(input_size=self.emb_size, hidden_size=self.hidden_size, num_layers=self.num_decoder_layers, batch_first=False, bidirectional=False)
        
        self.statenctx_to_prefinal = nn.Linear(3*hidden_size, hidden_size, bias=True)
        self.project_to_decoder_input = nn.Linear(emb_size+2*hidden_size, emb_size, bias=True)
        
        self.output_projector = torch.nn.Conv1d(self.pre_output_dim, self.vocab_size, kernel_size=1, bias=True)
        self.softmax = nn.Softmax(dim=-1)

        if model_weights_file is not None:
            x = torch.load(model_weights_file)
            self.load_state_dict(x)
    

    def forward(self, source_tokens, target_tokens, meta=None):
        inp_with_unks = source_tokens["ids_with_unks"]
        inp_with_oovs = source_tokens["ids_with_oovs"]
        max_oovs = int(torch.max(source_tokens["num_oovs"]))

        feed_tensor = target_tokens["ids_with_unks"][:, :-1]
        if self.use_copy_mech:
            target_tensor = target_tokens["ids_with_oovs"][:,1:]
        else:
            target_tensor = target_tokens["ids_with_unks"][:, 1:]

        batch_size = inp_with_unks.size(0)
        # preparing intial state for feeding into decoder. layers of decoder after first one get zeros as initial state
        inp_enc_seq, (last_h_value, last_c_value) = self.encode(inp_with_unks)


        # inp_enc_seq is batchsizeXseqlenX2*hiddensize
        h_value = self.pad_zeros_to_init_state(last_h_value)
        c_value = self.pad_zeros_to_init_state(last_c_value)
        state_from_inp = (h_value, c_value)

        input_pad_mask=torch.where(inp_with_unks!=0, self.true_rep, self.false_rep)
        
        output_embedded = self.output_embedder(feed_tensor)
        seqlen_first = output_embedded.permute(1,0,2)
        output_seq_len = seqlen_first.size(0)

        #initial values
        decoder_hidden_state=state_from_inp

        # CONTROVERSIAL DIFFERENCE FROM SEE ET AL
        decoder_hstates_batchfirst = state_from_inp[0].permute(1, 0, 2)
        decoder_cstates_batchfirst = state_from_inp[1].permute(1, 0, 2)
        concatenated_decoder_states = torch.cat([decoder_cstates_batchfirst, decoder_hstates_batchfirst], dim=-1)
        context_vector, _ = self.attention_layer(inp_enc_seq, concatenated_decoder_states, input_pad_mask)
        #

        output_probs=[]

        for _i in range(output_seq_len):
            seqlen_first_onetimestep = seqlen_first[_i:_i+1]                # shape is 1xbatchsizexembsize
            context_vector_seqlenfirst = context_vector.permute(1,0,2)     # seqlen is 1 always
            pre_input_to_decoder=torch.cat([seqlen_first_onetimestep, context_vector_seqlenfirst], dim=-1)
            input_to_decoder=self.project_to_decoder_input(pre_input_to_decoder)   # shape is 1xbatchsizexembsize

            decoder_h_values, decoder_hidden_state = self.decoder_rnn(input_to_decoder, decoder_hidden_state)
            # decoder_h_values is shape 1XbatchsizeXhiddensize

            decoder_h_values_batchfirst = decoder_h_values.permute(1,0,2)

            decoder_hstates_batchfirst = decoder_hidden_state[0].permute(1, 0, 2)
            decoder_cstates_batchfirst = decoder_hidden_state[1].permute(1, 0, 2)
            concatenated_decoder_states = torch.cat([decoder_cstates_batchfirst, decoder_hstates_batchfirst], dim=-1)

            context_vector, attn_weights = self.attention_layer(inp_enc_seq, concatenated_decoder_states, input_pad_mask)

            decstate_and_context=torch.cat([decoder_h_values_batchfirst, context_vector], dim=-1)  #batchsizeXdec_seqlenX3*hidden_size
            prefinal_tensor = self.statenctx_to_prefinal(decstate_and_context)
            seqlen_last = prefinal_tensor.permute(0,2,1)   #batchsizeXpre_output_dimXdec_seqlen
            logits = self.output_projector(seqlen_last)
            logits = logits.permute(0,2,1)   # batchXdec_seqlenXvocab

            # now doing copymechanism
            if self.use_copy_mech:
                probs_after_copying, pgen = self.copymech(logits, attn_weights, concatenated_decoder_states, input_to_decoder.permute(1,0,2), context_vector, inp_with_oovs, max_oovs)
                output_probs.append(probs_after_copying)
            else:
                output_probs.append(self.softmax(logits))

        # now calculating loss and numpreds
        targets_tensor_seqfirst = target_tensor.permute(1,0)
        pad_mask=torch.where(targets_tensor_seqfirst!=self.PAD_ID, self.true_rep, self.false_rep)

        loss=0.0
        numpreds=0
        numpreds_placewise=torch.zeros((output_seq_len)).cuda()


        for _i in range(len(output_probs)):
            predicted_probs = output_probs[_i].squeeze(1)
            true_labels = targets_tensor_seqfirst[_i]
            mask_labels = pad_mask[_i]      
            selected_probs=torch.gather(input=predicted_probs, dim=1, index=true_labels.unsqueeze(1))
            selected_probs = selected_probs.squeeze(1)
            selected_probs = torch.clamp(selected_probs, min=1e-36, max=999)
            selected_neg_logprobs=-1*torch.log(selected_probs)
            loss+=torch.sum(selected_neg_logprobs*mask_labels)
            
            this_numpreds=torch.sum(mask_labels).detach()
            numpreds+=this_numpreds

            for metric in self.metrics.values():
                metric(predicted_probs, true_labels, mask_labels)

        return {
            "loss": loss/numpreds
        }

            
    def pad_zeros_to_init_state(self, h_value):
        '''can also be c_value'''
        assert(h_value.size(0)==1)       # h_value should only be of last layer of lstm
        return torch.cat([h_value]+[torch.zeros_like(h_value) for _i in range(self.num_encoder_layers-1)], dim=0) 

    
    def encode(self, inp):
        '''Get the encoding of input'''
        batch_size = inp.size(0)
        inp_seq_len = inp.size(1)
        inp_encoded = self.input_encoder(inp)
        output_seq=inp_encoded[0]
        h_value, c_value = inp_encoded[1]

        h_value_layerwise=h_value.reshape(self.num_encoder_layers, 2, batch_size, self.hidden_size)  # numlayersXbidirecXbatchXhid 
        c_value_layerwise=c_value.reshape(self.num_encoder_layers, 2, batch_size, self.hidden_size)  # numlayersXbidirecXbatchXhid 
        
        last_layer_h=h_value_layerwise[-1:,:,:,:]
        last_layer_c=c_value_layerwise[-1:,:,:,:]
        
        last_layer_h=last_layer_h.permute(0,2,1,3).contiguous().view(1, batch_size, 2*self.hidden_size)
        last_layer_c=last_layer_c.permute(0,2,1,3).contiguous().view(1, batch_size, 2*self.hidden_size)
        
        last_layer_h_fused=self.fuse_h_layer(last_layer_h)
        last_layer_c_fused=self.fuse_c_layer(last_layer_c)

        return output_seq, (last_layer_h_fused, last_layer_c_fused)
    
        
    def decode_onestep(self, past_outp_input, past_state_tuple, past_context_vector, inp_enc_seq, inp_with_oovs, input_pad_mask, max_oovs):
        '''run one step of decoder. outp_input is batchsizex1
        past_context_vector is batchsizeX1Xtwice_of_hiddensize'''
        outp_embedded = self.output_embedder(past_outp_input)
        tok_seqlen_first = outp_embedded.permute(1,0,2)
        assert(tok_seqlen_first.size(0)==1) # only one timestep allowed
        
        context_vector_seqlenfirst = past_context_vector.permute(1,0,2)     # seqlen is 1 always
        pre_input_to_decoder=torch.cat([tok_seqlen_first, context_vector_seqlenfirst], dim=-1)
        input_to_decoder=self.project_to_decoder_input(pre_input_to_decoder)   # shape is 1xbatchsizexembsize
                

        decoder_h_values, decoder_hidden_state = self.decoder_rnn(input_to_decoder, past_state_tuple)
        # decoder_h_values is shape 1XbatchsizeXhiddensize       
        decoder_h_values_batchfirst = decoder_h_values.permute(1,0,2)

        decoder_hstates_batchfirst = decoder_hidden_state[0].permute(1, 0, 2)
        decoder_cstates_batchfirst = decoder_hidden_state[1].permute(1, 0, 2)
        concatenated_decoder_states = torch.cat([decoder_cstates_batchfirst, decoder_hstates_batchfirst], dim=-1)

        context_vector, attn_weights = self.attention_layer(inp_enc_seq, concatenated_decoder_states, input_pad_mask)
        
        decstate_and_context=torch.cat([decoder_h_values_batchfirst, context_vector], dim=-1)  #batchsizeXdec_seqlenX3*hidden_size
        prefinal_tensor = self.statenctx_to_prefinal(decstate_and_context)   
        seqlen_last = prefinal_tensor.permute(0,2,1)   #batchsizeXpre_output_dimXdec_seqlen
        logits = self.output_projector(seqlen_last)
        logits = logits.permute(0,2,1)   # batchXdec_seqlenXvocab

        # now doing copymechanism
        if self.use_copy_mech:        
            probs_after_copying, _ = self.copymech(logits, attn_weights, concatenated_decoder_states, input_to_decoder.permute(1,0,2), context_vector, inp_with_oovs, max_oovs)        
            prob_to_return = probs_after_copying[0].squeeze(1)
        else:
            prob_to_return = self.softmax(logits).squeeze(1)
        
        return prob_to_return, decoder_hidden_state, context_vector, attn_weights
    
    
    
    def get_initial_state(self, start_ids, initial_decode_state):
        '''start_ids is tensor of size batchsizeXseqlen'''
        outp_embedded = self.output_embedder(start_ids)
        seqlen_first = outp_embedded.permute(1,0,2)
        feed=seqlen_first
        seqlen=feed.size(0)
        h_value, c_value =  initial_decode_state        
        for idx in range(seqlen):
            _ , (h_value, c_value) = self.decoder_rnn(feed[idx:idx+1], (h_value, c_value))
        
        return (h_value, c_value)


    @overrides
    def forward_on_instance(self, instance: SyncedFieldsInstance, decode_strategy) -> Dict[str, str]:
        """
        Takes an :class:`~allennlp.data.instance.Instance`, which typically has raw text in it,
        converts that text into arrays using this model's :class:`Vocabulary`, passes those arrays
        through :func:`self.forward()` and :func:`self.decode()` (which by default does nothing)
        and returns the result.  Before returning the result, we convert any
        ``torch.Tensors`` into numpy arrays and remove the batch dimension.
        """
        cuda_device = self._get_prediction_device()
        dataset = Batch([instance])
        dataset.index_instances(self.vocab)
        model_input = util.move_to_device(dataset.as_tensor_dict(), cuda_device)
        if decode_strategy=='beamsearch':
            output_ids, attn_dists = self.beam_search_decode(**model_input, min_length=self.min_decode_length, max_length=self.max_decode_length)
        elif decode_strategy=='greedy':
            output_ids, attn_dists, decoder_states = self.greedy_decode(**model_input, min_length=self.min_decode_length, max_length=self.max_decode_length)
        else:
            raise NotImplementedError

        output_words = []
        for _id in output_ids:
            if _id<self.vocab_size:
                output_words.append(self.vocab.get_token_from_index(_id))
            else:
                output_words.append(instance.oov_list[_id-self.vocab_size])

        assert output_words[0]==START_SYMBOL, "somehow the first symbol is not the START symbol. might be a bug"
        output_words=output_words[1:]
        attn_dists=attn_dists[1:]

        if output_words[-1]==END_SYMBOL:
            output_words = output_words[:-1]
            attn_dists=attn_dists[:-1]

        return {"tokens": output_words,
                "attn_dists": attn_dists}


    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics_to_return = {
            metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()
        }
        return metrics_to_return


    def beam_search_decode(self, source_tokens, target_tokens=None, meta=None, beam_width=4, min_length=35, max_length=120):
        inp_with_unks = source_tokens["ids_with_unks"]
        inp_with_oovs = source_tokens["ids_with_oovs"]
        max_oovs = int(torch.max(source_tokens["num_oovs"]))
        input_pad_mask=torch.where(inp_with_unks!=self.PAD_ID, self.true_rep, self.false_rep)
        inp_enc_seq, (intial_h_value, intial_c_value) = self.encode(inp_with_unks)
        h_value = self.pad_zeros_to_init_state(intial_h_value)
        c_value = self.pad_zeros_to_init_state(intial_c_value)        
        source_encoding=(h_value, c_value)

        # the first context vector is calculated by using the first lstm decoder state
        first_decoder_hstates_batchfirst = source_encoding[0].permute(1, 0, 2)
        first_decoder_cstates_batchfirst = source_encoding[1].permute(1, 0, 2)
        first_concatenated_decoder_states = torch.cat([first_decoder_cstates_batchfirst, first_decoder_hstates_batchfirst], dim=-1)
        first_context_vector, first_attn_dist = self.attention_layer(inp_enc_seq, first_concatenated_decoder_states, input_pad_mask)

        hypotheses = [   {"dec_state" : source_encoding,
                          "past_context_vector" : first_context_vector,
                          "logprobs" : [0.0],
                          "out_words" : [self.START_ID],
                          "attn_dists": [first_attn_dist]
                          }  ]

        finished_hypotheses = []
        
        def sort_hyps(list_of_hyps):
            return sorted(list_of_hyps, key=lambda x:sum(x["logprobs"])/len(x["logprobs"]), reverse=True)

        counter=0
        while counter<max_length and len(finished_hypotheses)<beam_width:
            counter+=1
            new_hypotheses=[]
            for hyp in hypotheses:
                old_out_words=hyp["out_words"]
                in_tok=hyp["out_words"][-1]
                if in_tok>=self.vocab_size:     # this guy is an OOV
                    in_tok=self.OOV_ID
                old_dec_state=hyp["dec_state"]
                past_context_vector=hyp["past_context_vector"]
                past_attn_dists=hyp["attn_dists"]
                old_logprobs=hyp["logprobs"]
                with torch.no_grad():
                    new_probs, new_dec_state, new_context_vector, attn_dist = self.decode_onestep( torch.tensor([[in_tok]]).cuda(), old_dec_state, past_context_vector, inp_enc_seq, inp_with_oovs, input_pad_mask, max_oovs)
                
                probs, indices = torch.topk(new_probs[0], dim=0, k=2*beam_width)
                for p, idx in zip(probs, indices):
                    new_dict = {"dec_state" : new_dec_state,
                                "past_context_vector" : new_context_vector,
                                "logprobs" : old_logprobs+[float(torch.log(p).detach().cpu().numpy())],
                                "out_words" : old_out_words+[idx.item()],
                                "attn_dists": past_attn_dists+[attn_dist]
                              }
                    new_hypotheses.append(new_dict)

            # time to pick the best of new hypotheses
            sorted_new_hypotheses = sort_hyps(new_hypotheses)
            hypotheses=[]
            for hyp in sorted_new_hypotheses:
                if hyp["out_words"][-1]==self.END_ID:
                    if len(hyp["out_words"])>min_length+1:
                        finished_hypotheses.append(hyp)
                else:
                    hypotheses.append(hyp) 
                if len(hypotheses) == beam_width or len(finished_hypotheses) == beam_width:
                    break
        
        # for hyp in finished_hypotheses:
        #     print(hyp["out_words"])
            
        if len(finished_hypotheses)>0:
            final_candidates = finished_hypotheses
        else:
            final_candidates = hypotheses
            
        sorted_final_candidates = sort_hyps(final_candidates)
        best_candidate = sorted_final_candidates[0]
        return best_candidate["out_words"], best_candidate["attn_dists"]


    def greedy_decode(self, source_tokens, target_tokens=None, meta=None, min_length=35, max_length=120):
        inp_with_unks = source_tokens["ids_with_unks"]
        inp_with_oovs = source_tokens["ids_with_oovs"]
        max_oovs = int(torch.max(source_tokens["num_oovs"]))
        input_pad_mask=torch.where(inp_with_unks!=self.PAD_ID, self.true_rep, self.false_rep)
        inp_enc_seq, (intial_h_value, intial_c_value) = self.encode(inp_with_unks)
        h_value = self.pad_zeros_to_init_state(intial_h_value)
        c_value = self.pad_zeros_to_init_state(intial_c_value)        
        source_encoding=(h_value, c_value)

        # the first context vector is calculated by using the first lstm decoder state
        first_decoder_hstates_batchfirst = source_encoding[0].permute(1, 0, 2)
        first_decoder_cstates_batchfirst = source_encoding[1].permute(1, 0, 2)
        first_concatenated_decoder_states = torch.cat([first_decoder_cstates_batchfirst, first_decoder_hstates_batchfirst], dim=-1)
        first_context_vector, first_attn_dist = self.attention_layer(inp_enc_seq, first_concatenated_decoder_states, input_pad_mask)

        hyp = {"dec_states" : [source_encoding],
              "past_context_vector" : first_context_vector,
              "logprobs" : [0.0],
              "out_words" : [self.START_ID],
              "attn_dists": [first_attn_dist]
                          }
        
        counter=0
        while counter<max_length:
            counter+=1

            old_out_words=hyp["out_words"]
            in_tok=hyp["out_words"][-1]
            if in_tok>=self.vocab_size:     # this guy is an OOV
                in_tok=self.OOV_ID
            old_dec_states=hyp["dec_states"]
            prev_dec_state = old_dec_states[-1]
            past_context_vector=hyp["past_context_vector"]
            old_logprobs=hyp["logprobs"]
            past_attn_dists=hyp["attn_dists"]
            new_probs, new_dec_state, new_context_vector, attn_dist = self.decode_onestep( torch.tensor([[in_tok]]).cuda(), prev_dec_state, past_context_vector, inp_enc_seq, inp_with_oovs, input_pad_mask, max_oovs)

            probs, indices = torch.topk(new_probs[0], dim=0, k=1)
            assert len(probs)==1 and len(indices)==1
            p = probs[0]
            idx = indices[0]

            hyp = {"dec_states" : old_dec_states+[new_dec_state],
                        "past_context_vector" : new_context_vector,
                        "logprobs" : old_logprobs+[float(torch.log(p).detach().cpu().numpy())],
                        "out_words" : old_out_words+[idx.item()],
                        "attn_dists": past_attn_dists+[attn_dist]
                      }
 
            # time to pick the best of new hypotheses
            if hyp["out_words"][-1]==self.END_ID:
                if len(hyp["out_words"])>min_length+1:
                    break
                    
        best_candidate = hyp

        # we need to get the h values out of the (h,c) lstm state tuples
        decoder_hstates_batchfirst = [decoder_hidden_state[0].permute(1, 0, 2) for decoder_hidden_state in best_candidate["dec_states"]]
        return best_candidate["out_words"], best_candidate["attn_dists"], decoder_hstates_batchfirst

