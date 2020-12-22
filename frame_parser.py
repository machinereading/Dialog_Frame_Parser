
# coding: utf-8

# In[1]:


import sys
import glob
import torch
sys.path.append('../')
import os
import numpy as np
from transformers import *
from src import utils
from src import thesis_utils
from src import dataio
import target_identifier
import inference
from src.modeling import *
from src.vtt_modeling import BIO_span_representation, transfer_model, transfer_spoken_model
from src.old_thesis_modeling import thesis_spoken_model
from koreanframenet.src import conll2textae
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch
from torch import nn
from torch.optim import Adam
from tqdm import tqdm, trange


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
# if device != "cpu":
#     torch.cuda.set_device(device)

print('\n###DEVICE:', device)


# In[30]:


class FrameParser():
    def __init__(self, fnversion=1.2, language='ko',masking=True, srl='framenet', 
                 model_path=False, gold_pred=False, viterbi=False, tgt=True, 
                 pretrained='bert-base-multilingual-cased', info=True, only_lu=True, adj=True, mode=""):
        self.fnversion = fnversion
        self.language = language
        self.masking = masking
        self.srl = srl
        self.gold_pred = gold_pred
        self.viterbi = viterbi
        self.pretrained = pretrained
        self.tgt = tgt #using <tgt> and </tgt> as a special token
        self.only_lu = only_lu
        self.adj = adj

        if info:
            print('srl model:', self.srl)
            print('language:', self.language)
            print('version:', self.fnversion)
            print('pretrained BERT:', self.pretrained)
            print('using TGT special token:', self.tgt)
        
        self.bert_io = thesis_utils.for_BERT(mode='predict', srl=self.srl, language=self.language,
                              masking=self.masking, fnversion=self.fnversion,
                              pretrained=self.pretrained, info=info)  
        
        #load model
        if model_path:
            self.model_path = model_path
        else:
            print('model_path={your_model_dir}')
#         self.model = torch.load(model_path, map_location=device)

        if self.srl == 'transfer_written':
            self.model = transfer_model.from_pretrained(self.model_path,
                                                               num_senses=len(self.bert_io.sense2idx),
                                                               num_args=len(self.bert_io.arg2idx),
                                                               frargmap=self.bert_io.frargmap)

        elif self.srl == 'transfer_spoken':
            self.model = transfer_spoken_model.from_pretrained(self.model_path,
                                                               num_senses=len(self.bert_io.sense2idx),
                                                               num_args=len(self.bert_io.arg2idx),
                                                               frargmap=self.bert_io.frargmap,
                                                               eval=True)
        elif self.srl == 'thesis_spoken':
            self.model = thesis_spoken_model.from_pretrained(self.model_path,
                                                               num_senses=len(self.bert_io.sense2idx),
                                                               num_args=len(self.bert_io.arg2idx),
                                                               frargmap=self.bert_io.frargmap,
                                                               eval=True)
        else:
            self.model = BertForJointShallowSemanticParsing.from_pretrained(self.model_path,
                                                                            num_senses=len(self.bert_io.sense2idx),
                                                                            num_args=len(self.bert_io.bio_arg2idx),
                                                                            lufrmap=self.bert_io.lufrmap,
                                                                            masking=self.masking,
                                                                            frargmap=self.bert_io.bio_frargmap)


        self.model.to(device)
        if info:
            print('...loaded model path:', self.model_path)
#         self.model = BertForJointShallowSemanticParsing
        self.model.eval()
        if info:
            print(self.model_path)
            print('...model is loaded')

        
    def parser_FrameType(self, input_d, sent_id=False, result_format=False, frame_candis=5):
        input_conll = dataio.preprocessor(input_d)
        
        #target identification
        if self.gold_pred:
            if len(input_conll[0]) == 2:
                pass
            else:
                input_conll = [input_conll]
            tgt_data = input_conll
        else:
            if self.srl == 'framenet':
                tgt_conll = self.targetid.target_id(input_conll)
            else:
                tgt_conll = self.targetid.pred_id(input_conll)
        
            # add <tgt> and </tgt> to target word
            tgt_data = dataio.data2tgt_data(tgt_conll, mode='parse')

        if tgt_data:
            
            # convert conll to bert inputs
            bert_inputs, args_inputs = self.bert_io.convert_to_bert_input_FrameType(tgt_data)
            dataloader = DataLoader(bert_inputs, sampler=None, batch_size=1)
            
            pred_senses, pred_args = [],[]
            sense_candis_list = []
            for batch in dataloader:
#                 torch.cuda.set_device(device)
                batch = tuple(t.to(device) for t in batch)
                b_input_ids, b_orig_tok_to_maps, b_lus, b_token_type_ids, b_masks = batch
                
                with torch.no_grad():
                    # tmp_eval_loss = self.model(b_input_ids, lus=b_lus,
                    #                            token_type_ids=b_token_type_ids, attention_mask=b_masks)
                    sense_logits = self.model(b_input_ids, lus=b_lus,
                                                          token_type_ids=b_token_type_ids, attention_mask=b_masks)
                

                lufr_masks = utils.get_masks(b_lus, 
                                             self.bert_io.lufrmap, 
                                             num_label=len(self.bert_io.sense2idx), 
                                             masking=self.masking).to(device)

                for b_idx in range(len(sense_logits)):
                    sense_logit = sense_logits[b_idx]
                    
                    lufr_mask = lufr_masks[b_idx]                        
                    masked_sense_logit = utils.masking_logit(sense_logit, lufr_mask)
                    pred_sense, sense_score = utils.logit2label(masked_sense_logit)
                    
                    sense_candis = utils.logit2candis(masked_sense_logit, 
                                                      candis=frame_candis, 
                                                      idx2label=self.bert_io.idx2sense)
                    sense_candis_list.append(sense_candis)

                    pred_senses.append([int(pred_sense)])

            pred_sense_tags = [self.bert_io.idx2sense[p_i] for p in pred_senses for p_i in p]

            conll_result = []

            for i in range(len(tgt_data)):
                
                raw = tgt_data[i]
                
                conll, toks, lus = [],[],[]
                for idx in range(len(raw[0])):
                    tok, lu = raw[0][idx], raw[1][idx]
                    if tok == '<tgt>' or tok == '</tgt>':
                        pass
                    else:
                        toks.append(tok)
                        lus.append(lu)
                conll.append(toks)
                conll.append(lus)
                
                sense_seq = ['_' for i in range(len(conll[1]))]
                for idx in range(len(conll[1])):
                    if conll[1][idx] != '_':
                        sense_seq[idx] = pred_sense_tags[i]
                        
                conll.append(sense_seq)
                conll_result.append(conll)
        else:
            conll_result = []
        
        result = conll_result
        return result

    def parser_args(self, input_d, sent_id=False, result_format=False, frame_candis=5):
        input_conll = dataio.preprocessor(input_d)

        # target identification
        if self.gold_pred:
            if len(np.shape(input_conll)) == 3: # train 시에 에러가 있는지 확인해야함.
                pass
            else:
                input_conll = [input_conll]
            tgt_data = input_conll
        else:
            if self.srl == 'framenet':
                tgt_conll = self.targetid.target_id(input_conll)
            else:
                tgt_conll = self.targetid.pred_id(input_conll)

            # add <tgt> and </tgt> to target word
            tgt_data = dataio.data2tgt_data(tgt_conll, mode='parse')

        if tgt_data:

            # convert conll to bert inputs
            args_inputs = self.bert_io.convert_to_bert_input_e2e(tgt_data)
            dataloader = DataLoader(args_inputs, sampler=None, batch_size=1)

            pred_labels, pred_starts, pred_ends, maps = [], [], [], []
            sense_candis_list = []
            for batch in dataloader:
                #                 torch.cuda.set_device(device)
                batch = tuple(t.to(device) for t in batch)
                inputs = {
                    "input_ids": batch[0],
                    "input_orig_tok_to_maps": batch[1],
                    "lus": batch[2],
                    "senses": batch[3],
                    "token_type_ids": batch[4],
                    "attention_mask": batch[5],
                    "invalid_pos": batch[6]
                }

                with torch.no_grad():
                    # tmp_eval_loss = self.model(b_input_ids, lus=b_lus,
                    #                            token_type_ids=b_token_type_ids, attention_mask=b_masks)
                    b_spans, b_probs = self.model(**inputs)


                for b_idx in range(len(b_spans)):
                    spans = b_spans[b_idx]
                    probs = b_probs[b_idx]  # [len(spans), #fe]
                    frame = inputs["senses"][b_idx].item()
                    candidates = self.bert_io.frargmap[str(frame)]
                    maps.append(inputs["input_orig_tok_to_maps"][b_idx])
                    for ii, span in enumerate(spans):
                        st, en = span.st, span.en
                        prob = probs[ii]
                        pred_starts.append(st)
                        pred_ends.append(en)
                        try:
                            label = candidates[torch.argmax(prob).item()]
                        except:
                            label = -1
                        pred_labels.append(label)

            # pred_sense_tags = [self.bert_io.idx2sense[p_i] for p in pred_senses for p_i in p]

            conll_result = []

            for i in range(len(tgt_data)):
                raw = tgt_data[i]
                conll, toks, lus, senses, args = [], [], [], [], []
                arg_seq = ['O' for i in range(len(raw[0]))]
                map = maps[i].tolist()

                for idx in range(len(pred_starts)):
                    st = pred_starts[idx]
                    en = pred_ends[idx]
                    if pred_labels[idx] == -1:
                        # label = "dummy"
                        continue
                    else:
                        label = self.bert_io.idx2arg[pred_labels[idx]]


                    if st == 0 or en == 0:
                        continue
                    if st > en:
                        continue

                    start, end = -1, -1

                    for x, y in enumerate(map):
                        if y <= st:
                            start = x
                        else:
                            break

                    end = len(raw[0]) - 1
                    for x, y in enumerate(map):
                        if y == -1:
                            break
                        if y == en:
                            end = x
                            break
                        if y < en:
                            end = x


                    if start == -1 or end == -1:
                        continue

                    for ii in range(start, end+1):
                        if ii == start:
                            try:
                                arg_seq[ii] = "B-" + label
                            except:
                                break
                        else:
                                arg_seq[ii] = "I-" + label


                for idx in range(len(raw[0])):
                    tok, lu, sense, arg = raw[0][idx], raw[1][idx], raw[2][idx], arg_seq[idx]
                    if tok == '<tgt>' or tok == '</tgt>':
                        pass
                    else:
                        toks.append(tok)
                        lus.append(lu)
                        senses.append(sense)
                        args.append(arg)
                conll.append(toks)
                conll.append(lus)
                conll.append(senses)
                conll.append(args)
                conll_result.append(conll)

        else:
            conll_result = []
        result = conll_result
        return result

    def parser_thesis_spoken(self, input_d, sent_id=False, result_format=False, frame_candis=5, mode=''):
        # data = [input_d]
        golds, preds = [],[]
        args_inputs = self.bert_io.data_converter(input_d)
        if args_inputs == None:
            return None, 0
        dataloader = DataLoader(args_inputs, sampler=None, batch_size=1)

        for batch in dataloader:
            batch = tuple(t.to(device) for t in batch)

            inputs = {
                "input_ids": batch[0],
                "utter_lens": batch[1],
                "orig_tok_to_maps": batch[2],
                "token_type_ids": batch[3],
                "attention_mask": batch[4],
                "targets": batch[5],
                "senses": batch[6],
                "gold_args": batch[7],
                "arg_lens": batch[8],
                "speakers": batch[9],
                "speaker_lens": batch[10],
                "special_tokens": batch[11],
                "use_gold_span": False,
                "use_pruning": True
            }

            with torch.no_grad():
                pred = self.model(**inputs, train=False)
            gold = [batch[6].tolist()[0][-1]]
            golds += gold
            preds += pred

        return golds, preds

    def parser_transfer_spoken(self, input_d, sent_id=False, result_format=False, frame_candis=5, mode=''):
        # if mode == 'demo':
        #     device = "cuda:0"
        data = [input_d]
        golds, preds = [],[]
        args_inputs = self.bert_io.convert_to_bert_input_transfer_spoken(data)
        if args_inputs == None:
            return None, 0
        dataloader = DataLoader(args_inputs, sampler=None, batch_size=1)

        for batch in dataloader:
            #                 torch.cuda.set_device(device)
            batch = tuple(t.to(device) for t in batch)

            inputs = {
                "input_ids": batch[0],
                "token_type_ids": batch[2],
                "attention_mask": batch[3],
                "lu_spans": batch[4],
                "senses": batch[5],
                "gold_spans": batch[6],
                "lu_speakers": batch[7],
                "fe_speakers": batch[8]
            }

            with torch.no_grad():
                pred = self.model(**inputs, train=False)
            gold = [batch[6].tolist()[0][-1]]
            golds += gold
            preds += pred

        return golds, preds

    def parser_transfer_written(self, input_d, sent_id=False, result_format=False, frame_candis=5):
        input_conll = dataio.preprocessor(input_d)

        gold_span = 0
        find_span = 0
        correct_span = 0

        # target identification
        if self.gold_pred:
            if len(np.shape(input_conll)) == 3: # train 시에 에러가 있는지 확인해야함.
                pass
            else:
                input_conll = [input_conll]
            tgt_data = input_conll
        else:
            if self.srl == 'framenet':
                tgt_conll = self.targetid.target_id(input_conll)
            else:
                tgt_conll = self.targetid.pred_id(input_conll)

            # add <tgt> and </tgt> to target word
            # tgt_data = dataio.data2tgt_data(tgt_conll, mode='parse')

        if tgt_data:

            # convert conll to bert inputs
            args_inputs = self.bert_io.convert_to_bert_input_transfer_written(tgt_data)
            if args_inputs == None:
                return None, 0
            dataloader = DataLoader(args_inputs, sampler=None, batch_size=1)

            pred_labels, pred_starts, pred_ends, maps = [], [], [], []
            for batch in dataloader:
                #                 torch.cuda.set_device(device)
                batch = tuple(t.to(device) for t in batch)

                b_input_ids, b_input_orig_tok_to_maps, b_input_lus, b_input_senses, b_input_args, b_token_type_ids, b_input_masks, \
                b_gold_spans, b_span_pad, b_lu_spans = batch

                with torch.no_grad():
                    # tmp_eval_loss = self.model(b_input_ids, lus=b_lus,
                    #                            token_type_ids=b_token_type_ids, attention_mask=b_masks)
                    preds = self.model(b_input_ids, lus=b_input_lus, senses=b_input_senses, args=b_input_args,
                         token_type_ids=b_token_type_ids, attention_mask=b_input_masks, gold_spans=b_gold_spans, span_pads=b_span_pad, lu_spans=b_lu_spans, train=False)

                for b_idx in range(len(preds)):
                    pred_start, pred_end, pred_label = [], [], []
                    pred = preds[b_idx]
                    maps.append(b_input_orig_tok_to_maps[b_idx])
                    gold_span = b_gold_spans[b_idx][:b_span_pad[b_idx].item()]

                    assert len(pred) == len(gold_span)

                    for t in gold_span:
                        st, en = t[0].item(), t[1].item()
                        pred_start.append(st)
                        pred_end.append(en)
                    pred_starts.append(pred_start)
                    pred_ends.append(pred_end)
                    pred_labels.append(pred)

            # pred_sense_tags = [self.bert_io.idx2sense[p_i] for p in pred_senses for p_i in p]

            conll_result = []

            for i in range(len(tgt_data)):
                raw = tgt_data[i]
                conll, toks, lus, senses, args = [], [], [], [], []
                arg_seq = ['O' for i in range(len(raw[0]))]
                map = maps[i].tolist()

                for idx in range(len(pred_starts[i])):
                    if len(pred_starts[i]) == 0:  # no span
                        continue
                    st = pred_starts[i][idx]
                    en = pred_ends[i][idx]

                    if pred_labels[i][idx] == -1:
                        # label = "dummy"
                        continue
                    else:
                        label = self.bert_io.idx2arg[pred_labels[i][idx]]

                    if st == 0 or en == 0:
                        continue
                    if st > en:
                        continue

                    start, end = -1, -1

                    for x, y in enumerate(map):
                        if y <= st:
                            start = x
                        else:
                            break

                    end = len(raw[0]) - 1
                    for x, y in enumerate(map):
                        if y == -1:
                            break
                        if y == en:
                            end = x
                            break
                        if y < en:
                            end = x


                    if start == -1 or end == -1:
                        continue

                    for ii in range(start, end+1):
                        if ii == start:
                            try:
                                arg_seq[ii] = "B-" + label
                            except:
                                break
                        else:
                            try:
                                arg_seq[ii] = "I-" + label
                            except:
                                break


                for idx in range(len(raw[0])):
                    tok, lu, sense, arg = raw[0][idx], raw[1][idx], raw[2][idx], arg_seq[idx]
                    if tok == '<tgt>' or tok == '</tgt>':
                        pass
                    else:
                        toks.append(tok)
                        lus.append(lu)
                        senses.append(sense)
                        args.append(arg)
                conll.append(toks)
                conll.append(lus)
                conll.append(senses)
                conll.append(args)
                conll_result.append(conll)

        else:
            conll_result = []
        result = conll_result
        return result, find_span

    def parser_bio_span(self, input_d, sent_id=False, result_format=False, frame_candis=5):
        input_conll = dataio.preprocessor(input_d)

        gold_span = 0
        find_span = 0
        correct_span = 0

        # target identification
        if self.gold_pred:
            if len(np.shape(input_conll)) == 3: # train 시에 에러가 있는지 확인해야함.
                pass
            else:
                input_conll = [input_conll]
            tgt_data = input_conll
        else:
            if self.srl == 'framenet':
                tgt_conll = self.targetid.target_id(input_conll)
            else:
                tgt_conll = self.targetid.pred_id(input_conll)

            # add <tgt> and </tgt> to target word
            tgt_data = dataio.data2tgt_data(tgt_conll, mode='parse')

        if tgt_data:

            # convert conll to bert inputs
            args_inputs = self.bert_io.convert_to_bert_input_bio_span(tgt_data)
            dataloader = DataLoader(args_inputs, sampler=None, batch_size=1)

            pred_labels, pred_starts, pred_ends, maps, pred_frames = [], [], [], [], []
            sense_candis_list = []
            for batch in dataloader:
                #                 torch.cuda.set_device(device)
                batch = tuple(t.to(device) for t in batch)

                b_input_ids, b_orig_tok_to_maps, b_lus, b_token_type_ids, b_masks = batch
                inputs = {
                    "input_ids": b_input_ids,
                    # "input_orig_tok_to_maps": b_orig_tok_to_maps,
                    "lus": b_lus,
                    "token_type_ids": b_token_type_ids,
                    "attention_mask": b_masks
                }

                with torch.no_grad():
                    # tmp_eval_loss = self.model(b_input_ids, lus=b_lus,
                    #                            token_type_ids=b_token_type_ids, attention_mask=b_masks)
                    pred_frame, pred_spans, pred_scores = self.model(**inputs)

                for b_idx in range(len(pred_frame)):
                    pred_start, pred_end, pred_label = [], [], []
                    frame = pred_frame[b_idx].item()
                    pred_frames.append(frame)
                    spans = pred_spans[b_idx]
                    scores = pred_scores[b_idx]  # [len(spans), #fe]
                    candidates = self.bert_io.frargmap[str(frame)]
                    maps.append(b_orig_tok_to_maps[b_idx])
                    find_span += len(spans)
                    for ii, (st,en) in enumerate(spans):
                        pred_start.append(st)
                        pred_end.append(en)
                        score = scores[ii]
                        try:
                            label = candidates[torch.argmax(score).item()]
                        except:
                            label= -1

                        pred_label.append(label)
                    pred_starts.append(pred_start)
                    pred_ends.append(pred_end)
                    pred_labels.append(pred_label)

            # pred_sense_tags = [self.bert_io.idx2sense[p_i] for p in pred_senses for p_i in p]

            conll_result = []

            for i in range(len(tgt_data)):
                raw = tgt_data[i]
                conll, toks, lus, senses, args = [], [], [], [], []
                arg_seq = ['O' for i in range(len(raw[0]))]
                map = maps[i].tolist()

                for idx in range(len(pred_starts[i])):
                    if len(pred_starts[i]) == 0:  # no span
                        continue
                    st = pred_starts[i][idx]
                    en = pred_ends[i][idx]



                    a = 1

                    if pred_labels[i][idx] == -1:
                        # label = "dummy"
                        continue
                    else:
                        label = self.bert_io.idx2arg[pred_labels[i][idx]]

                    if st == 0 or en == 0:
                        continue
                    if st > en:
                        continue



                    ##


                    start, end = -1, -1

                    for x, y in enumerate(map):
                        if y <= st:
                            start = x
                        else:
                            break

                    end = len(raw[0]) - 1
                    for x, y in enumerate(map):
                        if y == -1:
                            break
                        if y == en:
                            end = x
                            break
                        if y < en:
                            end = x


                    if start == -1 or end == -1:
                        continue

                    for ii in range(start, end+1):
                        if ii == start:
                            try:
                                arg_seq[ii] = "B-" + label
                            except:
                                break
                        else:
                            try:
                                arg_seq[ii] = "I-" + label
                            except:
                                break


                for idx in range(len(raw[0])):
                    tok, lu, sense, arg = raw[0][idx], raw[1][idx], raw[2][idx], arg_seq[idx]
                    if tok == '<tgt>' or tok == '</tgt>':
                        pass
                    else:
                        toks.append(tok)
                        lus.append(lu)
                        if sense == '_':
                            senses.append(sense)
                        else:
                            senses.append(self.bert_io.idx2sense[pred_frames[i]])
                        args.append(arg)
                conll.append(toks)
                conll.append(lus)
                conll.append(senses)
                conll.append(args)
                conll_result.append(conll)

        else:
            conll_result = []
        result = conll_result
        return result, find_span

    def parser(self, input_d, sent_id=False, result_format=False, frame_candis=5):
        input_conll = dataio.preprocessor(input_d)

        # target identification
        if self.gold_pred:
            if len(input_conll[0]) == 2:
                pass
            else:
                input_conll = [input_conll]
            tgt_data = input_conll
        else:
            if self.srl == 'framenet':
                tgt_conll = self.targetid.target_id(input_conll)
            else:
                tgt_conll = self.targetid.pred_id(input_conll)

            # add <tgt> and </tgt> to target word
            tgt_data = dataio.data2tgt_data(tgt_conll, mode='parse')

        if tgt_data:

            # convert conll to bert inputs
            bert_inputs = self.bert_io.convert_to_bert_input_JointShallowSemanticParsing(tgt_data)
            dataloader = DataLoader(bert_inputs, sampler=None, batch_size=1)

            pred_senses, pred_args = [], []
            sense_candis_list = []
            for batch in dataloader:
                #                 torch.cuda.set_device(device)
                batch = tuple(t.to(device) for t in batch)
                b_input_ids, b_orig_tok_to_maps, b_lus, b_token_type_ids, b_masks = batch

                with torch.no_grad():
                    sense_logits, arg_logits = self.model(b_input_ids, lus=b_lus,
                                                          token_type_ids=b_token_type_ids, attention_mask=b_masks)

                lufr_masks = utils.get_masks(b_lus,
                                             self.bert_io.lufrmap,
                                             num_label=len(self.bert_io.sense2idx),
                                             masking=self.masking).to(device)

                b_input_ids_np = b_input_ids.detach().cpu().numpy()
                arg_logits_np = arg_logits.detach().cpu().numpy()

                b_input_ids, arg_logits = [], []

                for b_idx in range(len(b_orig_tok_to_maps)):
                    orig_tok_to_map = b_orig_tok_to_maps[b_idx]
                    bert_token = self.bert_io.tokenizer.convert_ids_to_tokens(b_input_ids_np[b_idx])
                    tgt_idx = utils.get_tgt_idx(bert_token, tgt=self.tgt)

                    input_id, sense_logit, arg_logit = [], [], []

                    for idx in orig_tok_to_map:  # 어절의 첫번째 token인 애들의 로짓 중 1번째에 .. -inf 부여?
                        if idx != -1:
                            if idx not in tgt_idx:
                                try:
                                    input_id.append(b_input_ids_np[b_idx][idx])
                                    arg_logits_np[b_idx][idx][1] = np.NINF
                                    arg_logit.append(arg_logits_np[b_idx][idx])
                                except KeyboardInterrupt:
                                    raise
                                except:
                                    pass

                    b_input_ids.append(input_id)
                    arg_logits.append(arg_logit)

                b_input_ids = torch.Tensor(b_input_ids).to(device)
                arg_logits = torch.Tensor(arg_logits).to(device)

                for b_idx in range(len(sense_logits)):
                    input_id = b_input_ids[b_idx]
                    sense_logit = sense_logits[b_idx]
                    arg_logit = arg_logits[b_idx]

                    lufr_mask = lufr_masks[b_idx]
                    masked_sense_logit = utils.masking_logit(sense_logit, lufr_mask)
                    pred_sense, sense_score = utils.logit2label(masked_sense_logit)

                    sense_candis = utils.logit2candis(masked_sense_logit,
                                                      candis=frame_candis,
                                                      idx2label=self.bert_io.idx2sense)
                    sense_candis_list.append(sense_candis)

                    if self.srl == 'framenet':
                        arg_logit_np = arg_logit.detach().cpu().numpy()
                        arg_logit = []
                        frarg_mask = utils.get_masks([pred_sense],
                                                     self.bert_io.bio_frargmap,
                                                     num_label=len(self.bert_io.bio_arg2idx),
                                                     masking=True).to(device)[0]
                        for logit in arg_logit_np:
                            masked_logit = utils.masking_logit(logit, frarg_mask)
                            arg_logit.append(np.array(masked_logit))
                        arg_logit = torch.Tensor(arg_logit).to(device)
                    else:
                        pass

                    pred_arg = []
                    for logit in arg_logit:  # word의 head인 애들만.
                        label, score = utils.logit2label(logit)
                        pred_arg.append(int(label))

                    pred_senses.append([int(pred_sense)])
                    pred_args.append(pred_arg)

            pred_sense_tags = [self.bert_io.idx2sense[p_i] for p in pred_senses for p_i in p]
            if self.srl == 'framenet':
                pred_arg_tags = [[self.bert_io.idx2bio_arg[p_i] for p_i in p] for p in pred_args]
            elif self.srl == 'framenet-argid':
                pred_arg_tags = [[self.bert_io.idx2bio_argument[p_i] for p_i in p] for p in pred_args]
            else:
                pred_arg_tags = [[self.bert_io.idx2bio_arg[p_i] for p_i in p] for p in pred_args]

            conll_result = []

            for i in range(len(pred_arg_tags)):

                raw = tgt_data[i]

                conll, toks, lus = [], [], []
                for idx in range(len(raw[0])):
                    tok, lu = raw[0][idx], raw[1][idx]
                    if tok == '<tgt>' or tok == '</tgt>':
                        pass
                    else:
                        toks.append(tok)
                        lus.append(lu)
                conll.append(toks)
                conll.append(lus)

                sense_seq = ['_' for i in range(len(conll[1]))]
                for idx in range(len(conll[1])):
                    if conll[1][idx] != '_':
                        sense_seq[idx] = pred_sense_tags[i]

                conll.append(sense_seq)
                conll.append(pred_arg_tags[i])

                conll_result.append(conll)
        else:
            conll_result = []

        result = []
        if result_format == 'all':
            result = {}
            result['conll'] = conll_result

            if conll_result:
                textae = conll2textae.get_textae(conll_result)
                frdf = dataio.frame2rdf(conll_result, sent_id=sent_id)
                topk = dataio.topk(conll_result, sense_candis_list)
            else:
                textae = []
                frdf = []
                topk = {}
            result['textae'] = textae
            result['graph'] = frdf
            result['topk'] = topk
        elif result_format == 'textae':
            if conll_result:
                textae = conll2textae.get_textae(conll_result)
            else:
                textae = []
            result = textae
        elif result_format == 'graph':
            if conll_result:
                frdf = dataio.frame2rdf(conll_result, sent_id=sent_id, language=self.language)
            else:
                frdf = []
            result = frdf
        elif result_format == 'topk':
            if conll_result:
                topk = dataio.topk(conll_result, sense_candis_list)
            else:
                topk = {}
            result = topk
        else:
            result = conll_result

        return result