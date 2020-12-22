import json
import sys
import torch
from transformers import *
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from torch import nn

sys.path.insert(0,'../')
sys.path.insert(0,'../../')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()

MAX_LEN = 256

import os
try:
    dir_path = os.path.dirname(os.path.abspath( __file__ ))
except:
    dir_path = '.'
    
dir_path = dir_path+'/..'

class for_BERT():
    def __init__(self, srl='framenet', language='ko', fnversion=1.2, mode='train', masking=True, pretrained='bert-base-multilingual-cased', task='', use_definition=False, info=True):
        self.mode = mode
        self.masking = masking
        self.srl = srl
        self.definitions = None

        self.span2idx = {
            'X':0,
            'O':1,
            'B':2,
            'I':3
        }
        self.idx2span = dict(zip(self.span2idx.values(), self.span2idx.keys()))

        if 'multilingual' in pretrained:
            vocab_file_path = dir_path+'/data/bert-multilingual-cased-dict-add-tgt'
            self.tokenizer = BertTokenizer(vocab_file_path, do_lower_case=False, max_len=256)
            self.tokenizer.additional_special_tokens = ['<tgt>', '</tgt>']
        elif 'large' in pretrained:
            vocab_file_path = dir_path+'/data/bert-large-cased-dict-add-tgt'
            self.tokenizer = BertTokenizer(vocab_file_path, do_lower_case=False, max_len=256)
            self.tokenizer.additional_special_tokens = ['<tgt>', '</tgt>']
        else:
            vocab_file_path = dir_path+'/data/bert-multilingual-cased-dict-add-tgt'
            self.tokenizer = BertTokenizer(vocab_file_path, do_lower_case=False, max_len=256)
            self.tokenizer.additional_special_tokens = ['<tgt>', '</tgt>']

        if language == 'en':
            fnversion=1.7
            data_path = dir_path+'/koreanframenet/resource/info/fn'+str(fnversion)+'_'
        elif language == 'ko':
            data_path = dir_path+'/koreanframenet/resource/info/kfn'+str(fnversion)+'_'
            with open(dir_path + '/koreanframenet/data/1.2/used_FE.json', 'r') as f:
                self.used_fe = json.load(f)
        elif 'mul' in language:
            data_path = dir_path+'/koreanframenet/resource/info/mul_'
        else:
            data_path = dir_path+'/koreanframenet/resource/info/kfn'+str(fnversion)+'_'


        
        # lu dic = multilingual
        with open(data_path+'lu2idx.json','r') as f:
            self.lu2idx = json.load(f)
        self.idx2lu = dict(zip(self.lu2idx.values(),self.lu2idx.keys()))
        
        # frame, fe dic = FN1.7
        fname = dir_path+'/koreanframenet/resource/info/fn1.7_frame2idx.json'
        with open(fname,'r') as f:
            self.sense2idx = json.load(f)

        with open(data_path+'lufrmap.json','r') as f:
            self.lufrmap = json.load(f) # lu idx와 해당 lu가 속할 수 있는 frame indices.

        with open(dir_path+'/koreanframenet/resource/info/fn1.7_fe2idx.json','r') as f:
            self.arg2idx = json.load(f)

        with open(dir_path+'/koreanframenet/resource/info/fn1.7_bio_fe2idx.json','r') as f:
            self.bio_arg2idx = json.load(f)  #각 FE에 대한 BI, O 태그 indices
        self.idx2bio_arg = dict(zip(self.bio_arg2idx.values(),self.bio_arg2idx.keys()))
            
        with open(dir_path+'/data/bio_arg2idx.json','r') as f:
            self.bio_argument2idx = json.load(f) # O, X, B-ARG, I-ARG
        self.idx2bio_argument = dict(zip(self.bio_argument2idx.values(),self.bio_argument2idx.keys()))

        with open(dir_path+'/data/framenet_info.json','r') as f:
            self.frame_info = json.load(f)

        with open(dir_path+'/koreanframenet/resource/info/fn1.7_frame_definitions.json','r') as f:
            self.frame_def = json.load(f)
            
        if language == 'en':
            frargmap_path = dir_path+'/koreanframenet/resource/info/fn1.7_bio_frargmap.json'
        else:
            frargmap_path = dir_path+'/koreanframenet/resource/info/mul_bio_frargmap.json'
            frargmap_path2 = dir_path + '/koreanframenet/resource/info/mul_frargmap.json'

        with open(frargmap_path,'r') as f:
            self.bio_frargmap = json.load(f)
        with open(frargmap_path2,'r') as f:
            self.frargmap = json.load(f)
            
        if info:
            print('used dictionary:')
            print('\t', data_path+'lu2idx.json')
            print('\t', data_path+'lufrmap.json')
            print('\t', frargmap_path)
            
        self.idx2sense = dict(zip(self.sense2idx.values(),self.sense2idx.keys()))
        self.idx2arg = dict(zip(self.arg2idx.values(),self.arg2idx.keys()))

        self.speaker2idx = {'Seohee': 0, 'Yijoon': 1, 'Jeongsuk': 2, 'Heeran': 3, 'Haeyoung2': 4, 'Jiya': 5, 'Deogi': 6, 'Soontack': 7, 'Anna': 8, 'Chairman': 9, 'Gitae': 10, 'Kyungsu': 11, 'Hun': 12, 'Sukyung': 13, 'Dokyung': 14, 'Sangseok': 15, 'Taejin': 16, 'Sungjin': 17, 'Jinsang': 18, 'Haeyoung1': 19, 'None': 20}
        self.idx2speaker = dict(zip(self.speaker2idx.values(), self.speaker2idx.keys()))

        if use_definition:
            self.definitions = {}
            for k, v in self.frame_info.items():
                fes = v['fes']
                tokenized_fes = {}
                for fe_name, fe_v in fes.items():
                    orig_tokens, bert_tokens, orig_to_tok_map = self.bert_tokenizer(fe_v['definition'])
                    tokenized_fes[fe_name] = {
                        'tokens': bert_tokens,
                        'map': orig_to_tok_map
                    }
                self.definitions[k] = tokenized_fes

    def idx2tag(self, predictions, model='senseid'):
        if model == 'senseid':
            pred_tags = [self.idx2sense[p_i] for p in predictions for p_i in p]
        elif model == 'argid-dp':
            pred_tags = [self.idx2arg[p_i] for p in predictions for p_i in p]
        elif model == 'argid-span':
            pred_tags = [self.idx2bio_arg[p_i] for p in predictions for p_i in p]
        return pred_tags
    
    def bert_tokenizer(self, text):
        orig_tokens = text.split(' ')
        bert_tokens = []
        orig_to_tok_map = []
        bert_tokens.append("[CLS]")
        for orig_token in orig_tokens:
            orig_to_tok_map.append(len(bert_tokens))
            bert_tokens.extend(self.tokenizer.tokenize(orig_token))
        bert_tokens.append("[SEP]")

        return orig_tokens, bert_tokens, orig_to_tok_map

    def convert_to_bert_input_FrameType(self, input_data):
        definitions = self.definitions
        tokenized_texts, lus, senses, args = [], [], [], []
        orig_tok_to_maps = []

        args_tokenized_texts, args_start_positions, args_end_positions, args_labels, args_token_maps = [], [], [], [], []

        for i in range(len(input_data)):
            data = input_data[i]
            text = ' '.join(data[0])
            orig_tokens, bert_tokens, orig_to_tok_map = self.bert_tokenizer(text)
            # orig_to_tok_map : origin token에 대응하는 bert start token idx

            orig_tok_to_maps.append(orig_to_tok_map)
            tokenized_texts.append(bert_tokens)

            ori_lus = data[1]
            lu_sequence = []
            for i in range(len(bert_tokens)):
                if i in orig_to_tok_map:
                    idx = orig_to_tok_map.index(i)
                    l = ori_lus[idx]
                    lu_sequence.append(l)
                else:
                    lu_sequence.append('_')
            lus.append(lu_sequence)  # bert token에 맞춘 lu(ex-이익.n), 단 매칭되는 ori token의 첫번째 bert token에 달린다.

            ori_senses = data[2]
            try:  # train
                ori_args = data[3]
            except:  # eval
                ori_args = None
            if self.mode == 'train':
                sense_sequence, arg_sequence = [], []
                for i in range(len(bert_tokens)):
                    if i in orig_to_tok_map:
                        idx = orig_to_tok_map.index(i)
                        fr = ori_senses[idx]
                        sense_sequence.append(fr)
                        if ori_args == None: # eval
                            ar = 'O'
                        else: # train
                            ar = ori_args[idx]
                        arg_sequence.append(ar)
                    else:
                        sense_sequence.append('_')
                        arg_sequence.append('X')
                senses.append(sense_sequence)
                args.append(arg_sequence)

            # args_tokenized_texts 여기에 완성된 text 추가.
            cur_sense = [x for x in ori_senses if x != '_']
            if ori_args == None: # eval
                cur_args = []
            else:
                cur_args = list(set([a.split('-')[-1] for a in ori_args if a[0] == 'B' or a[0] == 'I']))
            for cur_arg in cur_args:
                try:
                    definition = definitions[cur_sense[0]][cur_arg]['tokens']
                except:
                    continue

                indices = []
                for ii, x in enumerate(ori_args):
                    if x.find(cur_arg) != -1:
                        indices.append(ii)

                st = orig_to_tok_map[indices[0]] - 1 + len(definition)
                try:
                    en = orig_to_tok_map[indices[-1]+1] - 1 - 1 + len(definition)
                except:
                    en = len(definition + bert_tokens[1:]) - 2
                    if bert_tokens[-1] != '[SEP]':
                        en += 1

                if True: # 사이즈 넘는건 무시함.
                    if len(definition + bert_tokens[1:]) >= MAX_LEN:
                        continue
                if en >= MAX_LEN:
                    continue
                args_token_map = [x+len(definitions[cur_sense[0]][cur_arg]['tokens'])-1 for x in orig_to_tok_map]

                args_token_maps.append(args_token_map)
                args_start_positions.append(st)
                args_end_positions.append(en)
                args_tokenized_texts.append(definition + bert_tokens[1:])
                args_labels.append(self.arg2idx[cur_arg])


            for cur_arg in definitions[cur_sense[0]]:
                if cur_sense[0] not in self.used_fe.keys():    # train set 내에서 등장하지 않은 sense..?
                    continue
                if cur_arg not in self.used_fe[cur_sense[0]]:  # train set 내에서 등장하지 않은 (sense-element) pair이면 보지않는다.
                    continue

                if cur_arg in cur_args:  # 위에서 고려 되었다.
                    continue
                try:
                    definition = definitions[cur_sense[0]][cur_arg]['tokens']
                except:
                    continue

                if True:  # 사이즈 넘는건 무시함.
                    if len(definition + bert_tokens[1:]) >= MAX_LEN:
                        continue
                try:
                    args_labels.append(self.arg2idx[cur_arg])
                except:
                    continue
                args_token_map = [x + len(definitions[cur_sense[0]][cur_arg]['tokens']) - 1 for x in orig_to_tok_map]
                args_token_maps.append(args_token_map)
                args_start_positions.append(0)
                args_end_positions.append(0)
                args_tokenized_texts.append(definition + bert_tokens[1:])




        input_ids = pad_sequences([self.tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                                  maxlen=MAX_LEN, dtype="long", truncating="post",
                                  padding="post")  # bert token to idx & append padding

        orig_tok_to_maps = pad_sequences(orig_tok_to_maps, maxlen=MAX_LEN, dtype="long", truncating="post",
                                         padding="post", value=-1)

        args_ids = pad_sequences([self.tokenizer.convert_tokens_to_ids(txt) for txt in args_tokenized_texts],
                                  maxlen=MAX_LEN, dtype="long", truncating="post",
                                  padding="post")  # bert token to idx & append padding




        lu_seq, sense_seq = [], []
        for sent_idx in range(len(lus)):
            lu_items = lus[sent_idx]
            lu = []
            for idx in range(len(lu_items)):
                if lu_items[idx] != '_':
                    if len(lu) == 0:
                        if self.mode != 'train' and self.masking == False:
                            lu.append(1)
                        else:
                            lu.append(self.lu2idx[lu_items[idx]])

            lu_seq.append(lu)

            if self.mode == 'train':
                sense_items = senses[sent_idx]
                sense = []
                for idx in range(len(sense_items)):
                    if sense_items[idx] != '_':
                        if len(sense) == 0:
                            sense.append(self.sense2idx[sense_items[idx]])
                sense_seq.append(sense)


        attention_masks = [[float(i > 0) for i in ii] for ii in input_ids]
        token_type_ids = [[0 if idx > 0 else 1 for idx in input_id] for input_id in input_ids]

        data_inputs = torch.tensor(input_ids)
        data_orig_tok_to_maps = torch.tensor(orig_tok_to_maps)
        data_lus = torch.tensor(lu_seq)
        data_token_type_ids = torch.tensor(token_type_ids)
        data_masks = torch.tensor(attention_masks)

        if self.mode == 'train':
            data_senses = torch.tensor(sense_seq)
            bert_inputs = TensorDataset(data_inputs, data_orig_tok_to_maps, data_lus, data_senses,
                                        data_token_type_ids, data_masks)
        else:
            bert_inputs = TensorDataset(data_inputs, data_orig_tok_to_maps, data_lus, data_token_type_ids, data_masks)

        arg_attention_masks = [[float(i > 0) for i in ii] for ii in args_ids]
        arg_token_type_ids = []
        x = 0

        sep_id = self.tokenizer.convert_tokens_to_ids('[SEP]')

        for args_id in args_ids:
            arg_token_type_id = []
            for cur in args_id:
                arg_token_type_id.append(x)
                if cur == sep_id:
                    x = 1
            arg_token_type_ids.append(arg_token_type_id)

        args_token_maps = pad_sequences(args_token_maps, maxlen=MAX_LEN, dtype="long", truncating="post",
                                         padding="post", value=-1)

        arg_inputs = torch.tensor(args_ids)
        arg_starts = torch.tensor(args_start_positions)
        arg_ends = torch.tensor(args_end_positions)
        arg_token_type_ids = torch.tensor(arg_token_type_ids)
        arg_attention_masks = torch.tensor(arg_attention_masks)
        args_labels = torch.tensor(args_labels)
        args_token_maps = torch.tensor(args_token_maps) # ValueError: expected sequence of length 31 at dim 1 (got 16)

        args_inputs = TensorDataset(arg_inputs, arg_attention_masks, arg_token_type_ids,
                                    arg_starts, arg_ends, args_labels, args_token_maps)
        return bert_inputs, args_inputs
        # else:  # eval의 경우 dataset 만드는 함수 따로 만들어 줘야함.. seq한 2단계를 거치기 때문에 한번에 만들 수 없음.
        #     args_inputs = TensorDataset(arg_inputs, arg_starts, arg_ends,
        #                                 data_token_type_ids, data_masks)

    def convert_to_bert_input_bio_span(self, input_data):
        tokenized_texts, lus, senses, args, gold_spans, span_bio = [], [], [], [], [], []
        orig_tok_to_maps = []
        for i in range(len(input_data)):
            data = input_data[i]
            text = ' '.join(data[0])
            orig_tokens, bert_tokens, orig_to_tok_map = self.bert_tokenizer(text)

            orig_tok_to_maps.append(orig_to_tok_map)
            tokenized_texts.append(bert_tokens)

            ori_lus = data[1]
            lu_sequence = []
            for i in range(len(bert_tokens)):
                if i in orig_to_tok_map:
                    idx = orig_to_tok_map.index(i)
                    l = ori_lus[idx]
                    lu_sequence.append(l)
                else:
                    lu_sequence.append('_')
            lus.append(lu_sequence)

            if self.mode == 'train':
                ori_senses, ori_args = data[2], data[3]
                sense_sequence, arg_sequence, span = [], [], []
                ar = 'X'
                for i in range(len(bert_tokens)):
                    if i in orig_to_tok_map:
                        idx = orig_to_tok_map.index(i)
                        fr = ori_senses[idx]
                        sense_sequence.append(fr)
                        ar = ori_args[idx]
                        arg_sequence.append(ar)
                        span.append(ar[0])
                    else:
                        sense_sequence.append('_')
                        if ar[0] == 'B':
                            arg_sequence.append('I' + ar[1:])
                            span.append('I')
                        else:
                            arg_sequence.append(ar)
                            span.append(ar[0])
                senses.append(sense_sequence)  # [_, _, _, frame_type, _, _, ..]
                args.append(arg_sequence)   # [X O O O,.. B-fe, I-fe, ...]
                span_bio.append(span)   # [X O O O,.. B, I, ...]

                span = []
                st, en, label = -1, -1, -1
                for ii, tag in enumerate(arg_sequence):
                    if tag[0] == 'B' or tag[0] == 'O' or tag[0] == 'X':
                        if st != -1:
                            en = ii - 1
                            span.append((st, en, label))
                            st = -1
                            en = -1
                    if tag[0] == 'B':
                        st = ii
                        label = self.arg2idx[tag[2:]]
                if st != -1 and en == -1:
                    en = len(bert_tokens) - 2
                    span.append((st, en, label))




                gold_spans.append(span)

        input_ids = pad_sequences([self.tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                                  maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

        orig_tok_to_maps = pad_sequences(orig_tok_to_maps, maxlen=MAX_LEN, dtype="long", truncating="post",
                                         padding="post", value=-1)

        if self.mode == 'train':
            if self.srl == 'propbank-dp':
                arg_ids = pad_sequences([[self.arg2idx.get(ar) for ar in arg] for arg in args],
                                        maxlen=MAX_LEN, value=self.arg2idx["X"], padding="post",
                                        dtype="long", truncating="post")
            elif self.srl == 'framenet-argid':
                arg_ids = pad_sequences([[self.bio_argument2idx.get(ar) for ar in arg] for arg in args],
                                        maxlen=MAX_LEN, value=self.bio_argument2idx["X"], padding="post",
                                        dtype="long", truncating="post")
            else:
                arg_ids = pad_sequences([[self.bio_arg2idx.get(ar) for ar in arg] for arg in args],
                                        maxlen=MAX_LEN, value=self.bio_arg2idx["X"], padding="post",
                                        dtype="long", truncating="post")
                span_ids = pad_sequences([[self.span2idx.get(ar) for ar in arg] for arg in span_bio],
                                        maxlen=MAX_LEN, value=self.span2idx["X"], padding="post",
                                        dtype="long", truncating="post")

        lu_seq, sense_seq = [], []
        for sent_idx in range(len(lus)):
            lu_items = lus[sent_idx]
            lu = []
            for idx in range(len(lu_items)):
                if lu_items[idx] != '_':
                    if len(lu) == 0:
                        if self.mode != 'train' and self.masking == False:
                            lu.append(1)
                        else:
                            lu.append(self.lu2idx[lu_items[idx]])  # LU에 대한 index를 추가.
            lu_seq.append(lu)

            if self.mode == 'train':
                sense_items, arg_items = senses[sent_idx], args[sent_idx]
                sense = []
                for idx in range(len(sense_items)):
                    if sense_items[idx] != '_':
                        if len(sense) == 0:
                            sense.append(self.sense2idx[sense_items[idx]])  # frame type에 대한 index를 추가.
                sense_seq.append(sense)

        attention_masks = [[float(i > 0) for i in ii] for ii in input_ids]
        token_type_ids = [[0 if idx > 0 else 1 for idx in input_id] for input_id in input_ids]

        data_inputs = torch.tensor(input_ids)
        data_orig_tok_to_maps = torch.tensor(orig_tok_to_maps)
        data_lus = torch.tensor(lu_seq)
        data_token_type_ids = torch.tensor(token_type_ids)
        data_masks = torch.tensor(attention_masks)

        if self.mode == 'train':
            data_senses = torch.tensor(sense_seq)
            data_args = torch.tensor(arg_ids)  # [X O O O,.. B-fe, I-fe, ...]에 대한 index
            data_span_ids = torch.tensor(span_ids)  # [X O O O,.. B, I, ...]에 대한 index
            span_pad = [len(x) for x in gold_spans]
            gold_spans = pad_sequences(gold_spans,
                                       maxlen=20, value=(-1, -1, -1), padding="post",
                                       dtype="long", truncating="post")   # start, end, fe index

            gold_spans = torch.tensor(gold_spans)
            span_pad = torch.tensor(span_pad)

            bert_inputs = TensorDataset(data_inputs, data_orig_tok_to_maps, data_lus, data_senses, data_args,
                                        data_token_type_ids, data_masks, gold_spans, span_pad, data_span_ids)

        else:
            bert_inputs = TensorDataset(data_inputs, data_orig_tok_to_maps, data_lus, data_token_type_ids, data_masks)
        return bert_inputs

    def convert_to_bert_input_transfer_written(self, input_data):
        tokenized_texts, lus, senses, args, gold_spans, lu_spans, sense_indices = [], [], [], [], [], [], []
        orig_tok_to_maps = []
        errors = 0
        for i in range(len(input_data)):
            data = input_data[i]
            text = ' '.join(data[0])
            orig_tokens, bert_tokens, orig_to_tok_map = self.bert_tokenizer(text)

            ori_lus = data[1]
            lu_sequence = []

            lu_ori_idx = [ii for ii, x in enumerate(ori_lus) if x != '_']
            try:
                lu_span = (orig_to_tok_map[lu_ori_idx[0]], orig_to_tok_map[lu_ori_idx[-1]+1]-1)
            except:
                lu_span = (orig_to_tok_map[lu_ori_idx[0]], len(bert_tokens)-2)


            for i in range(len(bert_tokens)):
                if i in orig_to_tok_map:
                    idx = orig_to_tok_map.index(i)
                    l = ori_lus[idx]
                    lu_sequence.append(l)
                else:
                    lu_sequence.append('_')

            ori_senses, ori_args = data[2], data[3]
            sense_sequence, arg_sequence, span = [], [], []
            ar = 'X'
            for i in range(len(bert_tokens)):
                if i in orig_to_tok_map:
                    idx = orig_to_tok_map.index(i)
                    fr = ori_senses[idx]
                    sense_sequence.append(fr)
                    ar = ori_args[idx]
                    arg_sequence.append(ar)
                    span.append(ar[0])
                else:
                    sense_sequence.append('_')
                    if ar[0] == 'B':
                        arg_sequence.append('I' + ar[1:])
                        span.append('I')
                    else:
                        arg_sequence.append(ar)
                        span.append(ar[0])


            span = []
            label_list = []
            st, en, label = -1, -1, -1
            for ii, tag in enumerate(arg_sequence):
                if tag[0] == 'B' or tag[0] == 'O' or tag[0] == 'X':
                    if st != -1:
                        en = ii - 1
                        span.append((st, en, label))
                        label_list.append(label)

                        st = -1
                        en = -1
                if tag[0] == 'B':
                    st = ii
                    label = self.arg2idx[tag[2:]]
            if st != -1 and en == -1:
                en = len(bert_tokens) - 2
                span.append((st, en, label))
                label_list.append(label)



            if len(span) == 0:
                continue

            sense_items = sense_sequence
            sense = set()
            for idx in range(len(sense_items)):
                if sense_items[idx] != '_':
                    if len(sense) == 0:
                        sense.add(self.sense2idx[sense_items[idx]])  # frame type에 대한 index를 추가.
            sense = list(sense)
            is_error = False
            for ll in label_list:
                if ll not in self.frargmap[str(sense[0])]:
                    is_error = True
            if is_error:
                errors += 1
                continue

            sense_indices.append(sense)
            gold_spans.append(span)
            args.append(arg_sequence)  # [X O O O,.. B-fe, I-fe, ...]
            orig_tok_to_maps.append(orig_to_tok_map)
            tokenized_texts.append(bert_tokens)
            senses.append(sense_sequence)  # [_, _, _, frame_type, _, _, ..]
            lu_spans.append(lu_span)
            lus.append(lu_sequence)

        a = 1

        if len(tokenized_texts) == 0:
            return None

        input_ids = pad_sequences([self.tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                                  maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

        orig_tok_to_maps = pad_sequences(orig_tok_to_maps, maxlen=MAX_LEN, dtype="long", truncating="post",
                                         padding="post", value=-1)

        if self.srl == 'propbank-dp':
            arg_ids = pad_sequences([[self.arg2idx.get(ar) for ar in arg] for arg in args],
                                    maxlen=MAX_LEN, value=self.arg2idx["X"], padding="post",
                                    dtype="long", truncating="post")
        elif self.srl == 'framenet-argid':
            arg_ids = pad_sequences([[self.bio_argument2idx.get(ar) for ar in arg] for arg in args],
                                    maxlen=MAX_LEN, value=self.bio_argument2idx["X"], padding="post",
                                    dtype="long", truncating="post")
        else:
            arg_ids = pad_sequences([[self.bio_arg2idx.get(ar) for ar in arg] for arg in args],
                                    maxlen=MAX_LEN, value=self.bio_arg2idx["X"], padding="post",
                                    dtype="long", truncating="post")

        lu_seq, sense_seq = [], []
        for sent_idx in range(len(lus)):
            lu_items = lus[sent_idx]
            lu = []
            for idx in range(len(lu_items)):
                if lu_items[idx] != '_':
                    if len(lu) == 0:
                        if self.mode != 'train' and self.masking == False:
                            lu.append(1)
                        else:
                            lu.append(self.lu2idx[lu_items[idx]])  # LU에 대한 index를 추가.
            lu_seq.append(lu)


        attention_masks = [[float(i > 0) for i in ii] for ii in input_ids]
        token_type_ids = [[0 if idx > 0 else 1 for idx in input_id] for input_id in input_ids]

        data_inputs = torch.tensor(input_ids)
        data_orig_tok_to_maps = torch.tensor(orig_tok_to_maps)
        data_lus = torch.tensor(lu_seq)
        data_token_type_ids = torch.tensor(token_type_ids)
        data_masks = torch.tensor(attention_masks)
        lu_spans = torch.tensor(lu_spans)


        data_senses = torch.tensor(sense_indices)
        data_args = torch.tensor(arg_ids)  # [X O O O,.. B-fe, I-fe, ...]에 대한 index
        span_pad = [len(x) for x in gold_spans]
        gold_spans = pad_sequences(gold_spans,
                                   maxlen=20, value=(-1, -1, -1), padding="post",
                                   dtype="long", truncating="post")   # start, end, fe index

        gold_spans = torch.tensor(gold_spans)
        span_pad = torch.tensor(span_pad)

        bert_inputs = TensorDataset(data_inputs, data_orig_tok_to_maps, data_lus, data_senses, data_args,
                                    data_token_type_ids, data_masks, gold_spans, span_pad, lu_spans)

        print("{}: frame-fe pair doesn't exist in framenet_info.json".format(errors))
        print("rest: {}".format(data_inputs.shape))
        return bert_inputs

    def convert_to_bert_input_transfer_spoken(self, input_data):
        tokenized_texts, orig_tok_to_maps, gold_spans, lu_spans, sense_indices, lu_speakers, fe_speakers = [], [], [], [], [], [], []
        """
        lu_spans : list of (utter id, start, end)  # [#frame, 3]
        lu_speakers : list of speaker of target utterance  # [#frame]
        sense_indices : list of frame type id  [#frame]
        gold_spans : list of fe span (fe가 발생한 utter_id, start, end, fe_id)  # [#frame, 4]
        tokenized_texts : list of pair token [#frame, #fe, 2, #token] 
            - pair token : (target utterance tokens, fe utterance tokens)
        orig_tok_to_maps : tokenized_texts에 대응하는 origin token, bert token map [#frame, #fe, 2, ?] 
        fe_speakers : list of fe speaker  [#frame, #fe]
        """
        errors = 0
        for i in range(len(input_data)):
            data = input_data[i]
            scene_tokens = []
            scene_maps = []
            scene_speaker = [self.speaker2idx[x] for x in data['speakers']]
            utter_speaker = []

            sense = [self.sense2idx[data['frame']['frame']]]
            for utter in data['utterances']:
                orig_tokens, bert_tokens, orig_to_tok_map = self.bert_tokenizer(utter['ko_utter'])
                utter_speaker.append(self.speaker2idx[utter['speaker']])
                scene_tokens.append(bert_tokens)
                scene_maps.append(orig_to_tok_map)

            lu_utter_id = int(data['frame']['utter_id'].split('#')[-1])
            lu_tokens = scene_tokens[lu_utter_id]
            lu_map = scene_maps[lu_utter_id]
            lu_speaker = utter_speaker[lu_utter_id]
            lu_ori_idx = data['frame']['target_index']
            try:
                lu_span = (lu_utter_id, lu_map[lu_ori_idx], lu_map[lu_ori_idx+1]-1)
            except:
                lu_span = (lu_utter_id, lu_map[lu_ori_idx], len(scene_tokens[lu_utter_id])-2)

            fes = data['frame']['elements']
            for fe_name, fe_info in fes.items():
                fe_id = self.arg2idx[fe_name]
                fe_idx = fe_info['idx']

                if fe_info['type'] == 'utterance':
                    fe_utter_id = int(fe_info['utter_id'].split('#')[-1])
                    fe_map = scene_maps[fe_utter_id]
                    try:
                        fe_span = (fe_utter_id, fe_map[fe_idx[0]], fe_map[fe_idx[-1] + 1] - 1, fe_id)
                    except:
                        fe_span = (fe_utter_id, fe_map[fe_idx[0]], len(scene_tokens[fe_utter_id]) - 2, fe_id)

                else:  # speaker
                    candidates = []
                    for s_i, sp in enumerate(utter_speaker):
                        if self.speaker2idx[fe_info['text']] == sp:
                            candidates.append(s_i)
                    # candidates와 lu_utter_id와 가장 가까운 애를 찾아야함.
                    min_diff = 10000
                    nearest_utter_id = -1
                    for c in candidates:
                        diff = abs(c-lu_utter_id)
                        if min_diff > diff:
                            min_diff = diff
                            nearest_utter_id = c
                    fe_utter_id = nearest_utter_id
                    fe_map = scene_maps[fe_utter_id]
                    fe_span = (nearest_utter_id, 0, 0, fe_id)

                pair_token = [lu_tokens, scene_tokens[fe_utter_id]]
                pair_map = [lu_map, fe_map]
                fe_speaker = utter_speaker[fe_utter_id]

                lu_speakers.append(lu_speaker)
                sense_indices.append(sense)
                gold_spans.append(fe_span)
                orig_tok_to_maps.append(pair_map)
                tokenized_texts.append(pair_token)
                lu_spans.append(lu_span)
                fe_speakers.append(fe_speaker)

        if len(tokenized_texts) == 0:
            return None

        tokenized_ids = []
        # utter_len = []
        for scene_txt in tokenized_texts:
            # utter_len.append(len(scene_txt))
            scene_ids = pad_sequences([self.tokenizer.convert_tokens_to_ids(txt) for txt in scene_txt], maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
            tokenized_ids.append(scene_ids)
        input_ids = tokenized_ids  # pad_sequences(tokenized_ids, maxlen=90, dtype="long", truncating="post", padding="post")

        tensor_maps = []
        for scene_map in orig_tok_to_maps:
            map = pad_sequences(scene_map, maxlen=MAX_LEN, dtype="long", truncating="post",
                                         padding="post", value=-1)
            tensor_maps.append(map)
        orig_tok_to_maps = tensor_maps  # pad_sequences(tensor_maps, maxlen=90, dtype="long", truncating="post", padding="post", value=-1)


        attention_masks = [[[float(i > 0) for i in ii] for ii in input_id] for input_id in input_ids]
        token_type_ids = [[[0 if idx > 0 else 1 for idx in input_id] for input_id in input] for input in input_ids]

        # gold_spans = pad_sequences(gold_spans,
        #                            maxlen=20, value=(-1, -1, -1, -1), padding="post",
        #                            dtype="long", truncating="post")  # start, end, fe index


        data_inputs = torch.tensor(input_ids)
        data_orig_tok_to_maps = torch.tensor(orig_tok_to_maps)
        data_token_type_ids = torch.tensor(token_type_ids)
        data_masks = torch.tensor(attention_masks)
        lu_spans = torch.tensor(lu_spans)
        data_senses = torch.tensor(sense_indices)
        gold_spans = torch.tensor(gold_spans)
        lu_speakers = torch.tensor(lu_speakers)
        fe_speakers = torch.tensor(fe_speakers)

        bert_inputs = TensorDataset(data_inputs, data_orig_tok_to_maps, data_token_type_ids, data_masks, lu_spans,
                                    data_senses, gold_spans, lu_speakers, fe_speakers)

        # print("{}: frame-fe pair doesn't exist in framenet_info.json".format(errors))
        # print("rest: {}".format(data_inputs.shape))
        return bert_inputs

    def convert_to_bert_input_JointShallowSemanticParsing(self, input_data):
        tokenized_texts, lus, senses, args = [], [], [], []
        orig_tok_to_maps = []
        for i in range(len(input_data)):
            data = input_data[i]
            text = ' '.join(data[0])
            orig_tokens, bert_tokens, orig_to_tok_map = self.bert_tokenizer(text)

            orig_tok_to_maps.append(orig_to_tok_map)
            tokenized_texts.append(bert_tokens)

            ori_lus = data[1]
            lu_sequence = []
            for i in range(len(bert_tokens)):
                if i in orig_to_tok_map:
                    idx = orig_to_tok_map.index(i)
                    l = ori_lus[idx]
                    lu_sequence.append(l)
                else:
                    lu_sequence.append('_')
            lus.append(lu_sequence)

            if self.mode == 'train':
                ori_senses, ori_args = data[2], data[3]
                sense_sequence, arg_sequence = [], []
                for i in range(len(bert_tokens)):
                    if i in orig_to_tok_map:
                        idx = orig_to_tok_map.index(i)
                        fr = ori_senses[idx]
                        sense_sequence.append(fr)
                        ar = ori_args[idx]
                        arg_sequence.append(ar)
                    else:
                        sense_sequence.append('_')
                        arg_sequence.append('X')
                senses.append(sense_sequence)  # [_, _, _, frame_type, _, _, ..]
                args.append(arg_sequence)   # [X O X X,.. B-fe, I-fe, ...]

        input_ids = pad_sequences([self.tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                                  maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

        orig_tok_to_maps = pad_sequences(orig_tok_to_maps, maxlen=MAX_LEN, dtype="long", truncating="post",
                                         padding="post", value=-1)

        if self.mode == 'train':
            if self.srl == 'propbank-dp':
                arg_ids = pad_sequences([[self.arg2idx.get(ar) for ar in arg] for arg in args],
                                        maxlen=MAX_LEN, value=self.arg2idx["X"], padding="post",
                                        dtype="long", truncating="post")
            elif self.srl == 'framenet-argid':
                arg_ids = pad_sequences([[self.bio_argument2idx.get(ar) for ar in arg] for arg in args],
                                        maxlen=MAX_LEN, value=self.bio_argument2idx["X"], padding="post",
                                        dtype="long", truncating="post")
            else:
                arg_ids = pad_sequences([[self.bio_arg2idx.get(ar) for ar in arg] for arg in args],
                                        maxlen=MAX_LEN, value=self.bio_arg2idx["X"], padding="post",
                                        dtype="long", truncating="post")

        lu_seq, sense_seq = [], []
        for sent_idx in range(len(lus)):
            lu_items = lus[sent_idx]
            lu = []
            for idx in range(len(lu_items)):
                if lu_items[idx] != '_':
                    if len(lu) == 0:
                        if self.mode != 'train' and self.masking == False:
                            lu.append(1)
                        else:
                            lu.append(self.lu2idx[lu_items[idx]])

            lu_seq.append(lu)

            if self.mode == 'train':
                sense_items, arg_items = senses[sent_idx], args[sent_idx]
                sense = []
                for idx in range(len(sense_items)):
                    if sense_items[idx] != '_':
                        if len(sense) == 0:
                            sense.append(self.sense2idx[sense_items[idx]])
                sense_seq.append(sense)

        attention_masks = [[float(i > 0) for i in ii] for ii in input_ids]
        token_type_ids = [[0 if idx > 0 else 1 for idx in input_id] for input_id in input_ids]

        data_inputs = torch.tensor(input_ids)
        data_orig_tok_to_maps = torch.tensor(orig_tok_to_maps)
        data_lus = torch.tensor(lu_seq)
        data_token_type_ids = torch.tensor(token_type_ids)
        data_masks = torch.tensor(attention_masks)

        if self.mode == 'train':
            data_senses = torch.tensor(sense_seq)
            data_args = torch.tensor(arg_ids)
            bert_inputs = TensorDataset(data_inputs, data_orig_tok_to_maps, data_lus, data_senses, data_args,
                                        data_token_type_ids, data_masks)
        else:
            bert_inputs = TensorDataset(data_inputs, data_orig_tok_to_maps, data_lus, data_token_type_ids, data_masks)
        return bert_inputs

    def convert_to_bert_input_e2e(self, input_data):
        tokenized_texts, lus, senses, args, gold_spans, tgt_pos = [],[],[],[],[],[]
        orig_tok_to_maps = []
        for i in range(len(input_data)):    
            data = input_data[i]
            text = ' '.join(data[0])
            orig_tokens, bert_tokens, orig_to_tok_map = self.bert_tokenizer(text)
            # orig_to_tok_map : origin token에 대응하는 bert start token idx
            tgt = []
            for ii, tok in enumerate(bert_tokens):
                if tok in ['[CLS]', '[SEP]','<tgt>', '</tgt>']:
                    tgt.append(ii)
            tgt_pos.append(tgt)

            orig_tok_to_maps.append(orig_to_tok_map)
            tokenized_texts.append(bert_tokens)

            ori_lus = data[1]    
            lu_sequence = []
            for j in range(len(bert_tokens)):
                if j in orig_to_tok_map:
                    idx = orig_to_tok_map.index(j)
                    l = ori_lus[idx]
                    lu_sequence.append(l)
                else:
                    lu_sequence.append('_')
            lus.append(lu_sequence)  # bert token에 맞춘 lu(ex-이익.n), 단 매칭되는 ori token의 첫번째 bert token에 달린다.

            ori_senses = data[2]
            sense_sequence = []
            for j in range(len(bert_tokens)):
                if j in orig_to_tok_map:
                    idx = orig_to_tok_map.index(j)
                    fr = ori_senses[idx]
                    sense_sequence.append(fr)
                else:
                    sense_sequence.append('_')
            senses.append(sense_sequence)

            if self.mode == 'train':
                ori_args = data[3]
                arg_sequence = []
                for j in range(len(bert_tokens)):
                    if j in orig_to_tok_map:
                        idx = orig_to_tok_map.index(j)
                        ar = ori_args[idx]
                        arg_sequence.append(ar)
                    else:
                        arg_sequence.append('X')
                args.append(arg_sequence)

                span = []
                st, en, label = -1, -1, -1
                for ii, tag in enumerate(arg_sequence):
                    if tag[0] == 'B' or tag[0] == 'O':
                        if st != -1:
                            en = ii - 1
                            span.append((st,en, label))
                            st = -1
                            en = -1
                    if tag[0] == 'B':
                        st = ii
                        label = self.arg2idx[tag[2:]]
                if st != -1 and en == -1:
                    en = len(bert_tokens) - 2
                    span.append((st, en, label))
                gold_spans.append(span)

        input_ids = pad_sequences([self.tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                              maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")  # bert token to idx & append padding
        
        orig_tok_to_maps = pad_sequences(orig_tok_to_maps, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post", value=-1)
        
        if self.mode =='train':
            if self.srl == 'propbank-dp':
                arg_ids = pad_sequences([[self.arg2idx.get(ar) for ar in arg] for arg in args],
                                        maxlen=MAX_LEN, value=self.arg2idx["X"], padding="post",
                                        dtype="long", truncating="post")
            elif self.srl == 'framenet-argid':
                arg_ids = pad_sequences([[self.bio_argument2idx.get(ar) for ar in arg] for arg in args],
                                        maxlen=MAX_LEN, value=self.bio_argument2idx["X"], padding="post",
                                        dtype="long", truncating="post")
            else:
                arg_ids = pad_sequences([[self.bio_arg2idx.get(ar) for ar in arg] for arg in args],
                                        maxlen=MAX_LEN, value=self.bio_arg2idx["X"], padding="post",
                                        dtype="long", truncating="post")

        lu_seq, sense_seq = [],[]
        for sent_idx in range(len(lus)):
            lu_items = lus[sent_idx]
            lu = []
            for idx in range(len(lu_items)):
                if lu_items[idx] != '_':
                    if len(lu) == 0:
                        if self.mode != 'train' and self.masking == False:
                            lu.append(1)
                        else:
                            lu.append(self.lu2idx[lu_items[idx]])
                            
            lu_seq.append(lu)

            sense_items = senses[sent_idx]
            sense = []
            for idx in range(len(sense_items)):
                if sense_items[idx] != '_':
                    if len(sense) == 0:
                        sense.append(self.sense2idx[sense_items[idx]])
            sense_seq.append(sense)

        attention_masks = [[float(i>0) for i in ii] for ii in input_ids]
        token_type_ids = [[0 if idx > 0 else 1 for idx in input_id]for input_id in input_ids] # 일반 토큰은 0, pad는 1
        
        data_inputs = torch.tensor(input_ids)
        data_orig_tok_to_maps = torch.tensor(orig_tok_to_maps)
        data_lus = torch.tensor(lu_seq)
        data_token_type_ids = torch.tensor(token_type_ids)
        data_masks = torch.tensor(attention_masks)
        data_senses = torch.tensor(sense_seq)
        tgt_pos = torch.tensor(tgt_pos)

        if self.mode == 'train':
            data_args = torch.tensor(arg_ids)

            span_pad = [len(x) for x in gold_spans]

            gold_spans = pad_sequences(gold_spans,
                                    maxlen=20, value=(-1,-1,-1), padding="post",
                                    dtype="long", truncating="post")


            gold_spans = torch.tensor(gold_spans)
            span_pad = torch.tensor(span_pad)
            bert_inputs = TensorDataset(data_inputs, data_orig_tok_to_maps, data_lus, data_senses, data_args, data_token_type_ids, data_masks, gold_spans, span_pad, tgt_pos)
        else:
            bert_inputs = TensorDataset(data_inputs, data_orig_tok_to_maps, data_lus, data_senses, data_token_type_ids, data_masks, tgt_pos)
        return bert_inputs
    
    def convert_to_bert_input_label_definition(self, input_data, label2idx):
        tokenized_texts, orig_tok_to_maps = [],[]
        labels = []
        for label in input_data:    
            text = input_data[label]
            orig_tokens, bert_tokens, orig_to_tok_map = self.bert_tokenizer(text)
            orig_tok_to_maps.append(orig_to_tok_map)
            tokenized_texts.append(bert_tokens)
            labels.append(label2idx[label])

        input_ids = pad_sequences([self.tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
        
        orig_tok_to_maps = pad_sequences(orig_tok_to_maps, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post", value=-1)

        attention_masks = [[float(i>0) for i in ii] for ii in input_ids]
        token_type_ids = [[0 if idx > 0 else 1 for idx in input_id]for input_id in input_ids]
        
        data_inputs = torch.tensor(input_ids)
        data_orig_tok_to_maps = torch.tensor(orig_tok_to_maps)
        data_masks = torch.tensor(attention_masks)
        data_token_type_ids = torch.tensor(token_type_ids)
        
        bert_inputs = TensorDataset(data_inputs, data_orig_tok_to_maps, data_token_type_ids, data_masks)
        return bert_inputs, tuple(labels)

    
def get_masks(datas, mapdata, num_label=2, masking=True):  # datas : lus [bsz, 1]
    masks = []
    with torch.no_grad():
        if masking == True:
            for idx in datas:
                torch.cuda.set_device(0)
                indx = str(idx).split('[')[-1].split(']')[0] # current lu idx
                mask = torch.zeros(num_label)
                candis = mapdata[indx]  # 해당 lu가 가질 수 있는 frame type indices.
                for candi_idx in candis:
                    mask[candi_idx] = 1
                masks.append(mask)
        else:
            for idx in datas:
                mask = torch.ones(num_label)
                masks.append(mask)
    masks = torch.stack(masks)
    return masks

def masking_logit(logit, mask):
    with torch.no_grad():
        if type(logit) is np.ndarray:
            pass
        else:
            logit = logit.cpu().numpy()
        mask = mask.cpu().numpy()
        masking = np.multiply(logit, mask)
    masking[masking==0] = np.NINF
    masking = torch.tensor(masking)
    return masking

def probs2idx(probs):
    return None

def logit2pos(start_logit, end_logit):
    sm = nn.Softmax()
    st_logits = sm(start_logit).view(1, -1)
    en_logits = sm(end_logit).view(1, -1)
    score, st = st_logits.max(1)
    score, en = en_logits.max(1)
    return int(st), int(en)


def logit2span(logit):
    """ 'X':0, 'O':1, 'B':2, 'I':3 """
    pred_bio = torch.argmax(logit, dim=1)  # [tok_len]
    in_span = False
    st, en = -1, -1
    span = []
    for i, x in enumerate(pred_bio):
        if x == 2:  # B
            if in_span == True:  # 직전까지를 span으로 결정.
                en = i - 1
                span.append((st, en))
            # 새로운 span 생성.
            in_span = True
            st = i
        if x == 0 or x == 1:  # X or O
            if in_span:  # 직전까지를 span으로 결정.
                en = i - 1
                in_span = False
                span.append((st, en))
        elif (i == len(logit) - 1) and in_span:
            en = i
            span.append((st, en))
    return span

def logit2label(masked_logit):
    sm = nn.Softmax()
    pred_logits = sm(masked_logit).view(1,-1)
    score, label = pred_logits.max(1)
    score = float(score)
    
    return label, score

def logit2candis(masked_logit, candis=1, idx2label=False):
    sm = nn.Softmax()
    pred_logits = sm(masked_logit).view(1,-1)
    
    logit_len = pred_logits.size()[-1]
    if candis >= logit_len:
        candis = logit_len
    
    scores, labels = pred_logits.topk(candis)
    
    candis = []
    for i in range(len(scores[0])):
        score = round(float(scores[0][i]),4)
        idx = int(labels[0][i])
        if idx2label:
            label = idx2label[idx]
        else:
            label = idx
        
        candi = (label, score)
        candis.append(candi)
    
    return candis

def get_tgt_idx(bert_tokens, tgt=False):
    tgt_idx = []
    try:
        if tgt == False:
            for i in range(len(bert_tokens)):
                if bert_tokens[i] == '<':
                    if bert_tokens[i+1] == 't' and bert_tokens[i+2] == '##gt' and bert_tokens[i+3] == '>':
                        tgt_idx.append(i)
                        tgt_idx.append(i+1)
                        tgt_idx.append(i+2)
                        tgt_idx.append(i+3)
                    elif bert_tokens[i+1] == '/' and bert_tokens[i+2] == 't' and bert_tokens[i+3] == '##gt' and bert_tokens[i+4] == '>':
                        tgt_idx.append(i)
                        tgt_idx.append(i+1)
                        tgt_idx.append(i+2)
                        tgt_idx.append(i+3)
                        tgt_idx.append(i+4)
        else:
            tgt_token_list = ['<tgt>', '</tgt>']
            for i in range(len(bert_tokens)):
                if bert_tokens[i] in tgt_token_list:
                    tgt_idx.append(i)
    except KeyboardInterrupt:
        raise
    except:
        pass
    
    return tgt_idx


# 파일 이름으로 json 로드(utf-8만 해당)
def jsonload(fname, encoding="utf-8"):
    with open(fname, encoding=encoding) as f:
        j = json.load(f)
    return j

# json 개체를 파일이름으로 깔끔하게 저장
def jsondump(j, fname):
    with open(fname, "w", encoding="UTF8") as f:
        json.dump(j, f, ensure_ascii=False, indent="\t")