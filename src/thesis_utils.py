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

torch.manual_seed(0)
MAX_LEN = 256

import os
try:
    dir_path = os.path.dirname(os.path.abspath( __file__ ))
except:
    dir_path = '.'
    
dir_path = dir_path+'/..'


def get_enlu(self, token, pos):  # only_lu=True
    result = False

    p = False
    if pos == 'NN' or pos == 'NNS':
        p = 'n'
    elif pos.startswith('V'):
        p = 'v'
    elif pos.startswith('J'):
        p = 'a'
    else:
        p = False

    # lemmatize

    if p:
        lemma = self.lemmatizer.lemmatize(token, p)
        if lemma:
            #                 if lemma != 'be':
            if self.masking == True:
                for lu in self.targetdic:
                    lu_pos = lu.split('.')[-1]
                    if self.only_lu == True:
                        if p == lu_pos:
                            candi = self.targetdic[lu]
                            if lemma in candi:
                                result = lu
                            else:
                                pass
                    else:
                        candi = self.targetdic[lu]
                        if lemma in candi:
                            result = lu
                        else:
                            pass
            else:
                result = lemma + '.' + pos

    return result

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
            self.tokenizer.additional_special_tokens = ['<tgt>', '</tgt>', '<sp>']
        elif 'large' in pretrained:
            vocab_file_path = dir_path+'/data/bert-large-cased-dict-add-tgt'
            self.tokenizer = BertTokenizer(vocab_file_path, do_lower_case=False, max_len=256)
            self.tokenizer.additional_special_tokens = ['<tgt>', '</tgt>', '<sp>']
        else:
            vocab_file_path = dir_path+'/data/bert-multilingual-cased-dict-add-tgt'
            self.tokenizer = BertTokenizer(vocab_file_path, do_lower_case=False, max_len=256)
            self.tokenizer.additional_special_tokens = ['<tgt>', '</tgt>', '<sp>']

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
            if language == "ko":
                self.lufrmap["5489"] = [x for x in range(len(self.sense2idx))]  # LU dict에 없는 애들
            else:
                self.lufrmap["10466"] = [x for x in range(len(self.sense2idx))]  # LU dict에 없는 애들

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

        if language == 'ko':
            with open(frargmap_path2,'r') as f:
                self.frargmap = json.load(f)
            
        if info:
            print('used dictionary:')
            print('\t', data_path+'lu2idx.json')
            print('\t', data_path+'lufrmap.json')
            print('\t', frargmap_path)
            
        self.idx2sense = dict(zip(self.sense2idx.values(),self.sense2idx.keys()))
        self.idx2arg = dict(zip(self.arg2idx.values(),self.arg2idx.keys()))

        self.speaker2idx = {'seohee': 0, 'yijoon': 1, 'jeongsuk': 2, 'heeran': 3, 'haeyoung2': 4, 'jiya': 5, 'deogi': 6, 'soontack': 7, 'anna': 8, 'chairman': 9, 'gitae': 10, 'kyungsu': 11, 'hun':12, 'sukyung': 13, 'dokyung': 14, 'sangseok': 15, 'taejin': 16, 'sungjin': 17, 'jinsang': 18, 'haeyoung1': 19, 'none': 20}
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
    
    def bert_tokenizer(self, text, use_sp_token):
        orig_tokens = text.split(' ')
        bert_tokens = []
        orig_to_tok_map = []
        bert_tokens.append("[CLS]")
        for orig_token in orig_tokens:
            orig_to_tok_map.append(len(bert_tokens))
            bert_tokens.extend(self.tokenizer.tokenize(orig_token))
        if use_sp_token == 'sp':
            orig_to_tok_map.append(len(bert_tokens))  # sp 아니면 지워야함
            bert_tokens.append("<sp>")   # sp 아니면 지워야함.
        bert_tokens.append("[SEP]")

        return orig_tokens, bert_tokens, orig_to_tok_map

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
                args.append(arg_sequence)  # [X O X X,.. B-fe, I-fe, ...]

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


    def written_converter(self, input_data):
        tokenized_texts, orig_tok_to_maps, gold_args, targets, senses, speakers, special_tokens, bios, lus = [], [], [], [], [], [], [], [], []
        errors = 0
        for i in range(len(input_data)):
            data = input_data[i]
            text = ' '.join(data[0])
            scene_tokens = []
            scene_maps = []
            scene_special_tokens = []
            utter_speaker = []
            fe_spans = []

            orig_tokens, bert_tokens, orig_to_tok_map = self.bert_tokenizer(text)
            utter_speaker.append(-1)
            scene_tokens.append(bert_tokens)
            scene_maps.append(orig_to_tok_map)
            scene_special_tokens.append([0, len(bert_tokens) - 1])

            ori_lus = data[1]
            lu_sequence = []
            lu_ori_idx = [ii for ii, x in enumerate(ori_lus) if x != '_']
            try:
                lu_span = (0, orig_to_tok_map[lu_ori_idx[0]], orig_to_tok_map[lu_ori_idx[-1] + 1] - 1)
            except:
                lu_span = (0, orig_to_tok_map[lu_ori_idx[0]], len(bert_tokens) - 2)

            for i in range(len(bert_tokens)):
                if i in orig_to_tok_map:
                    idx = orig_to_tok_map.index(i)
                    l = ori_lus[idx]
                    lu_sequence.append(l)
                else:
                    lu_sequence.append('_')

            ori_senses, ori_args = data[2], data[3]
            sense_sequence, arg_sequence, span = [], [], []
            ar = 'O'
            for i in range(len(bert_tokens)):
                if i in orig_to_tok_map:
                    idx = orig_to_tok_map.index(i)
                    fr = ori_senses[idx]
                    sense_sequence.append(fr)
                    ar = ori_args[idx]
                    arg_sequence.append(ar)
                else:
                    sense_sequence.append('_')
                    if ar[0] == 'B':
                        arg_sequence.append('I' + ar[1:])
                    else:
                        arg_sequence.append(ar)




            bio = torch.ones(1, MAX_LEN).long()
            bio[0][orig_to_tok_map] = 0
            span = []
            label_list = []
            st, en, label = -1, -1, -1
            for ii, tag in enumerate(arg_sequence):
                if tag[0] == 'B' or tag[0] == 'O' or tag[0] == 'X':
                    if st != -1:
                        en = ii - 1
                        span.append((0, st, en, label))
                        label_list.append(label)

                        st = -1
                        en = -1
                if tag[0] == 'B':
                    st = ii
                    label = self.arg2idx[tag[2:]]
            if st != -1 and en == -1:
                en = len(bert_tokens) - 2
                span.append((0, st, en, label))
                label_list.append(label)

            if len(span) == 0:
                continue



            for uid, st, en, label in span:  # idx2arg, self.bio_arg2idx, idx2bio_arg
                label_str = self.idx2arg[label]
                bio[uid][st] = self.bio_arg2idx["B-{}".format(label_str)]
                bio[uid][st+1:en+1] = self.bio_arg2idx["I-{}".format(label_str)]

            for uid in range(1):
                cur_bio = bio[uid]
                cur_map = scene_maps[uid]
                for ii in range(256):
                    if ii not in cur_map:
                        cur_bio[ii] = 1    # [X O X X,.. B-fe, I-fe, ...]

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

            text_bio = [self.idx2bio_arg[x.item()] for x in bio[0]]
            lu_idx = list(set([self.lu2idx[x] for x in ori_lus if x != '_']))

            senses += sense
            lus += lu_idx
            bios.append(bio)
            gold_args.append(span)
            orig_tok_to_maps.append(scene_maps)
            tokenized_texts.append(scene_tokens)
            targets.append(lu_span)
            speakers.append(utter_speaker)
            special_tokens.append(scene_special_tokens)

        if len(tokenized_texts) == 0:
            return None

        tokenized_ids = []
        for scene_txt in tokenized_texts:
            scene_ids = pad_sequences([self.tokenizer.convert_tokens_to_ids(txt) for txt in scene_txt], maxlen=MAX_LEN,
                                      dtype="long", truncating="post", padding="post")
            tokenized_ids.append(scene_ids)
        max_utter = np.max([tokenized.shape[0] for tokenized in tokenized_ids])  # 50
        utter_len = [tokenized.shape[0] for tokenized in tokenized_ids]
        input_ids = pad_sequences(tokenized_ids, maxlen=max_utter, dtype="long", truncating="post", padding="post")

        tensor_maps = []
        for scene_map in orig_tok_to_maps:
            map = pad_sequences(scene_map, maxlen=MAX_LEN, dtype="long", truncating="post",
                                padding="post", value=-1)
            tensor_maps.append(map)
        orig_tok_to_maps = pad_sequences(tensor_maps, maxlen=max_utter, dtype="long", truncating="post", padding="post",
                                         value=-1)
        special_tokens = pad_sequences(special_tokens, maxlen=max_utter, dtype="long", truncating="post",
                                       padding="post", value=-1)

        attention_masks = [[[float(i > 0) for i in ii] for ii in input_id] for input_id in input_ids]
        token_type_ids = [[[0 if idx > 0 else 1 for idx in input_id] for input_id in input] for input in input_ids]

        args_len = [len(arg) for arg in gold_args]
        gold_args = pad_sequences(gold_args,
                                  maxlen=20, value=(-1, -1, -1, -1), padding="post",
                                  dtype="long", truncating="post")  # start, end, fe index

        speaker_len = [len(speaker) for speaker in speakers]
        speakers = pad_sequences(speakers,
                                 maxlen=max_utter, value=-1, padding="post",
                                 dtype="long", truncating="post")  # start, end, fe index

        data_inputs = torch.tensor(input_ids)
        utter_len = torch.tensor(utter_len)
        data_orig_tok_to_maps = torch.tensor(orig_tok_to_maps)
        data_token_type_ids = torch.tensor(token_type_ids)
        data_masks = torch.tensor(attention_masks)
        bios = torch.stack(bios)

        gold_args = torch.tensor(gold_args)
        args_len = torch.tensor(args_len)
        targets = torch.tensor(targets)
        senses = torch.tensor(senses)
        lus = torch.tensor(lus)
        speakers = torch.tensor(speakers)
        speaker_len = torch.tensor(speaker_len)
        special_tokens = torch.tensor(special_tokens)

        bert_inputs = TensorDataset(data_inputs, utter_len, data_orig_tok_to_maps, data_token_type_ids, data_masks,
                                    targets,
                                    senses, gold_args, args_len, speakers, speaker_len, special_tokens, bios, lus)

        return bert_inputs

    def debug_lu(self, input_data):
        not_in_lu_dict = []
        n_instance = 0
        for scene_id, data in input_data.items():  # 172
            for instance in data['frames']:  # 20
                n_instance += 1
                lu = instance['lu']
                if lu not in self.lu2idx.keys():
                    not_in_lu_dict.append(lu)
        return not_in_lu_dict, n_instance



    def data_converter(self, input_data, tgt=False, lang='ko', only_lu_dict=True, use_sp_token=False):
        """ input_data : list of frame instance [#frame] """
        instance_ids = []
        total_num = 0
        long_txt = 0
        not_in_lu_dict = 0
        not_in_lufr_dict = 0
        made = 0
        err_flag = False
        tokenized_texts, orig_tok_to_maps, gold_args, targets, senses, speakers, special_tokens, bios, lus = [], [], [], [], [], [], [], [], []
        """
        tokenized_texts : list of utter token [#instance(frame), #max_utter, #token]
        orig_tok_to_maps : tokenized_texts에 대응하는 origin token, bert token map [#frame, #max_utter, ?]
        gold_args : list of fe span (fe가 발생한 utter_id, start, end, fe_id)  # [#frame, #fe, 4]
        targets : list of target span (utter id, start, end)  # [#frame, 3]
        senses : list of frame type id  [#frame]
        speakers : list of fe speaker  [#frame, #max_utter]  # scene내의 utter마다의 speakers
        """
        if lang[0] == 'k':
            utter_key = 'ko_utter'
        else:
            utter_key = 'plain'

        kk = list(self.lu2idx.keys())
        kk = [x.split('.')[-1] for x in kk]
        kk = list(set(kk))


        for scene_id, data in input_data.items():  # 172
            # if len(data['utterances']) > 10:
            #     continue
            for iid, instance in enumerate(data['frames']):
                total_num += 1
                scene_tokens = []
                scene_maps = []
                scene_special_tokens = []
                utter_speaker = []
                fe_spans = []
                # len(data['utterances'])
                lu_utter_id = int(instance['utter_id'].split('#')[-1])
                lu_ori_idx = instance['target_index']
                min_utter = max(0, lu_utter_id - 6)
                max_utter = min(len(data['utterances']) - 1, lu_utter_id + 4)
                constrant = list(range(min_utter, max_utter + 1))

                sense = self.sense2idx[instance['frame']]
                if tgt:
                    origin_utter = data['utterances'][lu_utter_id][utter_key]
                    origin_utter_word = origin_utter.split()
                    target_txt = [origin_utter_word[lu_ori_idx]]
                    tgted_target = " ".join(['<tgt>'] + target_txt + ['</tgt>'])
                    len_tgted_target = len([origin_utter_word[lu_ori_idx]]) + 2

                for uid, utter in enumerate(data['utterances']):
                    if tgt:
                        if uid == lu_utter_id:
                            origin_utter = data['utterances'][uid][utter_key]
                            origin_utter_word = origin_utter.split()
                            origin_utter_word = origin_utter_word[:lu_ori_idx] + ['<tgt>'] + [
                                origin_utter_word[lu_ori_idx]] + ['</tgt>'] + origin_utter_word[lu_ori_idx + 1:]
                            utter_txt = " ".join(origin_utter_word)
                        # elif uid < lu_utter_id:
                        #     utter_txt = utter[utter_key] + ' ' + tgted_target
                        # elif uid > lu_utter_id:
                        #     utter_txt = tgted_target + ' ' + utter[utter_key]
                        else:
                            utter_txt = utter[utter_key]

                    orig_tokens, bert_tokens, orig_to_tok_map = self.bert_tokenizer(utter_txt, use_sp_token)
                    utter_speaker.append(self.speaker2idx[utter['speaker']])
                    scene_tokens.append(bert_tokens)
                    scene_maps.append(orig_to_tok_map)
                    scene_special_tokens.append([0, len(bert_tokens) - 1])



                if max([max(map) for map in scene_maps]) > 255:
                    long_txt += 1
                    break

                bio = torch.ones(len(data['utterances']), MAX_LEN).long()
                for uid in range(len(data['utterances'])):
                    bio[uid][scene_maps[uid]] = 0


                lu_map = scene_maps[lu_utter_id]
                try:
                    lu_span = (lu_utter_id, lu_map[lu_ori_idx+1], lu_map[lu_ori_idx+2]-1)
                except:
                    lu_span = (lu_utter_id, lu_map[lu_ori_idx+1], len(scene_tokens[lu_utter_id])-2)

                # print(instance['lu'])
                # print(scene_tokens[lu_span[0]][lu_span[1]:lu_span[2]+1])


                fes = instance['elements']
                for fe_name, fe_info in fes.items():
                    if fe_name not in self.arg2idx.keys():
                        continue
                    fe_id = self.arg2idx[fe_name]
                    fe_idx = fe_info['idx']
                    if fe_idx[0] == -1:
                        continue
                    if fe_info['utter_id'][-1] != "r": # type: utter
                        fe_utter_id = int(fe_info['utter_id'].split('#')[-1])
                        fe_map = scene_maps[fe_utter_id]

                        for ii, fe_i in enumerate(fe_idx):
                            if fe_utter_id == lu_utter_id:
                                if fe_i > lu_ori_idx:
                                    fe_idx[ii] = fe_i + 2
                                elif fe_i == lu_ori_idx:
                                    fe_idx[ii] = fe_i + 1
                            # if fe_utter_id > lu_utter_id:
                            #     fe_idx[ii] += len_tgted_target

                        try:
                            fe_span = (fe_utter_id, fe_map[fe_idx[0]], fe_map[fe_idx[-1] + 1] - 1, fe_id)
                        except:
                            try:
                                if use_sp_token == 'sp':
                                    fe_span = (fe_utter_id, fe_map[fe_idx[0]], len(scene_tokens[fe_utter_id]) - 3, fe_id) #sp 아니면 -3 -> -2
                                else:
                                    fe_span = (fe_utter_id, fe_map[fe_idx[0]], len(scene_tokens[fe_utter_id]) - 2,
                                               fe_id)  # sp 아니면 -3 -> -2
                            except:
                                continue
                    else:  # speaker
                        candidates = []
                        for s_i, sp in enumerate(utter_speaker):

                            try:
                                a = self.speaker2idx[fe_info['text']]
                            except:
                                print(instance)

                            try:
                                a = self.speaker2idx[fe_info['text']]
                            except:
                                print(instance)

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
                        # fe_span = (fe_utter_id, 0, 0, fe_id)  # CLS token 사용
                        if use_sp_token == 'sp':
                            fe_span = (fe_utter_id, len(scene_tokens[fe_utter_id]) - 2, len(scene_tokens[fe_utter_id]) - 2, fe_id)  # SEP 사용 #sp 토큰 아니면 -2, -2 -> -1 -1
                        elif use_sp_token == 'cls':
                            fe_span = (fe_utter_id, 0, 0, fe_id)
                        else:
                            fe_span = (
                            fe_utter_id, len(scene_tokens[fe_utter_id]) - 1, len(scene_tokens[fe_utter_id]) - 1, fe_id)  # SEP 사용 #sp 토큰 아니면 -2, -2 -> -1 -1
                    a,b,c,d = fe_span

                    # print(lu_utter_id, a)
                    # print(fe_info)
                    # print(scene_tokens[a][b:c + 1], d)
                    # print()

                    fe_spans.append(fe_span)


                for uid, st, en, label in fe_spans:  # idx2arg, self.bio_arg2idx, idx2bio_arg
                    if st > 255:
                        continue
                    if en > 255:
                        continue
                    label_str = self.idx2arg[label]
                    bio[uid][st] = self.bio_arg2idx["B-{}".format(label_str)]
                    for ii in range(st+1, en+1):
                        if ii in scene_maps[uid]:
                            bio[uid][ii] = self.bio_arg2idx["I-{}".format(label_str)]

                if instance['lu'] not in self.lu2idx.keys():
                    # print(instance['lu'])
                    if lang == 'ko':
                        not_in_lu_dict += 1
                        continue
                    else:

                        # TODO get_enlu로 고치기.

                        token, pos = instance['lu'].split('.')
                        instance['lu'] = get_enlu(token, pos)

                    try:
                        lu_idx = self.lu2idx[instance['lu']]   # lu 사전에 없는애들이 발생하면 거르자..
                    except:
                        not_in_lu_dict += 1
                        continue

                text_bio = [[self.idx2bio_arg[x.item()] for x in b] for b in bio]
                lu_idx = self.lu2idx[instance['lu']]
                if sense not in self.lufrmap[str(lu_idx)]:  # 이거도 사전 업데이트를 해야함.
                    not_in_lufr_dict += 1
                    continue

                if constrant != list(range(len(data['utterances']))):
                    mapper = {}

                    for ii, val in enumerate(constrant):
                        mapper[val] = ii
                    bio = bio[constrant]

                    fe_spans = [(mapper[a],b,c,d) for (a,b,c,d) in fe_spans if a in constrant]
                    scene_maps = [map for ii, map in enumerate(scene_maps) if ii in constrant]
                    scene_tokens = [toks for ii, toks in enumerate(scene_tokens) if ii in constrant]
                    lu_span = (mapper[lu_span[0]], lu_span[1], lu_span[2])
                    utter_speaker = [sp for ii, sp in enumerate(utter_speaker) if ii in constrant]
                    scene_special_tokens = [toks for ii, toks in enumerate(scene_special_tokens) if ii in constrant]




                made += 1
                lus.append(lu_idx)
                bios.append(bio)
                senses.append(sense)
                gold_args.append(fe_spans) # fe_utter_id, st, en, fe_id
                orig_tok_to_maps.append(scene_maps)
                tokenized_texts.append(scene_tokens)
                targets.append(lu_span)
                speakers.append(utter_speaker)
                special_tokens.append(scene_special_tokens)

                # scene_id, iid 섞어만들기
                ep_id, s_id = scene_id.split('_')[:-1]
                ep_id = int(ep_id[-2:])
                s_id = int(s_id)
                instance_id = ep_id*10000000 + s_id*10000 + iid  # 00 000 0000 ep_id, scene_id, frame_id
                instance_ids.append(instance_id)

            if err_flag:
                err_flag = False
                break

        print("total instance: {}".format(total_num))
        print("not found LU in LU dict: {}".format(not_in_lu_dict))
        print("not found Frame in lufr dict: {}".format(not_in_lufr_dict))
        print("long_txt: {}".format(long_txt))
        print("input instance: {}".format(made))
        if len(tokenized_texts) == 0:
            return None

        tokenized_ids = []
        for scene_txt in tokenized_texts:
            scene_ids = pad_sequences([self.tokenizer.convert_tokens_to_ids(txt) for txt in scene_txt], maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
            tokenized_ids.append(scene_ids)
        max_utter = np.max([tokenized.shape[0] for tokenized in tokenized_ids])  # 50
        utter_len = [tokenized.shape[0] for tokenized in tokenized_ids]
        input_ids = pad_sequences(tokenized_ids, maxlen=max_utter, dtype="long", truncating="post", padding="post")

        tensor_maps = []
        for scene_map in orig_tok_to_maps:
            map = pad_sequences(scene_map, maxlen=MAX_LEN, dtype="long", truncating="post",
                                         padding="post", value=-1)
            tensor_maps.append(map)
        orig_tok_to_maps = pad_sequences(tensor_maps, maxlen=max_utter, dtype="long", truncating="post", padding="post", value=-1)
        tensor_bios = pad_sequences(bios, maxlen=max_utter, dtype="long", truncating="post", padding="post", value=-1)
        special_tokens = pad_sequences(special_tokens, maxlen=max_utter, dtype="long", truncating="post", padding="post", value=-1)

        attention_masks = [[[float(i > 0) for i in ii] for ii in input_id] for input_id in input_ids]
        token_type_ids = [[[0 if idx > 0 else 1 for idx in input_id] for input_id in input] for input in input_ids]

        args_len = [len(arg) for arg in gold_args]
        gold_args = pad_sequences(gold_args,
                                   maxlen=20, value=(-1, -1, -1, -1), padding="post",
                                   dtype="long", truncating="post")  # start, end, fe index

        speaker_len = [len(speaker) for speaker in speakers]
        speakers = pad_sequences(speakers,
                                  maxlen=max_utter, value=-1, padding="post",
                                  dtype="long", truncating="post")  # start, end, fe index

        data_inputs = torch.tensor(input_ids)
        utter_len = torch.tensor(utter_len)
        data_orig_tok_to_maps = torch.tensor(orig_tok_to_maps)
        data_token_type_ids = torch.tensor(token_type_ids)
        data_masks = torch.tensor(attention_masks)

        lus = torch.tensor(lus)
        gold_args = torch.tensor(gold_args)
        args_len = torch.tensor(args_len)
        targets = torch.tensor(targets)
        senses = torch.tensor(senses)
        speakers = torch.tensor(speakers)
        speaker_len = torch.tensor(speaker_len)
        special_tokens = torch.tensor(special_tokens)
        tensor_bios = torch.tensor(tensor_bios)
        instance_ids = torch.tensor(instance_ids)

        bert_inputs = TensorDataset(data_inputs, utter_len, data_orig_tok_to_maps, data_token_type_ids, data_masks, targets,
                                    senses, gold_args, args_len, speakers, speaker_len, special_tokens, tensor_bios, lus, instance_ids)

        return bert_inputs


def get_masks(datas, mapdata, num_label=2, masking=True):  # datas : lus [bsz, 1]
    masks = []
    with torch.no_grad():
        if masking == True:
            for idx in datas:
                torch.cuda.set_device(0)
                indx = idx.item()
                mask = torch.zeros(num_label)
                candis = mapdata[str(indx)]  # 해당 lu가 가질 수 있는 frame type indices.
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