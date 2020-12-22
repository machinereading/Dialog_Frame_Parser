import json
import sys
import glob
import torch
sys.path.insert(0,'../')
sys.path.insert(0,'../../')
from transformers import *
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from src import utils

from koreanframenet import koreanframenet
from src import kotimex

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import os
try:
    dir_path = os.path.dirname(os.path.abspath( __file__ ))
except:
    dir_path = '.'
    
dir_path = dir_path+'/..'

def fe2arg(data):
    result = []
    for i in data:
        tokens, preds, senses, args = i[0], i[1], i[2], i[3]
        new_args = []
        for arg in args:
            if '-' in arg:
                bio = arg.split('-')[0]
                new_arg = bio+'-'+'ARG'
            else:
                new_arg = arg
            new_args.append(new_arg)
            
        sent = []
        sent.append(tokens)
        sent.append(preds)
        sent.append(senses)
        sent.append(new_args)
        
        result.append(sent)
        
    return result                
    
def conll2tagseq(data):
    tokens, preds, senses, args = [],[],[],[]
    result = []
    for line in data:
        line = line.strip()
        if line.startswith('#'):
            pass
        elif line != '':
            t = line.split('\t')
            token, pred, sense, arg = t[1], t[2], t[3], t[4]            
            tokens.append(token)
            preds.append(pred)
            senses.append(sense)
            args.append(arg)
        else:
            sent = []
            sent.append(tokens)
            sent.append(preds)
            sent.append(senses)
            sent.append(args)
            result.append(sent)
            tokens, preds, senses, args = [],[],[],[]
            
    return result


def load_thesis_data(data_dir, auto_split=True, tgt=False, task='full'):  # auto_split : demo일 시 False
    j = utils.jsonload(data_dir)
    n_deleted_frame = 0
    total_frame = 0

    for scene_id, scene in j.items():  # FE가 0개, LU 조사에 따른 filter
        del_list = []
        for i, frame in enumerate(scene['frames']):
            total_frame+=1
            if len(frame['elements']) == 0:
                del_list.append(i)
            elif frame['lu'].split('.')[-1].lower() not in ['jj', 'vb', 'a', 'jjr', 'jjs', 'v', 'vbd', 'vbg', 'vbn', 'vbp', 'vbz']:
                del_list.append(i)

        del_list.sort(reverse=True)
        for i in del_list:
            n_deleted_frame += 1
            del scene['frames'][i]

    print("# total frames in load_thesis_data : {}".format(total_frame))

    if auto_split:
        trn, dev, test = dict(list(j.items())[:int(len(j)*0.8)]), dict(list(j.items())[int(len(j)*0.8):int(len(j)*0.9)]), dict(list(j.items())[int(len(j)*0.9):])
        print("dataset scene-len (total, train, dev, test):")
        print(len(j), len(trn),len(dev),len(test))
        print()
        cnt = 0
        for k,v in trn.items():
            cnt += len(v['frames'])
        print("# frames in trn : {}".format(cnt))

        return trn, dev, test
    else:
        return j



def load_vtt_data(data_dir, auto_split=True):  # auto_split : demo일 시 False
    j = utils.jsonload(data_dir)

    print()
    del_list = []
    for i, anno in enumerate(j):
        if len(anno['frame']['elements']) == 0:
            del_list.append(i)
        if anno['frame']['lu'].split('.')[-1] not in ['JJ','VB','a','JJR','JJS','v','VBD','VBG','VBN','VBP','VBZ']:
            del_list.append(i)
        # for s in anno['speakers']:
        #     if s not in []:
        #         del_list.append(i)

    del_list = list(set(del_list))
    del_list.sort(reverse=True)
    for i in del_list:
        del j[i]

    if auto_split:
        # trn, dev, test = j[:int(len(j)*0.8)], j[int(len(j)*0.8):int(len(j)*0.9)], j[int(len(j)*0.9):]
        # trn, dev, test = j[:239], j[239:439], j[439:]
        # trn, dev, test = j[:71], j[71:121], j[121:]
        trn, dev, test = j[:71], j[71:], []

        print("dataset len (total, train, dev, test):")
        print(len(j), len(trn),len(dev),len(test))

        return trn, dev, test
    else:
        return j



    
def load_data(srl='framenet', language='ko', fnversion=1.2, task='', path=False, exem=False, info=True, debug=False, tgt=True):
    if 'framenet' in srl:
        if language == 'ko':
            kfn = koreanframenet.interface(version=fnversion, info=info)
            trn_d, dev_d, tst_d = kfn.load_data(debug=debug, task=task)
            '''
            list of example
            example : conll(list of 4 list)
                text, LU, LU에 대한 frame type, args에 대한 BIO
            '''
        else:
            if path == False:
                fn_dir = 'data/fn1.7/'
            else:
                fn_dir = path
            with open(fn_dir+'fn1.7.fulltext.train.syntaxnet.conll') as f:
                d = f.readlines()
            trn_d = conll2tagseq(d)
            
            if exem:
                with open(fn_dir+'fn1.7.exemplar.train.syntaxnet.conll') as f:
                    d = f.readlines()
                exem_d = conll2tagseq(d)
            with open(fn_dir+'fn1.7.dev.syntaxnet.conll') as f:
                d = f.readlines()
            dev_d = conll2tagseq(d)
            with open(fn_dir+'fn1.7.test.syntaxnet.conll') as f:
                d = f.readlines()
            tst_d = conll2tagseq(d)
    else:
        if language == 'ko':
            if path == False:
                fn_dir = '/disk/data/corpus/koreanPropBank/original/'
            else:
                fn_dir = path
            if srl == 'propbank-dp':
                
                with open(fn_dir+'srl.dp_based.train.conll') as f:
                    d = f.readlines()
                trn_d = conll2tagseq(d)
                with open(fn_dir+'srl.dp_based.test.conll') as f:
                    d = f.readlines()
                tst_d = conll2tagseq(d)
                dev_d = []
            else:
                with open(fn_dir+'srl.span_based.train.conll') as f:
                    d = f.readlines()
                trn_d = conll2tagseq(d)
                with open(fn_dir+'srl.span_based.test.conll') as f:
                    d = f.readlines()
                tst_d = conll2tagseq(d)
                dev_d = []


    if tgt:
        trn_d = data2tgt_data(trn_d, mode='train')  # conll에 <tgt> token 추가.
    if language == 'en':
        if exem:
            exem_d = data2tgt_data(exem_d, mode='train')
    if tgt:
        tst = data2tgt_data(tst_d, mode='train')
    else:
        tst = tst_d
    if dev_d:
        if tgt:
            dev = data2tgt_data(dev_d, mode='train')
        else:
            dev = dev_d
    else:
        dev = []
        
#     too_long_in_exem = [35285, 35286, 58002, 77448, 77993, 82010, 82061, 98118, 120524, 153131]
# #     too_long_in_exem = []
#     new_exem = []
#     for idx in range(len(exem)):
#         if idx in too_long_in_exem:
#             pass
#         else:
#             item = exem[idx]
#             new_exem.append(item)
        
#     trn += new_exem

    if language == 'en':
        if exem == True:
            ori_trn = trn_d + exem_d
            trn = []

            too_long = [35285, 35286, 58002, 77448, 77993, 82010, 82061, 98118, 120524, 153131]
            for idx in range(len(ori_trn)):
                if idx in too_long:
                    pass
                else:
                    item = ori_trn[idx]
                    trn.append(item)
        else:
            trn = trn_d
    else:
        trn = trn_d
    
        
#     print('# of instances in trn:', len(trn))
#     print('# of instances in dev:', len(dev))
#     print('# of instances in tst:', len(tst))
#     print('data example:', trn[0])
    
    return trn, dev, tst
                
def data2tgt_data(input_data, mode=False):
    result = []
    for item in input_data:
        
        if mode == 'train':
            ori_tokens, ori_preds, ori_senses, ori_args = item[0],item[1],item[2],item[3]
        else:
            ori_tokens, ori_preds = item[0],item[1]
        for idx in range(len(ori_preds)):
            pred = ori_preds[idx]
            if pred != '_':
                if idx == 0:
                    begin = idx
                elif ori_preds[idx-1] == '_':
                    begin = idx
                end = idx
                
        tokens, preds, senses, args = [],[],[],[]
        for idx in range(len(ori_preds)):
            token = ori_tokens[idx]
            pred = ori_preds[idx]
            if mode == 'train':
                sense = ori_senses[idx]
                arg = ori_args[idx]
                
            if idx == begin:
                tokens.append('<tgt>')
                preds.append('_')
                if mode == 'train':
                    senses.append('_')
                    args.append('X')

            tokens.append(token)
            preds.append(pred)
            
            if mode == 'train':
                senses.append(sense)
                args.append(arg)

            if idx == end:
                tokens.append('</tgt>')
                preds.append('_')
                
                if mode == 'train':
                    senses.append('_')
                    args.append('X')
        sent = []
        sent.append(tokens)
        sent.append(preds)
        if mode == 'train':
            sent.append(senses)
            sent.append(args)
            
        result.append(sent)
    return result

def preprocessor(input_data):
    if type(input_data) == str:
        data = input_data.split(' ')
        result = []
        result.append(data)
    elif type(input_data) == list:
        result = input_data
    else:
        data = input_data['text'].split(' ')
        result = []
        result.append(data)
    return result
             
def lu2target(lus):
    target_idx = []
    for i in range(len(lus)):
        lu = lus[i]
        if lu != '_':
            target_idx.append(i)
    
    return target_idx

def target_idx2target(tokens, target_idx):
    target_tokens = []
    for i in range(len(tokens)):
        token = tokens[i]
        if i in target_idx:
            target_tokens.append(token)
    target = ' '.join(target_tokens)
    
    return target
        

def topk(conll_result, sense_candis_list):
    result = {}
    
    if conll_result:
        targets = []
        tokens = conll_result[0][0]
        for i in range(len(conll_result)):
            lus = conll_result[i][1]
            target_idx = lu2target(lus)
            target_word = target_idx2target(tokens, target_idx)
            sense_candis = sense_candis_list[i]
            
            target = {}
            target['target_index'] = target_idx
            target['target'] = target_word
            target['frame_candidates'] = sense_candis
            targets.append(target)

        result['tokens'] = tokens
        result['#_of_candidates'] = len(sense_candis)
        result['targets'] = targets
        
    return result
        
    
def get_frame_lu(tokens, frames, lus):
    lu_token_list = []
    frame = False
    for i in range(len(frames)):
        f = frames[i]
        if f != '_':
            frame = f
            lu = lus[i]
            lu_token_list.append(tokens[i])            
    lu_token = ' '.join(lu_token_list)
    
    return frame, lu, lu_token
            
def remove_josa(phrase):
    from konlpy.tag import Kkma
    kkma = Kkma()
    import jpype
    jpype.attachThreadToJVM()
    
    tokens = phrase.split(' ')

    result = []
    for i in range(len(tokens)):
        token = tokens[i]
        if i < len(tokens)-1:
            result.append(token)
        else:
            m = kkma.pos(tokens[i])
            if m[-1][-1].startswith('J'):
                m.pop(-1)
                token = ''.join([t for t,p in m])
            result.append(token)
    result = ' '.join(result)
    return result
            
def frame2rdf(frame_conll, sent_id=False, language='ko'):
    triples = []
    n = 0
    for anno in frame_conll:
        tokens, lus, frames, args = anno[0],anno[1],anno[2],anno[3]
        frame, lu, lu_token = get_frame_lu(tokens, frames, lus)
        if frame:
            if type(sent_id) != bool:
                triple = ('frame'+'#'+str(sent_id)+'-'+str(n)+':'+frame, 'lu', lu_token)
                triples.append(triple)
#                 triple = ('frame:'+frame+'#'+str(sent_id), 'frdf:target', lu_token)
#                 triples.append(triple)
            else:
                triple = ('frame'+'#'+str(n)+':'+frame, 'lu', lu_token)
                triples.append(triple)
#                 triple = ('frame:'+frame, 'frdf:target', lu_token)
#                 triples.append(triple)

        if frame:
            sbj = False
            pred_obj_tuples = []
            
            for idx in range(len(args)):
                arg_tag = args[idx]
                arg_tokens = []
                if arg_tag.startswith('B'):
                    fe_tag = arg_tag.split('-')[1]
                    arg_tokens.append(tokens[idx])
                    next_idx = idx + 1
                    while next_idx < len(args) and args[next_idx] == 'I-'+fe_tag:
                        arg_tokens.append(tokens[next_idx])
                        next_idx +=1
                    arg_text = ' '.join(arg_tokens)
                    
                    if language =='ko':
                        arg_text = remove_josa(arg_text)
                    else:
                        pass
                    fe = fe_tag

                    # string to xsd
                    if fe == 'Time':
                        arg_text = kotimex.time2xsd(arg_text)
                    else:
                        pass
#                         arg_text = '\"'+arg_text+'\"'+'^^xsd:string'

#                     rel = 'frdf:'+frame+'-'+fe
                    rel = 'fe:'+fe

                    if rel == 'S':
                        pass
                    else:
                        p = rel
                        o = arg_text
                        pred_obj_tuples.append( (p,o) )

            for p, o in pred_obj_tuples:
                if sbj:
                    s = sbj
                else:
                    if type(sent_id) != bool:
                        s = 'frame'+'#'+str(sent_id)+'-'+str(n)+':'+frame
                    else:
                        s = 'frame'+'#'+str(n)+':'+frame
                triple = (s, p, o)
                triples.append(triple)
        n +=1
    return triples


        
    