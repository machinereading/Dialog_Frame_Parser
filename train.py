import json
import sys
import glob
import torch
sys.path.append('../')
import os
from transformers import *
from src import thesis_utils
from src import dataio
from src import eval_fn
from src.thesis_eval_fn import *
import frame_parser
from src.old_thesis_modeling import thesis_spoken_model
from src.thesis_modeling import bio_model
from seqeval.metrics.sequence_labeling import get_entities
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch
from torch import nn
from torch.optim import Adam
from tqdm import tqdm, trange
from pprint import pprint
import shutil
import pickle
import numpy as np
import random
from torch import autograd
use_sp_token = False
torch.manual_seed(1000)
np.random.seed(1000)
random.seed(1000)

model_dir = 'models/temp'
data_path = 'input_data/xxx.json'

###########################################################  평가시에 복붙

task = 'full' # full, argument, 34, 4, sentence
model_type = 'bio'  # bio
data_type = 'spoken' # written, spoken
language = 'en'  # ko, en
debug = False
only_lu_dict = True
epochs = 60
model_opts = {
    "fr_target_emb": False,
    "fr_context": False,
    "pair_dist": True,
    "pair_sense": True,
    "pair_speaker": True,
    "fe_target_emb": True,
    "fe_sense": True,
    "fe_lu": False,
    "fe_dist": True,
    "fe_speaker": True,
    "is_baseline": False,
    "use_sp_token": 'cls' #cls, sp, True, False
}
use_tgt = True
use_transfer = False
pretrained_dir = ''
early_stopping = True
early_stack = 5
batch_size = 1
lr = 3e-5

###########################################################

use_sp_token = model_opts["use_sp_token"]
print(model_dir)
datas = None
auto_split = False
srl = 'framenet'
fnversion = '1.2'
torch.cuda.empty_cache()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 실행시간 측정 함수
import time

if not os.path.exists(model_dir):
    os.makedirs(model_dir)


if model_dir[-1] != '/':
    model_dir = model_dir + '/'
f = open(model_dir + "results.txt", 'w')

_start_time = time.time()


def tic():
    global _start_time
    _start_time = time.time()


def tac():
    t_sec = round(time.time() - _start_time)
    (t_min, t_sec) = divmod(t_sec, 60)
    (t_hour, t_min) = divmod(t_min, 60)

    result = '{}hour:{}min:{}sec'.format(t_hour, t_min, t_sec)
    return result

try:
    dir_path = os.path.dirname(os.path.abspath( __file__ ))
except:
    dir_path = '.'

def train(PRETRAINED_MODEL="bert-base-multilingual-cased",
          model_dir=False, epochs=20, fnversion=False, early_stopping=True, datas=None, batch_size=4):
    tic()

    if model_dir[-1] != '/':
        model_dir = model_dir + '/'

    if early_stopping == True:
        model_saved_path = model_dir + 'best/'
        model_dummy_path = model_dir + 'dummy/'
        if not os.path.exists(model_dummy_path):
            os.makedirs(model_dummy_path)
    else:
        model_saved_path = model_dir

    if data_type == 'written':
        trn_data = bert_io.written_converter(trn)
        eval_data = bert_io.written_converter(dev)
    else:
        if auto_split == False:
            datas = bert_io.data_converter(datas, tgt=use_tgt, lang=language, use_sp_token=use_sp_token)
            trn_size = int(0.8*len(datas))
            eval_size = len(datas) - trn_size
            trn_data, eval_data = torch.utils.data.random_split(datas, [trn_size, eval_size])
        else:
            trn_data = bert_io.data_converter(datas, tgt=use_tgt, lang=language)
            eval_data = bert_io.written_converter(dev)




    if not os.path.exists(model_saved_path):
        os.makedirs(model_saved_path)
    print('\nyour model would be saved at', model_saved_path)

    # load a pre-trained model first
    print('\nloading a pre-trained model...')


    if model_type == 'bio':
        if use_transfer:
            model = bio_model.from_pretrained(PRETRAINED_MODEL,
                                                        task=task,
                                                        num_lus=len(bert_io.lu2idx),
                                                        num_senses=len(bert_io.sense2idx),
                                                        num_args=len(bert_io.bio_arg2idx),
                                                        lufrmap=bert_io.lufrmap,
                                                        frargmap=bert_io.bio_frargmap,
                                                        eval=False,
                                                        data_type=data_type,
                                                        model_opts=model_opts
                                                        )  # don't use written
            pre_model = bio_model.from_pretrained(pretrained_dir,
                                              task=task,
                                                num_lus=len(bert_io.lu2idx),
                                              num_senses=len(bert_io.sense2idx),
                                              num_args=len(bert_io.bio_arg2idx),
                                              lufrmap=bert_io.lufrmap,
                                              frargmap=bert_io.bio_frargmap,
                                              eval=True,
                                              data_type=data_type,
                                              model_opts=model_opts
                                              )  # don't use written
            model.bert = pre_model.bert
            model.attention = pre_model.attention
            model.frame_embs = pre_model.frame_embs
            model.sense_classifier = pre_model.sense_classifier
            model.arg_classifier = pre_model.arg_classifier
            model.lu_encoder = pre_model.lu_encoder
            model.distance = pre_model.distance
            model.speaker = pre_model.speaker

            del pre_model

        else:
            model = bio_model.from_pretrained(PRETRAINED_MODEL,
                                              task=task,
                                              num_lus=len(bert_io.lu2idx),
                                              num_senses=len(bert_io.sense2idx),
                                              num_args=len(bert_io.bio_arg2idx),
                                              lufrmap=bert_io.lufrmap,
                                              frargmap=bert_io.bio_frargmap,
                                              eval=True,
                                              data_type=data_type,
                                              model_opts=model_opts
                                              )  # don't use written

    else:
        model = sentence_model.from_pretrained(PRETRAINED_MODEL,
                                          task=task,
                                          num_lus=len(bert_io.lu2idx),
                                          num_senses=len(bert_io.sense2idx),
                                          num_args=len(bert_io.bio_arg2idx),
                                          lufrmap=bert_io.lufrmap,
                                          frargmap=bert_io.bio_frargmap,
                                          eval=True,
                                          data_type=data_type,
                                          model_opts=model_opts
                                          )  # don't use written
    model.to(device)
    print('... is done.', tac())

    print('\nconverting data to BERT input...')

    # trn_data = bert_io.convert_to_bert_input_e2e(trn)
    # sampler = RandomSampler(trn_data)
    trn_dataloader = DataLoader(trn_data, batch_size=batch_size) # sampler=sampler,
    eval_dataloader = DataLoader(eval_data, batch_size=1)  # sampler=sampler,
    print('... is done', tac())

    # load optimizer
    FULL_FINETUNING = True
    if FULL_FINETUNING:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
    else:
        param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]
    optimizer = Adam(optimizer_grouped_parameters, lr=lr)
    # optimizer = AdamW(optimizer_grouped_parameters)

    max_grad_norm = 1.0
    num_of_epoch = 0

    best_score = -1
    best_epoch = -1
    renew_stack = 0


    for n_ep in trange(epochs, desc="Epoch"):
        # TRAIN loop
        model.train()
        tr_loss = 0
        tr_frame_loss, tr_pair_loss, tr_linking_loss = 0,0,0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(trn_dataloader):
            # add batch to gpu

            torch.cuda.set_device(device)
            batch = tuple(t.to(device) for t in batch)
            # data_inputs, data_orig_tok_to_maps, data_token_type_ids, data_masks, utter_len, utter_speakers, lu_spans,
            # data_senses, gold_spans, span_pad, data_speakers, speaker_pad

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
                "bios": batch[12],
                "lus": batch[13]
            }

            frame_loss, pair_loss, linking_loss = model(**inputs)

            if data_type == "written":
                loss = (frame_loss + linking_loss) / 2
            else:
                if task == 'full':
                    loss = (frame_loss + pair_loss + linking_loss) / 3
                elif task == 'sentence':
                    loss = (frame_loss +linking_loss) / 2
                else:
                    loss = (pair_loss + linking_loss) / 2

            try:
                loss.backward()
            except:
                pass

            # track train loss
            if type(loss) == torch.Tensor:
                tr_loss += loss.item()
            if type(frame_loss) == torch.Tensor:
                tr_frame_loss += frame_loss.item()
            if type(pair_loss) == torch.Tensor:
                tr_pair_loss += pair_loss.item()
            if type(linking_loss) == torch.Tensor:
                tr_linking_loss += linking_loss.item()

            nb_tr_examples += batch[0].size(0)
            nb_tr_steps += 1

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)

            # update parameters
            optimizer.step()
            model.zero_grad()

        print("\nloss:{}".format(tr_loss))
        print("frame, pair, linking: {}, {}, {}".format(tr_frame_loss, tr_pair_loss, tr_linking_loss))


        model.eval()
        total_pred_utter, total_pred_frame, total_pred_labels, total_gold_utter, total_gold_frame, total_gold_labels = [], [], [], [], [], []
        total_gold_full, total_pred_full = [], []
        for step, batch in enumerate(eval_dataloader):
            # add batch to gpu
            torch.cuda.set_device(device)
            batch = tuple(t.to(device) for t in batch)
            # data_inputs, data_orig_tok_to_maps, data_token_type_ids, data_masks, utter_len, utter_speakers, lu_spans,
            # data_senses, gold_spans, span_pad, data_speakers, speaker_pad

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
                "bios": batch[12],
                "lus": batch[13]
            }

            if (torch.max(batch[2][0]) > 256).item() == 1:
                continue


            pred_frame, pred_utter, pred_labels = model(**inputs, train=False)
            answers = torch.squeeze(batch[7])[:batch[8]]

            if task == 'full' or task == 'sentence':
                total_pred_frame.append(('frame', pred_frame.item(), step))
                total_pred_full.append(('frame', pred_frame.item(), step))
                total_gold_frame.append(('frame', batch[6][0].item(), step))
                total_gold_full.append(('frame', batch[6][0].item(), step))



            gold_utter = torch.unique(answers[:, 0]).tolist()
            gold_utter = [(x, step) for x in gold_utter]
            gold_spans = answers[:, :-1].tolist()
            gold_spans = [(x, y, z, step) for x, y, z in gold_spans]



            total_gold_utter += gold_utter
            for uid in range(batch[1][0].item()): #utter_len


                cur_map = batch[2][0][uid].tolist()
                cur_map_idx = [x for x in cur_map if x != -1]
                gold_utter_logit = batch[12][0][uid][cur_map_idx]
                try:
                    gold_utter_label = [bert_io.idx2bio_arg[ii.item()] for ii in gold_utter_logit]
                except:
                    print('!!!!')
                gold_entities = get_entities(gold_utter_label)
                for entity in gold_entities:
                    tag, st, en = entity
                    if tag == 'X' or tag == 'O':
                        continue
                    total_gold_labels.append((uid, st, en, tag, step)) # bert_io.arg2idx[tag]
                    total_gold_full.append((uid, st, en, tag, step)) # bert_io.arg2idx[tag]

            try:
                pred_utter = pred_utter.tolist()
                pred_utter_list = pred_utter
                pred_utter = [(x, step) for x in pred_utter]
                total_pred_utter += pred_utter
            except:
                continue

            try:
                a = pred_labels[0]
            except:
                continue

            for uid, pred_utter_logit in enumerate(pred_labels):  # bert_io.idx2bio_arg
                if task =='sentence' and uid != batch[5][0][0].item():
                    continue

                cur_map = batch[2][0][uid].tolist()
                cur_map_idx = [x for x in cur_map if x != -1]
                pred_utter_logit = pred_utter_logit[cur_map_idx]
                pred_utter_label = [bert_io.idx2bio_arg[ii.item()] for ii in pred_utter_logit]
                pred_entities = get_entities(pred_utter_label)

                for entity in pred_entities:
                    tag, st, en = entity
                    if tag == 'X' or tag == 'O':
                        continue
                    if model_opts["is_baseline"] == False and uid not in pred_utter_list:
                        continue

                    total_pred_labels.append((uid, st, en, tag, step)) # bert_io.arg2idx[tag]
                    total_pred_full.append((uid, st, en, tag, step))

                    # print("GOLD")
                    # print(total_gold_labels)
                    # print("PRED")
                    # print(total_pred_labels)


        utter_score = {
            "precision": precision_1d(total_gold_utter, total_pred_utter),
            "recall": recall_1d(total_gold_utter, total_pred_utter),
            "f1": f1_1d(total_gold_utter, total_pred_utter)
        }
        frame_score = {
            "precision": precision_1d(total_gold_frame, total_pred_frame),
            "recall": recall_1d(total_gold_frame, total_pred_frame),
            "f1": f1_1d(total_gold_frame, total_pred_frame)
        }
        label_score = {
            "precision": precision_1d(total_gold_labels, total_pred_labels),
            "recall": recall_1d(total_gold_labels, total_pred_labels),
            "f1": f1_1d(total_gold_labels, total_pred_labels)
        }
        full_score = {
            "precision": precision_1d(total_gold_full, total_pred_full),
            "recall": recall_1d(total_gold_full, total_pred_full),
            "f1": f1_1d(total_gold_full, total_pred_full)
        }

        summary = {
            "utter_score": utter_score,
            "frame_score": frame_score,
            "label_score": label_score,
            "full_score": full_score
        }

        print("utter_score")
        print(utter_score)
        print('----------')
        print("frame_score")
        print(frame_score)
        print('----------')
        print("label_score")
        print(label_score)
        print('----------')
        print("full_score")
        print(full_score)
        print('----------')

        num_of_epoch += 1

        if best_score < full_score["f1"]:
            best_score = full_score["f1"]
            best_epoch = num_of_epoch

            model_saved_path = model_dir + 'best' + '/'
            if os.path.exists(model_saved_path):
                shutil.rmtree(model_saved_path)
            os.makedirs(model_saved_path)
            model.save_pretrained(model_saved_path)

        # model_saved_path = model_dir + str(num_of_epoch) + '/'
        # if not os.path.exists(model_saved_path):
        #     os.makedirs(model_saved_path)
        # model.save_pretrained(model_saved_path)


        f.write('\n' + str(num_of_epoch))
        f.write('\n' + str(loss) )
        f.write("\nutter_prec: {}".format(utter_score['precision']))
        f.write("\nutter_rec: {}".format(utter_score['recall']))
        f.write("\nutter_f1: {}".format(utter_score['f1']))
        f.write('\n')
        f.write("\nframe_acc: {}".format(frame_score['precision']))
        f.write('\n')
        f.write("\nlabel_prec: {}".format(label_score['precision']))
        f.write("\nlabel_rec: {}".format(label_score['recall']))
        f.write("\nlabel_f1: {}".format(label_score['f1']))
        f.write('\n')
        f.write("\nfull_prec: {}".format(full_score['precision']))
        f.write("\nfull_rec: {}".format(full_score['recall']))
        f.write("\nfull_f1: {}".format(full_score['f1']))
        f.write('\n\n')

        cur_f = open(model_dir + "{}.txt".format(num_of_epoch), 'w')
        cur_f.write('\n' + str(loss) + '\n')
        cur_f.write('\n' + str(num_of_epoch))
        cur_f.write('\n' + str(loss))
        cur_f.write("\nutter_prec: {}".format(utter_score['precision']))
        cur_f.write("\nutter_rec: {}".format(utter_score['recall']))
        cur_f.write("\nutter_f1: {}".format(utter_score['f1']))
        cur_f.write('\n')
        cur_f.write("\nframe_acc: {}".format(frame_score['precision']))
        cur_f.write('\n')
        cur_f.write("\nlabel_prec: {}".format(label_score['precision']))
        cur_f.write("\nlabel_rec: {}".format(label_score['recall']))
        cur_f.write("\nlabel_f1: {}".format(label_score['f1']))
        cur_f.write('\n')
        cur_f.write("\nfull_prec: {}".format(full_score['precision']))
        cur_f.write("\nfull_rec: {}".format(full_score['recall']))
        cur_f.write("\nfull_f1: {}".format(full_score['f1']))
        cur_f.write('\n')
        cur_f.write("cur_best full-f1 score: {}, {}".format(best_epoch, best_score))
        cur_f.write('\n\n')

    print('...training is done. (', tac(), ')')
    f.write("best full-f1 score: {}, {}".format(best_epoch, best_score))




if data_type == 'written':
    bert_io = thesis_utils.for_BERT(mode='train', language=language, masking=True, fnversion=fnversion)
    trn, dev, tst = dataio.load_data(language=language, fnversion=fnversion, debug=debug, tgt=use_tgt)
else:
    bert_io = thesis_utils.for_BERT(mode='train', language=language, masking=True, fnversion=fnversion)

    if debug:
        path = '/home/fairy_of_9/frameBERT/vtt_data/converted/debug_1104.json'
    elif language == 'ko':
        path = '/home/fairy_of_9/frameBERT/vtt_data/converted/ko_1117.json'
    else:
        path = '/home/fairy_of_9/frameBERT/vtt_data/converted/en_1117.json'
    if auto_split:
        trn, dev, tst = dataio.load_thesis_data(
            path,
            auto_split=auto_split, tgt=use_tgt, task=task)
    else:
        datas = dataio.load_thesis_data(
            path,
            auto_split=auto_split, tgt=use_tgt, task=task)


train(epochs=epochs, model_dir=model_dir, fnversion=fnversion, early_stopping=early_stopping, batch_size=batch_size, datas=datas)

f.close()
print(model_dir)