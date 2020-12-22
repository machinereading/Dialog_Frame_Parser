from src import dataio
import frame_parser
from src.thesis_modeling import bio_model
from src import eval_fn
import torch
from src import thesis_utils
from pprint import pprint
from sklearn.metrics import confusion_matrix
from collections import defaultdict
import json
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import seaborn as sns
import numpy as np
import pandas as pd
import itertools
from seqeval.metrics.sequence_labeling import get_entities
from src.thesis_eval_fn import *
import pylab as plt
import time

model_dir = 'models/temp/best'
data_path = 'input_data/xxx.json'



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

####################################################################################


n_instance = 0

torch.cuda.empty_cache()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
fnversion=1.2
bert_io = thesis_utils.for_BERT(mode='train', language=language, masking=True, fnversion=fnversion)

with open('/home/fairy_of_9/frameBERT/koreanframenet/resource/info/fn1.7_bio_fe2idx.json', 'r') as f:
    bio_arg2idx = json.load(f)
idx2bio_arg = dict(zip(bio_arg2idx.values(), bio_arg2idx.keys()))

frame_cnt = defaultdict(int)
arg_list = []
for k in bio_arg2idx.keys():
    arg_list.append(k)

if data_type == 'written':
    trn, dev, tst = dataio.load_data(language=language, fnversion=fnversion, debug=debug, tgt=use_tgt)
else:
    trn, dev, tst = dataio.load_thesis_data(
                    data_path,
                    auto_split=True, tgt=use_tgt)


model = bio_model.from_pretrained(model_dir,
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

start = time.time()

if data_type == 'written':
    eval_data = bert_io.written_converter(dev)
else:
    eval_data = bert_io.data_converter(dev, tgt=use_tgt, lang=language)

eval_dataloader = DataLoader(eval_data, batch_size=1)


model.eval()
total_pred_utter, total_pred_frame, total_pred_labels, total_gold_utter, total_gold_frame, total_gold_labels = [], [], [], [], [], []
total_gold_full, total_pred_full = [], []
for step, batch in enumerate(eval_dataloader):
    n_instance += 1
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

    pred_frame, pred_utter, pred_labels = model(**inputs, train=False)
    answers = torch.squeeze(batch[7])[:batch[8]]

    if task == 'full':
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
        gold_utter_label = [bert_io.idx2bio_arg[ii.item()] for ii in gold_utter_logit]
        gold_entities = get_entities(gold_utter_label)
        for entity in gold_entities:
            tag, st, en = entity
            if tag == 'X' or tag == 'O':
                continue
            total_gold_labels.append((uid, st, en, tag, step)) # bert_io.arg2idx[tag]
            total_gold_full.append((uid, st, en, tag, step)) # bert_io.arg2idx[tag]

    try:
        pred_utter = pred_utter.tolist()
        pred_utter = [(x, step) for x in pred_utter]
        total_pred_utter += pred_utter
    except:
        continue

    try:
        a = pred_labels[0]
    except:
        continue

    for uid, pred_utter_logit in enumerate(pred_labels):  # bert_io.idx2bio_arg
        cur_map = batch[2][0][uid].tolist()
        cur_map_idx = [x for x in cur_map if x != -1]
        pred_utter_logit = pred_utter_logit[cur_map_idx]
        pred_utter_label = [bert_io.idx2bio_arg[ii.item()] for ii in pred_utter_logit]
        pred_entities = get_entities(pred_utter_label)

        for entity in pred_entities:
            tag, st, en = entity
            if tag == 'X' or tag == 'O':
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

print("n_instance: {}".format(n_instance))
print(time.time()-start)
print('second')

del model





level_result = {
    "sentence": [0,0],  # True, False
    "dialog": [0,0],
    "speaker": [0,0]
}

confusion_dict = {}
used_label = set()

def plot_confusion_matrix(cm, target_names=None, cmap=None, normalize=True, labels=True, title='Confusion matrix'):
    # accuracy = np.trace(cm) / float(np.sum(cm))
    # misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names)
        plt.yticks(tick_marks, target_names)

    if labels:
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
            else:
                plt.text(j, i, "{:,}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    # plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()