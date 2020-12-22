from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from boltons.iterutils import windowed
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, CosineSimilarity

import sys
sys.path.insert(0,'../')
sys.path.insert(0,'../../')
from src import dataio, utils
from src.ds import *
from torch.nn.parameter import Parameter
from transformers import BertModel, BertPreTrainedModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()

def to_var(x):
    """ Convert a tensor to a backprop tensor and put on GPU """
    return to_cuda(x).requires_grad_()

def to_cuda(x):
    """ GPU-enable a tensor """
    if torch.cuda.is_available():
        x = x.cuda()
    return x

class Score(nn.Module):
    """ Generic scoring module
    """
    def __init__(self, embeds_dim, hidden_dim=150):
        super().__init__()

        self.score = nn.Sequential(
            nn.Linear(embeds_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        """ Output a scalar score for an input x """
        return self.score(x)


class Distance(nn.Module):
    """ Learned, continuous representations for: span widths, distance
    between spans
    """

    bins = [1,2,3,4,8,16,32,64]

    def __init__(self, distance_dim=20):
        super().__init__()

        self.dim = distance_dim
        self.embeds = nn.Sequential(
            nn.Embedding(len(self.bins)+1, distance_dim),
            nn.Dropout(0.20)
        )

    def forward(self, *args):
        """ Embedding table lookup """
        return self.embeds(self.stoi(*args))

    def stoi(self, lengths):
        """ Find which bin a number falls into """
        return (torch.tensor([
            sum([True for i in self.bins if num >= i]) for num in lengths], requires_grad=False).to(device)
        )

class BIO_span_representation(BertPreTrainedModel):
    def __init__(self, config, num_senses=2, num_args=2, lufrmap=None, frargmap=None, masking=True,
                 return_pooled_output=False, original_loss=False):
        super(BIO_span_representation, self).__init__(config)

        ## TODO tgt 위치넣어서, distnace 및 trigger emb 추가.

        frame_size = config.hidden_size
        width_size = 20
        fe_size = config.hidden_size

        self.masking = masking
        self.num_senses = num_senses  # total number of all frames
        self.num_args = num_args  # total number of all frame elements
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.sense_classifier = nn.Linear(config.hidden_size, num_senses)
        self.span_classifier = nn.Linear(config.hidden_size, 4)  # all FEs
        self.lufrmap = lufrmap  # mapping table for lu to its frame candidates, lu idx와 해당 lu가 속할 수 있는 frame indices.
        self.frargmap = frargmap  # mapping table for frame to its frame element candidates
        self.return_pooled_output = return_pooled_output
        self.width = Distance(width_size)
        self.original_loss = original_loss
        self.attention = Score(config.hidden_size)

        self.frame_embs = nn.Embedding(num_senses, frame_size)
        self.fe_embs = nn.Embedding(num_args + 1, fe_size)  # 1285번째 idx는 None으로

        self.pair_score = Score(3 * config.hidden_size + fe_size + width_size + fe_size)

        self.hidden_size = config.hidden_size

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, lus=None, senses=None, args=None,
                using_gold_fame=False, position_ids=None, head_mask=None, gold_spans=None, span_pads=None, span_bios=None):
        # token_type_ids : segment token indices [0,1] 0: sent A, 1: sent B
        sequence_output, pooled_output = self.bert(input_ids, token_type_ids=token_type_ids,
                                                   attention_mask=attention_mask)
        # seq_output : [bsz, seq_size(256), hidden_size], pooled_output : [bsz, hidden_size]
        sequence_output = self.dropout(sequence_output)
        pooled_output = self.dropout(pooled_output)

        attns = self.attention(sequence_output)

        sense_logits = self.sense_classifier(pooled_output)  # [bsz, # of frame type]
        span_logits = self.span_classifier(sequence_output)  # [bsz, seq_size(256), # of FEs]

        lufr_masks = utils.get_masks(lus, self.lufrmap, num_label=self.num_senses, masking=self.masking).to(
            device)  # [bsz, # of frame type] lu가 가질 수 있는 frame type이라면 1, 아니면 0

        sense_loss = 0  # loss for sense id
        span_loss = 0
        arg_loss = 0  # loss for arg id

        if senses is not None:  # is train
            for i in range(sense_logits.shape[0]):  # batch를 iter.
                sense_logit = sense_logits[i]  # [# of frame types]
                span_logit = span_logits[i]  # [seq_size(256), 4]
                attn = attns[i]
                seq_output = sequence_output[i]

                lufr_mask = lufr_masks[i]  # [# of frame types]

                gold_sense = senses[i]  # [1]
                # gold_arg = args[i]  # [seq_size(256)]
                gold_span_bio = span_bios[i]
                gold_span = gold_spans[i][:span_pads[i]]
                span_pad = span_pads[i]

                # train sense classifier
                loss_fct_sense = CrossEntropyLoss(weight=lufr_mask)
                loss_per_seq_for_sense = loss_fct_sense(sense_logit.view(-1, self.num_senses), gold_sense.view(-1))
                sense_loss += loss_per_seq_for_sense

                masked_sense_logit = utils.masking_logit(sense_logit, lufr_mask)
                pred_sense, sense_score = utils.logit2label(masked_sense_logit)  # arg max
                pred_sense = pred_sense.to(device)
                frame_emb = self.frame_embs(pred_sense)

                # train span classifier
                # frarg_mask = \
                # utils.get_masks([pred_sense], self.frargmap, num_label=self.num_args, masking=True).to(device)[0]
                """ 'X':0, 'O':1, 'B':2, 'I':3 """
                weights = [1,1,5,2.5]
                weights = torch.FloatTensor(weights).to(device)
                loss_fct_span = CrossEntropyLoss(weights)

                # only keep active parts of loss
                if attention_mask is not None:  # attention_mask == pad 인 애들만 False,
                    active_loss = attention_mask[i].view(-1) == 1
                    active_logits = span_logit.view(-1, 4)[active_loss]
                    active_labels = gold_span_bio.view(-1)[active_loss]
                    loss_per_seq_for_span = loss_fct_span(active_logits, active_labels)
                else:
                    loss_per_seq_for_span = loss_fct_span(span_logit.view(-1, 4), gold_span_bio.view(-1))
                span_loss += loss_per_seq_for_span

                # train labeling classifier
                pred_spans = utils.logit2span(active_logits)



                n_span = len(pred_spans)
                if n_span == 0:
                    continue

                g_is = []
                for st, en in pred_spans:
                    span_attn, span_emb = attn[st:en + 1], seq_output[st:en + 1]  # [span_len,1], [span_len, 768]
                    attn_weights = F.softmax(span_attn, dim=1)  # [span_len, 1]
                    attn_embed = torch.sum(torch.mul(span_emb, attn_weights), dim=0)  # [768]
                    width = self.width([st - en + 1])  # [1, 20]
                    g_i = torch.cat((seq_output[st], seq_output[en], attn_embed, torch.squeeze(frame_emb),
                                     torch.squeeze(width)))  # [768*3 + 20*2]
                    g_is.append(g_i)

                g_is = torch.stack(g_is)  # [#span, 2344]
                fe_indices = (torch.tensor(self.frargmap[str(pred_sense.item())])).to(device)  # [#fe]
                n_fe = fe_indices.shape[0]
                # candidates를 가져와서 span과 scoring -> softmax
                # CELoss로 loss 계산.
                fe_embs = self.fe_embs(fe_indices)  # [#fe, 768]
                fe_embs = torch.stack([fe_embs] * n_span, dim=0)  # [#span, #fe, 768]

                g_is = torch.stack([g_is] * n_fe, dim=1)  # [#span, #fe, 2344]

                features = torch.cat((g_is, fe_embs), dim=2)  # [#span, #fe, 3112]
                pair_scores = self.pair_score(features)  # [#span, #fe, 1]
                pair_scores = torch.squeeze(pair_scores, dim=2)  # [#span, #fe]
                epsilon = (torch.zeros(n_span, 1)).to(device)  # [#span, 1]

                scores_with_epsilon = torch.cat((pair_scores, epsilon), dim=1)  # [span, fe+1]
                probs = scores_with_epsilon  # [span, fe+1]
                # probs = [F.softmax(score) for score in scores_with_epsilon]
                # probs = torch.stack(probs)  # [span, fe+1]
                gold_probs = []
                for s_i, (st,en) in enumerate(pred_spans):
                    gold_probs.append(n_fe)  # epsilon.
                    for ii, gold_arg in enumerate(gold_span):
                        if (st, en) == (gold_arg[0].item(), gold_arg[1].item()):
                            label = gold_arg[2].item()
                            for jj, idx, in enumerate(fe_indices):
                                if label == idx:
                                    gold_probs[-1] = jj
                                    break
                            break
                gold_probs = (torch.tensor(gold_probs)).to(device)

                weights = [1]*n_fe + [0]
                weights = torch.FloatTensor(weights).to(device)

                loss_fct_arg = CrossEntropyLoss(weights)
                arg_loss += loss_fct_arg(probs, gold_probs)


            span_loss = span_loss / len(sense_logits)
            sense_loss = sense_loss / len(sense_logits)
            arg_loss = arg_loss / len(sense_logits)

            if self.return_pooled_output:
                return pooled_output, sense_loss, span_loss, arg_loss
            else:
                return sense_loss, span_loss, arg_loss
        else:  # is inference
            pred_frames, pred_scores, pred_span = [], [], []
            for i in range(sense_logits.shape[0]):  # bsz, inference에선 1.
                sense_logit = sense_logits[i]  # [# of frame types]
                span_logit = span_logits[i]  # [seq_size(256), 4]
                attn = attns[i]
                seq_output = sequence_output[i]
                lufr_mask = lufr_masks[i]  # [# of frame types]

                masked_sense_logit = utils.masking_logit(sense_logit, lufr_mask)
                pred_sense, sense_score = utils.logit2label(masked_sense_logit)  # arg max
                pred_sense = pred_sense.to(device)
                frame_emb = self.frame_embs(pred_sense)

                # only keep active parts of loss
                if attention_mask is not None:  # attention_mask == pad 인 애들만 False,
                    active_loss = attention_mask[i].view(-1) == 1
                    active_logits = span_logit.view(-1, 4)[active_loss]

                # train labeling classifier
                pred_spans = utils.logit2span(active_logits)
                n_span = len(pred_spans)
                if n_span == 0:
                    pred_frames.append(pred_sense)
                    pred_scores.append(torch.tensor([]))
                    pred_span.append([])
                    continue

                g_is = []
                for st, en in pred_spans:
                    span_attn, span_emb = attn[st:en + 1], seq_output[st:en + 1]  # [span_len,1], [span_len, 768]
                    attn_weights = F.softmax(span_attn, dim=1)  # [span_len, 1]
                    attn_embed = torch.sum(torch.mul(span_emb, attn_weights), dim=0)  # [768]
                    width = self.width([st - en + 1])  # [1, 20]
                    g_i = torch.cat((seq_output[st], seq_output[en], attn_embed, torch.squeeze(frame_emb),
                                     torch.squeeze(width)))  # [768*3 + 20*2]
                    g_is.append(g_i)

                g_is = torch.stack(g_is)  # [#span, 2344]
                fe_indices = (torch.tensor(self.frargmap[str(pred_sense.item())])).to(device)  # [#fe]
                n_fe = fe_indices.shape[0]
                # candidates를 가져와서 span과 scoring -> softmax
                # CELoss로 loss 계산.
                fe_embs = self.fe_embs(fe_indices)  # [#fe, 768]
                fe_embs = torch.stack([fe_embs] * n_span, dim=0)  # [#span, #fe, 768]

                g_is = torch.stack([g_is] * n_fe, dim=1)  # [#span, #fe, 2344]

                features = torch.cat((g_is, fe_embs), dim=2)  # [#span, #fe, 3112]
                pair_scores = self.pair_score(features)  # [#span, #fe, 1]
                pair_scores = torch.squeeze(pair_scores, dim=2)  # [#span, #fe]
                epsilon = (torch.zeros(n_span, 1)).to(device)  # [#span, 1]

                scores_with_epsilon = torch.cat((pair_scores, epsilon), dim=1)  # [span, fe+1]
                pred_frames.append(pred_sense)
                pred_scores.append(scores_with_epsilon)
                pred_span.append(pred_spans)

            return pred_frames, pred_span, pred_scores

class transfer_model(BertPreTrainedModel):
    def __init__(self, config, num_senses=2, num_args=2, frargmap=None, masking=True,
                 return_pooled_output=False, original_loss=False):
        super(transfer_model, self).__init__(config)

        ## TODO tgt 위치넣어서, distnace 및 trigger emb 추가.

        frame_size = config.hidden_size
        width_size = 20
        fe_size = config.hidden_size
        self.masking = masking
        self.num_senses = num_senses  # total number of all frames
        self.num_args = num_args  # total number of all frame elements
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.span_classifier = nn.Linear(config.hidden_size, 4)  # all FEs
        self.frargmap = frargmap  # mapping table for frame to its frame element candidates
        self.return_pooled_output = return_pooled_output
        self.width = Distance(width_size)
        self.original_loss = original_loss
        self.attention = Score(config.hidden_size)
        self.frame_embs = nn.Embedding(num_senses, frame_size)
        self.fe_embs = nn.Embedding(num_args+1, fe_size)  # 1285번째 idx는 None으로
        # self.fe_embs = nn.Embedding.from_pretrained(torch.load('/home/fairy_of_9/frameBERT/FE_embs.pt'))  # 1285번째 idx는 None으로

        self.init_weights()

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, lus=None, senses=None, args=None,
                using_gold_fame=False, position_ids=None, head_mask=None, gold_spans=None, span_pads=None, lu_spans=None, train=True):
        # token_type_ids : segment token indices [0,1] 0: sent A, 1: sent B
        sequence_output, pooled_output = self.bert(input_ids, token_type_ids=token_type_ids,
                                                   attention_mask=attention_mask)
        # seq_output : [bsz, seq_size(256), hidden_size], pooled_output : [bsz, hidden_size]
        sequence_output = self.dropout(sequence_output)
        pooled_output = self.dropout(pooled_output)
        attns = self.attention(sequence_output)
        loss = 0  # loss for sense id

        if train:  # is train
            for i in range(sequence_output.shape[0]):  # batch를 iter.
                attn = attns[i]
                seq_output = sequence_output[i]
                gold_sense = senses[i]  # [1]
                gold_span = gold_spans[i][:span_pads[i]]

                lu_span = lu_spans[i]  # [2]
                st,en = lu_span[0], lu_span[1]
                lu_attn, lu_emb = attn[st:en + 1], seq_output[st:en + 1]  # [lu_len,1 ], [lu_len, 768]
                lu_attn_weights = F.softmax(lu_attn, dim=1)  # [lu_len, 1]
                lu_attn_embed = torch.sum(torch.mul(lu_emb, lu_attn_weights), dim=0)  # [768]
                lu_emb = torch.stack((seq_output[st], seq_output[en], lu_attn_embed)) # [3, 768]

                g_is = []
                n_span = len(gold_span)
                gold_labels = []
                for st, en, label in gold_span:
                    gold_labels.append(label)
                    span_attn, span_emb = attn[st:en + 1], seq_output[st:en + 1]  # [span_len,1], [span_len, 768]
                    attn_weights = F.softmax(span_attn, dim=1)  # [span_len, 1]
                    attn_embed = torch.sum(torch.mul(span_emb, attn_weights), dim=0)  # [768]
                    # width = self.width([st - en + 1])  # [1, 20]
                    g_i = torch.stack((seq_output[st], seq_output[en], attn_embed))  # [3, 768]   score를 쓰려면 concat으로.
                    g_i = torch.cat((g_i, lu_emb), dim=0)
                    g_i = torch.sum(g_i, dim=0)  # [768]
                    g_is.append(g_i)

                g_is = torch.stack(g_is)  # [#span, 768*3]
                fe_indices = (torch.tensor(self.frargmap[str(gold_sense.item())])).to(device)  # [#fe]

                gold_idx = []
                for label in gold_labels:
                    for ii, jj in enumerate(self.frargmap[str(gold_sense.item())]):
                        if jj == label:
                            gold_idx.append(ii)

                assert len(gold_idx) == len(gold_labels)
                gold_idx = torch.tensor(gold_idx).to(device)

                n_fe = fe_indices.shape[0]
                # candidates를 가져와서 span과 scoring -> softmax
                # CELoss로 loss 계산.
                fe_embs = self.fe_embs(fe_indices)  # [#fe, 768]
                fe_embs = torch.stack([fe_embs] * n_span, dim=0)  # [#span, #fe, 768]
                g_is = torch.stack([g_is] * n_fe, dim=1)  # [#span, #fe, 768]

                cos = CosineSimilarity(dim=2)
                sims = cos(fe_embs, g_is)  # [#span, #fe]

                loss_fct_arg = CrossEntropyLoss()  # TODO trainset에서 제대로 weights 계산해보기?
                loss += loss_fct_arg(sims, gold_idx)  # [#span, #fe]와 [#span] 비교.

            loss = loss / sequence_output.shape[0]

            if self.return_pooled_output:
                return pooled_output, loss
            else:
                return loss
        else:  # is inference
            pred_labels = []
            for i in range(sequence_output.shape[0]):  # bsz, inference에선 1.
                attn = attns[i]
                seq_output = sequence_output[i]
                gold_sense = senses[i]  # [1]
                gold_span = gold_spans[i][:span_pads[i]]
                n_span = len(gold_span)

                lu_span = lu_spans[i]  # [2]
                st, en = lu_span[0], lu_span[1]
                lu_attn, lu_emb = attn[st:en + 1], seq_output[st:en + 1]  # [lu_len,1 ], [lu_len, 768]
                lu_attn_weights = F.softmax(lu_attn, dim=1)  # [lu_len, 1]
                lu_attn_embed = torch.sum(torch.mul(lu_emb, lu_attn_weights), dim=0)  # [768]
                lu_emb = torch.stack((seq_output[st], seq_output[en], lu_attn_embed))  # [3, 768]

                g_is = []
                pred_label = []
                for st, en, label in gold_span:
                    span_attn, span_emb = attn[st:en + 1], seq_output[st:en + 1]  # [span_len,1], [span_len, 768]
                    attn_weights = F.softmax(span_attn, dim=1)  # [span_len, 1]
                    attn_embed = torch.sum(torch.mul(span_emb, attn_weights), dim=0)  # [768]
                    # width = self.width([st - en + 1])  # [1, 20]
                    g_i = torch.stack((seq_output[st], seq_output[en], attn_embed))  # [3, 768]   score를 쓰려면 concat으로.
                    g_i = torch.cat((g_i, lu_emb), dim=0)
                    g_i = torch.sum(g_i, dim=0)  # [768]
                    g_is.append(g_i)

                g_is = torch.stack(g_is)  # [#span, 768*3]
                fe_indices = (torch.tensor(self.frargmap[str(gold_sense.item())])).to(device)  # [#fe]

                n_fe = fe_indices.shape[0]
                # candidates를 가져와서 span과 scoring -> softmax
                # CELoss로 loss 계산.
                fe_embs = self.fe_embs(fe_indices)  # [#fe, 768]
                fe_embs = torch.stack([fe_embs] * n_span, dim=0)  # [#span, #fe, 768]
                g_is = torch.stack([g_is] * n_fe, dim=1)  # [#span, #fe, 768]

                cos = CosineSimilarity(dim=2)
                sims = cos(fe_embs, g_is)  # [#span, #fe]
                pred_index = torch.argmax(sims, dim=1)  # [#span]
                for ii in pred_index:
                    pred_label.append(self.frargmap[str(gold_sense.item())][ii.item()])
                pred_labels.append(pred_label)
            return pred_labels

class transfer_spoken_model(BertPreTrainedModel):
    def __init__(self, config, num_senses=2, num_args=2, frargmap=None, masking=True,
                 return_pooled_output=False, original_loss=False, eval=False):
        super(transfer_spoken_model, self).__init__(config)
        frame_size = config.hidden_size
        fe_size = config.hidden_size
        self.masking = masking
        self.num_senses = num_senses  # total number of all frames
        self.num_args = num_args  # total number of all frame elements
        if eval:
            self.bert = BertModel(config)
            self.attention = Score(config.hidden_size)
            self.fe_embs = nn.Embedding(num_args+1, fe_size)
        else:
            self.bert = None
            self.attention = None
            self.fe_embs = None

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.frargmap = frargmap  # mapping table for frame to its frame element candidates
        self.return_pooled_output = return_pooled_output
        self.original_loss = original_loss

        # self.frame_embs = nn.Embedding(num_senses, frame_size)
        # self.fe_embs = nn.Embedding(num_args, fe_size)  # 1285번째 idx는 None으로

        # self.fe_embs = nn.Embedding.from_pretrained(torch.load('/home/fairy_of_9/frameBERT/FE_embs.pt'))  # 1285번째 idx는 None으로

        self.init_weights()

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, lu_spans=None, senses=None,
                position_ids=None, head_mask=None, gold_spans=None, lu_speakers=None, fe_speakers=None, train=True):
        # token_type_ids : segment token indices [0,1] 0: sent A, 1: sent B
        loss = 0  # loss for sense id

        if train:  # is train
            for i in range(input_ids.shape[0]):  # batch를 iter.
                pair_ids = input_ids[i]
                sequence_output, pooled_output = self.bert(pair_ids, token_type_ids=token_type_ids[i],
                                                           attention_mask=attention_mask[i])
                sequence_output = self.dropout(sequence_output)  # [2,256,768] lu-seq, arg-seq
                attns = self.attention(sequence_output)  # [2,256,1]

                gold_sense = senses[i]  # [1]
                gold_span = gold_spans[i]  # [4] utter_id, st, en, label
                lu_span = lu_spans[i]  # [3] utter_id, st, en
                st,en = lu_span[1], lu_span[2]
                if st.item() > 255 or en.item() > 255:
                    continue

                lu_attn, lu_emb = attns[0][st:en + 1], sequence_output[0][st:en + 1]  # [lu_len,1], [lu_len, 768]
                lu_attn_weights = F.softmax(lu_attn, dim=1)  # [lu_len, 1]
                lu_attn_embed = torch.sum(torch.mul(lu_emb, lu_attn_weights), dim=0)  # [768]
                lu_emb = torch.stack((sequence_output[0][st], sequence_output[0][en], lu_attn_embed)) # [3, 768]

                st,en = gold_span[1], gold_span[2]
                if st.item() > 255 or en.item() > 255:
                    continue
                span_attn, span_emb = attns[1][st:en + 1], sequence_output[1][st:en + 1]  # [span_len,1], [span_len, 768]
                attn_weights = F.softmax(span_attn, dim=1)  # [span_len, 1]
                attn_embed = torch.sum(torch.mul(span_emb, attn_weights), dim=0)  # [768]
                # width = self.width([st - en + 1])  # [1, 20]
                g_i = torch.stack((sequence_output[1][st], sequence_output[1][en], attn_embed))  # [3, 768]   score를 쓰려면 concat으로.
                g_i = torch.cat((g_i, lu_emb), dim=0)  # [6, 768]
                g_i = torch.sum(g_i, dim=0)  # [768]

                fe_indices = (torch.tensor(self.frargmap[str(gold_sense.item())])).to(device)  # [#fe]

                gold_label = gold_span[3]
                for ii, jj in enumerate(self.frargmap[str(gold_sense.item())]):
                    if jj == gold_label:
                        gold_idx = [ii]
                gold_idx = torch.tensor(gold_idx).to(device)

                n_fe = fe_indices.shape[0]
                # candidates를 가져와서 span과 scoring -> softmax
                # CELoss로 loss 계산.
                fe_embs = self.fe_embs(fe_indices)  # [#fe, 768]
                # fe_embs = torch.stack([fe_embs] * n_span, dim=0)  # [#span, #fe, 768]
                g_i = torch.stack([g_i] * n_fe, dim=0)  # [#fe, 768]

                cos = CosineSimilarity(dim=1)
                sims = cos(fe_embs, g_i)  # [#fe]
                sims = torch.unsqueeze(sims, dim=0)  # [1, #fe]
                loss_fct_arg = CrossEntropyLoss()  # TODO trainset에서 제대로 weights 계산해보기?
                loss += loss_fct_arg(sims, gold_idx)  # [1, #fe]와 [1] 비교.

            loss = loss / sequence_output.shape[0]

            if self.return_pooled_output:
                return pooled_output, loss
            else:
                return loss
        else:  # is inference
            pred_labels = []
            for i in range(input_ids.shape[0]):  # batch를 iter.
                pair_ids = input_ids[i]
                sequence_output, pooled_output = self.bert(pair_ids, token_type_ids=token_type_ids[i], attention_mask=attention_mask[i])
                sequence_output = self.dropout(sequence_output)  # [2,256,768] lu-seq, arg-seq
                attns = self.attention(sequence_output)  # [2,256,1]

                gold_sense = senses[i]  # [1]
                gold_span = gold_spans[i]  # [4] utter_id, st, en, label
                lu_span = lu_spans[i]  # [3] utter_id, st, en
                st,en = lu_span[1], lu_span[2]

                if st.item() > 255 or en.item() > 255:
                    return [-1]

                lu_attn, lu_emb = attns[0][st:en + 1], sequence_output[0][st:en + 1]  # [lu_len,1], [lu_len, 768]
                lu_attn_weights = F.softmax(lu_attn, dim=1)  # [lu_len, 1]
                lu_attn_embed = torch.sum(torch.mul(lu_emb, lu_attn_weights), dim=0)  # [768]
                lu_emb = torch.stack((sequence_output[0][st], sequence_output[0][en], lu_attn_embed)) # [3, 768]

                st,en = gold_span[1], gold_span[2]
                if st.item() > 255 or en.item() > 255:
                    return [-1]
                span_attn, span_emb = attns[1][st:en + 1], sequence_output[1][st:en + 1]  # [span_len,1], [span_len, 768]
                attn_weights = F.softmax(span_attn, dim=1)  # [span_len, 1]
                attn_embed = torch.sum(torch.mul(span_emb, attn_weights), dim=0)  # [768]
                # width = self.width([st - en + 1])  # [1, 20]
                g_i = torch.stack((sequence_output[1][st], sequence_output[1][en], attn_embed))  # [3, 768]   score를 쓰려면 concat으로.
                g_i = torch.cat((g_i, lu_emb), dim=0)  # [6, 768]
                g_i = torch.sum(g_i, dim=0)  # [768]

                fe_indices = (torch.tensor(self.frargmap[str(gold_sense.item())])).to(device)  # [#fe]

                gold_label = gold_span[3]
                for ii, jj in enumerate(self.frargmap[str(gold_sense.item())]):
                    if jj == gold_label:
                        gold_idx = [ii]
                gold_idx = torch.tensor(gold_idx).to(device)

                n_fe = fe_indices.shape[0]
                # candidates를 가져와서 span과 scoring -> softmax
                # CELoss로 loss 계산.
                fe_embs = self.fe_embs(fe_indices)  # [#fe, 768]
                # fe_embs = torch.stack([fe_embs] * n_span, dim=0)  # [#span, #fe, 768]
                g_i = torch.stack([g_i] * n_fe, dim=0)  # [#fe, 768]

                cos = CosineSimilarity(dim=1)
                sims = cos(fe_embs, g_i)  # [#fe]
                pred_index = torch.argmax(sims)  # [#span]
                pred_label = self.frargmap[str(gold_sense.item())][pred_index.item()]
                pred_labels.append(pred_label)
            return pred_labels
