from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from boltons.iterutils import windowed
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, CosineSimilarity, BCEWithLogitsLoss
from seqeval.metrics import *

import sys
sys.path.insert(0,'../')
sys.path.insert(0,'../../')
from src import dataio, thesis_utils
from src.ds import *
from torch.nn.parameter import Parameter
from transformers import BertModel, BertPreTrainedModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()

def to_cuda(x):
    """ GPU-enable a tensor """
    if torch.cuda.is_available():
        x = x.cuda()
    return x

class Distance(nn.Module):
    """ Learned, continuous representations for: span widths, distance
    between spans
    """

    bins = [-3,-2,-1,0,1,2,3]

    def __init__(self, distance_dim=20):
        super().__init__()

        self.dim = distance_dim
        self.embeds = nn.Sequential(
            nn.Embedding(len(self.bins)+1, distance_dim),
            nn.Dropout(0.20)
        )

    def forward(self, idx):
        """ Embedding table lookup """
        return self.embeds(self.stoi(*args))

    def stoi(self, lengths):
        """ Find which bin a number falls into """
        return to_cuda(torch.tensor([
            sum([True for i in self.bins if num >= i]) for num in lengths], requires_grad=False
        ))

class Pair_Score(nn.Module):
    """ binary classification for pair-utterances
        True  : pair-utter가 target utter, args utter의 pair 이다.
        False : 아니다.
    """
    def __init__(self, embeds_dim, hidden_dim=150):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(embeds_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(hidden_dim, 2)
        )
        weights = [1, 3.5]
        self.weights = torch.FloatTensor(weights).to(device)
        self.criterion = CrossEntropyLoss(self.weights)

    def forward(self, x, labels):
        """ x:[n_utter, 768*2], dist_emb:[n_utter, 20], sense_emb:[frame_dim] """
        loss = 0
        outputs = self.classifier(x)  # [n_utter, 768*3+20] -> [n_utter, 2]
        loss += self.criterion(outputs, labels)  # [n_span, n_fe]와 [n_span] 비교.
        return loss, outputs

class ffnn(nn.Module):
    """ Generic scoring module
    """
    def __init__(self, embeds_dim, hidden_dim=150):
        super().__init__()

        self.ffnn = nn.Sequential(
            nn.Linear(embeds_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(hidden_dim, 768*3)
        )

    def forward(self, x):
        """ Output a scalar score for an input x """
        return self.ffnn(x)

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

class bio_model(BertPreTrainedModel):
    def __init__(self, config, num_senses=2, num_lus=2, lufrmap=None, num_args=2, frargmap=None, masking=True, task="argument", data_type="written",
                 return_pooled_output=False, original_loss=False, eval=False, model_opts=None):
        super(bio_model, self).__init__(config)
        self.task = task
        self.data_type = data_type
        self.frame_dim = 150 # config.hidden_size
        self.fe_dim = 150 # config.hidden_size
        self.lu_dim = 150
        self.distance_dim = 20
        self.speaker_dim = 20
        self.masking = masking
        self.num_lus = num_lus
        self.num_senses = num_senses  # total number of all frames
        self.num_args = num_args  # total number of all frame elements
        self.k = 10

        '''
            fr_target_emb, 
            pair_dist, pair_sense, pair_speaker
            fe_target_emb, fe_dist, fe_speaker
        '''
        self.is_baseline = False
        self.use_sp_token = False
        for k, v in model_opts.items():
            setattr(self, k, v)

        sense_cls_dim = config.hidden_size
        if self.fr_target_emb:
            sense_cls_dim += config.hidden_size

        pair_dim = config.hidden_size * 2
        if self.pair_dist:
            pair_dim += self.distance_dim
        if self.pair_sense:
            pair_dim += self.frame_dim
        if self.pair_speaker:
            pair_dim += self.speaker_dim

        fe_dim = config.hidden_size
        if self.fe_target_emb:
            fe_dim += config.hidden_size
        if self.fe_sense:
            fe_dim += self.frame_dim
        if self.fe_lu:
            fe_dim += self.lu_dim
        if self.fe_dist:
            fe_dim += self.distance_dim
        if self.fe_speaker:
            fe_dim += self.speaker_dim


        self.pair_score = Pair_Score(pair_dim)
        self.mention_score = Score(3*config.hidden_size + self.frame_dim)
        self.role_score = Score(4*config.hidden_size)
        self.sr_score = Score(3*config.hidden_size + self.fe_dim)
        self.er_encoder = ffnn(self.frame_dim + self.fe_dim)
        self.linking_score = Score(6*config.hidden_size)

        self.fe_embs = nn.Embedding(num_args+1, self.fe_dim)

        if eval:
            self.bert = BertModel(config)
            self.attention = Score(config.hidden_size)
            self.frame_embs = nn.Embedding(num_senses, self.frame_dim)
            self.sense_classifier = nn.Linear(sense_cls_dim, num_senses)  # + self.lu_dim
            self.arg_classifier = nn.Linear(fe_dim, num_args)
            self.lu_encoder = nn.Embedding(self.num_lus, self.lu_dim)
            self.distance = nn.Embedding(13, self.distance_dim)  # -7 ~ +5
            self.speaker = nn.Embedding(2, self.speaker_dim)
        else:
            self.bert = None
            self.attention = None
            self.frame_embs = None
            self.sense_classifier = None
            self.arg_classifier = None
            self.lu_encoder = None
            self.distance = None
            self.speaker = None

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.lufrmap = lufrmap
        self.frargmap = frargmap  # mapping table for frame to its frame element candidates
        self.return_pooled_output = return_pooled_output
        self.original_loss = original_loss

        # self.fe_embs = nn.Embedding.from_pretrained(torch.load('/home/fairy_of_9/frameBERT/FE_embs.pt'))  # 1285번째 idx는 None으로

        self.init_weights()

    def boundary_filter(self, i):
        if i <= -7:
            return -7
        if i >= 5:
            return 5
        return i

    def span_representation(self, emb, attn, st, en):

        span_attn, span_emb = attn[st:en + 1], emb[st:en + 1]  # [lu_len,1], [lu_len, 768]
        attn_weights = F.softmax(span_attn, dim=1)  # [lu_len, 1]
        attn_embed = torch.sum(torch.mul(span_emb, attn_weights), dim=0)  # [768]
        result = torch.stack((emb[st], emb[en], attn_embed))
        result = torch.sum(result, dim=0)
        return result

    def forward(self, input_ids=None, utter_lens=None, orig_tok_to_maps=None, token_type_ids=None, attention_mask=None,
                targets=None, senses=None, position_ids=None, head_mask=None, gold_args=None, arg_lens=None, bios=None,
                lus=None,
                speakers=None, speaker_lens=None, train=True, special_tokens=None, use_pruning=True):
        ''' token_type_ids : segment token indices [0,1] 0: sent A, 1: sent B '''

        frame_loss, pair_loss, linking_loss = 0, 0, 0
        pred_utter = None

        for i in range(input_ids.shape[0]):  # batch를 iter.
            utter_len, arg_len, speaker_len = utter_lens[i], arg_lens[i], speaker_lens[i]
            scene_bios = bios[i]  # [n_utter, 256]
            gold_lu = lus[i]
            gold_sense = senses[i]  # [1]
            sense_emb = self.frame_embs(gold_sense)  # [frame_dim]
            gold_spans = gold_args[i][:arg_len]  # [4] utter_id, st, en, label
            target = targets[i]  # [3] utter_id, st, en
            scene_ids, scene_speakers, special_token = input_ids[i][:utter_len], speakers[i][:utter_len], \
                                                       special_tokens[i][:utter_len]
            scene_maps = orig_tok_to_maps[i][:utter_len]
            lu_emb = self.lu_encoder(gold_lu)

            scene_token_type_ids = token_type_ids[i][:utter_len]
            scene_attention_mask = attention_mask[i][:utter_len]

            sequence_output, pooled_output = self.bert(scene_ids, token_type_ids=scene_token_type_ids,
                                                       attention_mask=scene_attention_mask)  # [n_utter, 256, 768], [n_utter, 768]

            sequence_output = self.dropout(sequence_output)  # [n_utter,256,768]
            pooled_output = self.dropout(pooled_output)
            attns = self.attention(sequence_output)  # [2,256,1]

            target_uid = target[0]
            args_uid = gold_spans[:, 0]
            unique_args_uid = torch.unique(args_uid)  # FE가 존재하는 uids
            if self.task == 'sentence':
                unique_args_uid = torch.LongTensor([target_uid.item()]).to(device)


            st, en = target[1], target[2]
            if st.item() > 255 or en.item() > 255:
                continue

            target_emb = self.span_representation(sequence_output[target_uid], attns[target_uid], st,
                                                  en)  # sense_emb, dist_emb, speaker_emb
            target_embs = torch.stack(([target_emb] * utter_len.item()), dim=0)  # [n_utter, dim]
            target_embs = torch.stack(([target_embs] * 256), dim=1)  # [n_utter, 256, dim]

            target_cls = pooled_output[target_uid]  # [768]

            ''' Frame Identification '''
            if self.task == 'full' or self.task == 'sentence':
                if self.fr_target_emb:
                    sense_logit = self.sense_classifier(torch.cat((target_cls, target_emb), dim=0))  # target_emb or target_cls  torch.cat((target_cls, lu_emb), dim=0)
                else:
                    if self.fr_context != False:
                        lens = list(map(torch.sum, scene_attention_mask))
                        tgt_ids = []
                        target_pos = target[1]

                        if isinstance(self.fr_context, int):  # watching context using window size
                            st = max(target_uid - self.fr_context, 0)
                            en = target_uid + 1
                            for ii in range(st, en):
                                tgt_ids.append(scene_ids[ii][:int(lens[ii].item())])
                                tgt_ids.append(torch.tensor([100]).to(device))

                            context_ids = torch.cat(tgt_ids, dim=0)
                            if context_ids.shape[0] > 256:
                                context_ids = context_ids[context_ids.shape[0]-256:]
                        else:  # context를 target 앞뒤로 많이 보기.
                            for ii in range(len(lens)):
                                tgt_ids.append(scene_ids[ii][:int(lens[ii].item())])
                                tgt_ids.append(torch.tensor([100]).to(device))
                                if ii < target_uid:
                                    target_pos += int(lens[ii].item()) + 1
                            context_st, context_en = max(target_pos - 127, 0), min(target_pos + 127, 255)
                            context_ids = torch.cat(tgt_ids, dim=0)
                            context_en = min(context_en, context_ids.shape[0])
                            context_ids = context_ids[context_st: context_en]

                        context_token_type_ids = torch.ones([256], dtype=torch.int64)
                        context_token_type_ids[:context_ids.shape[0]] = 0
                        context_attention_mask = torch.zeros([256], dtype=torch.float32)
                        context_attention_mask[:context_ids.shape[0]] = 1
                        context_ids_with_pad = torch.zeros([256], dtype=torch.int64).to(device)
                        context_ids_with_pad[:len(context_ids)] = context_ids

                        context_ids_with_pad = torch.unsqueeze(context_ids_with_pad, dim=0).to(device)
                        context_token_type_ids = torch.unsqueeze(context_token_type_ids, dim=0).to(device)
                        context_attention_mask = torch.unsqueeze(context_attention_mask, dim=0).to(device)

                        _, target_cls = self.bert(context_ids_with_pad, token_type_ids=context_token_type_ids, attention_mask=context_attention_mask)
                        target_cls = self.dropout(target_cls)
                    sense_logit = self.sense_classifier(target_cls)

                lufr_mask = thesis_utils.get_masks([gold_lu], self.lufrmap, num_label=self.num_senses, masking=True).to(
                    device)[0]
                loss_fct_sense = CrossEntropyLoss(weight=lufr_mask)
                frame_loss += loss_fct_sense(sense_logit.view(-1, self.num_senses), gold_sense.view(-1))
                inf = float('inf')
                sense_logit = torch.mul(sense_logit, lufr_mask)
                sense_logit[sense_logit == 0] = -inf
                pred_frame = torch.argmax(sense_logit)  # [256]
                sense_emb = self.frame_embs(pred_frame)

            target_cls = pooled_output[target_uid]
            if self.task != '4':
                ''' utterance-pair classification '''
                pair_clses = torch.cat((torch.stack([target_cls] * utter_len.item()), pooled_output),
                                       dim=1)  # [n_utter, 768*2]
                pair_labels = torch.zeros(utter_len.item(), dtype=torch.long)  # [n_utter]
                pair_labels[unique_args_uid] = 1  # [#utter]
                pair_labels = pair_labels.cuda()
                is_same_spaker = (scene_speakers[target_uid] == scene_speakers).to(device).long()
                speaker_emb = self.speaker(is_same_spaker)

                dist = torch.LongTensor([self.boundary_filter(i - target_uid) + 7 for i in
                                         range(utter_len)]).cuda()  # -7 ~ 5 의 값으로 나오기 때문에 +7을해서 0 ~ 12로  [n_utter]
                dist_emb = self.distance(dist)  # [n_utter, 20]

                sense_embs = torch.stack([sense_emb] * dist_emb.shape[0])  # [n_utter, 768]
                features = pair_clses
                if self.pair_sense:
                    features = torch.cat((features, sense_embs), dim=1)
                if self.pair_dist:
                    features = torch.cat((features, dist_emb), dim=1)
                if self.pair_speaker:
                    features = torch.cat((features, speaker_emb), dim=1)

                cur_pair_loss, pair_output = self.pair_score(features, pair_labels)
                if self.data_type != "written" and self.task != "sentence":
                    pair_loss += cur_pair_loss
                # pair_output : [#utter, 2]  0 == fe가 존재하지 않음, 1 == fe가 존재

                if not train:
                    pred_pair = torch.argmax(pair_output, dim=1)
                    unique_args_uid = (pred_pair > 0).nonzero().flatten()
                    if self.data_type == "written":
                        unique_args_uid = torch.LongTensor([0]).to(device)

                    if self.is_baseline:
                        unique_args_uid = torch.LongTensor(list(range(utter_len))).to(device)

                    # if unique_args_uid.shape[0] != 0:
                    #     print()
                    pred_utter = unique_args_uid



            sense_embs = torch.stack(([sense_emb] * utter_len.item()), dim=0)
            sense_embs = torch.stack(([sense_embs] * 256), dim=1)
            dist_embs = torch.stack(([dist_emb] * 256), dim=1)
            speaker_embs = torch.stack(([speaker_emb] * 256), dim=1)

            lu_embs = torch.stack(([lu_emb] * utter_len.item()), dim=0)
            lu_embs = torch.stack(([lu_embs] * 256), dim=1)

            features = sequence_output
            if self.fe_target_emb:
                features = torch.cat((features, target_embs), dim=2)
            if self.fe_sense:
                features = torch.cat((features, sense_embs), dim=2)
            if self.fe_lu:
                features = torch.cat((features, lu_embs), dim=2)
            if self.fe_dist:
                features = torch.cat((features, dist_embs), dim=2)
            if self.fe_speaker:
                features = torch.cat((features, speaker_embs), dim=2)

            arg_logits = self.arg_classifier(features)  # [n_utter, 256, 2572]

            if self.task == 'full' or self.task == 'sentence':
                frarg_mask = \
                thesis_utils.get_masks([pred_frame], self.frargmap, num_label=self.num_args, masking=True).to(device)[0]
            else:
                frarg_mask = \
                    thesis_utils.get_masks([gold_sense], self.frargmap, num_label=self.num_args, masking=True).to(
                        device)[0]

            # TODO self.task == '4' 일때를.. 이아래서 mask를 잘주면 해결이 될 거 같음.
            if not train:
                inf = float('inf')
                arg_logits = torch.mul(arg_logits, frarg_mask)
                arg_logits[arg_logits == 0] = -inf

                pred_labels = []
                for ii, arg_logit in enumerate(arg_logits):  # arg_logit = [256,2572]
                    temp = torch.argmax(arg_logit, dim=1)  # [256]
                    # if self.task == 'sentence':
                    #     if ii == target_uid:
                    #         pred_labels.append(temp)
                    #     else:
                    #         pred_labels.append([]) # (torch.ones(256)).to(device)
                    # else:
                    pred_labels.append(temp)
            else:
                ''' FE span extractor '''
                loss_fct_arg = CrossEntropyLoss(weight=frarg_mask)

                for u_idx, uid in enumerate(unique_args_uid):
                    cur_dist_emb = dist_emb[uid]
                    cur_speaker_emb = speaker_emb[uid]
                    utter_embs = sequence_output[uid]
                    utter_attns = attns[uid]
                    utter_mask = scene_attention_mask[uid]  # [256]
                    arg_logit = arg_logits[uid]  # [2572]
                    bio = scene_bios[uid]  # [256]

                    active_loss = utter_mask.view(-1) == 1  # [256]
                    active_logits = arg_logit.view(-1, self.num_args)[active_loss]
                    active_labels = bio.view(-1)[active_loss]
                    linking_loss += loss_fct_arg(active_logits, active_labels)


        if train:
            frame_loss, pair_loss, linking_loss = frame_loss / input_ids.shape[0], pair_loss / input_ids.shape[
                0], linking_loss / input_ids.shape[0]
            return frame_loss, pair_loss, linking_loss
        else:
            return pred_frame, pred_utter, pred_labels

    def compute_idx_spans(self, max_len, L=5):
        """ Compute span indexes for all possible spans up to length L in each
        sentence """

        def flatten(alist):
            """ Flatten a list of lists into one list """
            return [item for sublist in alist for item in sublist]

        idx_spans, shift = [], 0
        sent_spans = flatten([windowed(range(shift, max_len + shift), length)
                              for length in range(1, L)])

        return sent_spans

    def extract_span(self, ids, special_tokens, maps):
        """
        output: [#ids, #spans]
                list of list
                ids에 해당하는 id에서의 span candidates

        """
        span_candidates = []
        for i, id in enumerate(ids):
            tokens = special_tokens[id]
            map = maps[id]
            max_len = tokens[-1]
            invalid = set([0])
            span_candidate = [(id, t[0], t[-1]) for sid, t in enumerate(self.compute_idx_spans(max_len))]
            del_list = set()

            start_valid = [int(x) for x in map if x != -1]
            end_valid = [int(x) - 1 for x in start_valid[1:]] + [int(max_len) - 1]
            for ii, candidate in enumerate(span_candidate):
                if len(candidate) == 1:
                    indices = set([candidate[1]])
                else:
                    indices = set([x for x in range(candidate[1], candidate[2] + 1)])

                if len(indices.intersection(invalid)) > 0:
                    del_list.add(ii)
                elif candidate[1] not in start_valid:
                    del_list.add(ii)
                elif candidate[2] not in end_valid:
                    del_list.add(ii)
                elif candidate[1] > 255:
                    del_list.add(ii)
                elif candidate[2] > 255:
                    del_list.add(ii)
            del_list = list(del_list)
            del_list.sort()
            del_list.reverse()
            for ii in del_list:
                del span_candidate[ii]
            span_candidates.append(torch.LongTensor(span_candidate))


        return span_candidates
