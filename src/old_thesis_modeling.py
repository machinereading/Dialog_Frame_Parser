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

    def forward(self, x, labels, dist_emb, sense_emb, speaker_emb):
        """ x:[n_utter, 768*2], dist_emb:[n_utter, 20], sense_emb:[frame_dim] """
        loss = 0
        sense_embs = torch.stack([sense_emb]*dist_emb.shape[0])  # [n_utter, 768]
        x = torch.cat((x, dist_emb, speaker_emb, sense_embs), dim=1)  # [n_utter, 768*3+20]
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

class span_extract_model(BertPreTrainedModel):
    def __init__(self, config, num_senses=2, num_args=2, frargmap=None, masking=True,
                 return_pooled_output=False, original_loss=False, eval=False, use_gi_variation=False, use_target_emb=False):
        super(span_extract_model, self).__init__(config)
        self.frame_dim = 150 # config.hidden_size
        self.fe_dim = 150 # config.hidden_size
        self.distance_dim = 20
        self.speaker_dim = 20
        self.masking = masking
        self.num_senses = num_senses  # total number of all frames
        self.num_args = num_args  # total number of all frame elements
        self.k = 10
        self.use_gi_variation = use_gi_variation


        self.mention_score = Score(2*config.hidden_size + self.frame_dim + self.distance_dim + self.speaker_dim)
        self.pair_score = Pair_Score(config.hidden_size * 2 + self.frame_dim + self.distance_dim + self.speaker_dim)
        self.role_score = Score(4*config.hidden_size)
        self.sr_score = Score(3*config.hidden_size + self.fe_dim)
        self.er_encoder = ffnn(self.frame_dim + self.fe_dim)
        self.linking_score = Score(6*config.hidden_size)
        self.distance = nn.Embedding(13, self.distance_dim)  # -7 ~ +5
        self.speaker = nn.Embedding(2, self.speaker_dim)

        if eval:
            self.bert = BertModel(config)
            self.attention = Score(config.hidden_size)
            self.frame_embs = nn.Embedding(num_senses, self.frame_dim)
            self.fe_embs = nn.Embedding(num_args+1, self.fe_dim)
        else:
            self.bert = None
            self.attention = None
            self.fe_embs = None

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
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
                targets=None, senses=None, position_ids=None, head_mask=None, gold_args=None, arg_lens=None,
                speakers=None, speaker_lens=None, train=True, special_tokens=None, use_gold_span=True, use_pruning=True, use_gi_variation=False):
        ''' token_type_ids : segment token indices [0,1] 0: sent A, 1: sent B '''

        loss = 0  # loss for sense id
        span_loss, pair_loss, linking_loss = 0, 0, 0

        if train:  # is train
            for i in range(input_ids.shape[0]):  # batch를 iter.
                utter_len, arg_len, speaker_len = utter_lens[i], arg_lens[i], speaker_lens[i]
                gold_sense = senses[i]  # [1]
                sense_emb = self.frame_embs(gold_sense)  # [frame_dim]
                gold_spans = gold_args[i][:arg_len]  # [4] utter_id, st, en, label
                target = targets[i]  # [3] utter_id, st, en
                scene_ids, scene_speakers, special_token = input_ids[i][:utter_len], speakers[i][:utter_len], special_tokens[i][:utter_len]
                scene_maps = orig_tok_to_maps[i][:utter_len]

                scene_token_type_ids = token_type_ids[i][:utter_len]
                scene_attention_mask = attention_mask[i][:utter_len]

                sequence_output, pooled_output = self.bert(scene_ids, token_type_ids=scene_token_type_ids,
                                                           attention_mask=scene_attention_mask)  # [n_utter, 256, 768], [n_utter, 768]

                sequence_output = self.dropout(sequence_output)
                pooled_output = self.dropout(pooled_output)


                target_uid = target[0]
                args_uid = gold_spans[:,0]
                unique_args_uid = torch.unique(args_uid)  # FE가 존재하는 uids

                ''' utterance-pair classification '''
                target_cls = pooled_output[target_uid]  # [768]
                pair_clses = torch.cat((torch.stack([target_cls] * utter_len.item()), pooled_output), dim=1)  # [n_utter, 768*2]
                pair_labels = torch.zeros(utter_len.item(), dtype=torch.long)  # [n_utter]
                pair_labels[unique_args_uid] = 1  # [#utter]
                pair_labels = pair_labels.cuda()
                is_same_spaker = (speakers[i][target_uid] == speakers[i]).to(device).long()
                speaker_emb = self.speaker(is_same_spaker)

                # TODO: speaker emb도 추가해야함. 같은지 다른지.
                dist = torch.LongTensor([self.boundary_filter(i - target_uid) + 7 for i in range(utter_len)]).cuda()  # -7 ~ 5 의 값으로 나오기 때문에 +7을해서 0 ~ 12로  [n_utter]
                dist_emb = self.distance(dist)  # [n_utter, 20]

                cur_pair_loss, pair_output = self.pair_score(pair_clses, pair_labels, dist_emb, sense_emb, speaker_emb)
                pair_loss += cur_pair_loss
                # pair_output : [#utter, 2]  0 == fe가 존재하지 않음, 1 == fe가 존재

                attns = self.attention(sequence_output)  # [2,256,1]

                st,en = target[1], target[2]
                if st.item() > 255 or en.item() > 255:
                    continue

                lu_emb = self.span_representation(sequence_output[target_uid], attns[target_uid], st, en)

                ''' FE span extractor '''
                frarg_mask = thesis_utils.get_masks([gold_sense], self.frargmap, num_label=self.num_args+1, masking=True).to(device)[0]

                span_candidates = self.extract_span(unique_args_uid, special_token, scene_maps)
                gold_span_index = gold_spans[:,:-1]
                gold_span_label = gold_spans[:,-1]

                for u_idx, uid in enumerate(unique_args_uid):
                    cur_dist_emb = dist_emb[uid]
                    cur_speaker_emb = speaker_emb[uid]
                    utter_embs = sequence_output[uid]
                    utter_attns = attns[uid]
                    spans = span_candidates[u_idx].to(device)
                    n_span = len(spans)
                    g_is = []
                    gold_labels = []

                    if n_span == 0:
                        continue

                    for span in spans:
                        span_uid, st, en = span

                        g_i = self.span_representation(utter_embs, utter_attns, st, en)
                        g_i = torch.cat((g_i, lu_emb, cur_dist_emb, cur_speaker_emb), dim=0)
                        g_is.append(g_i)

                        gold_index = ((torch.sum((gold_span_index == span), dim=1) == 3).nonzero()).flatten()  # gold_span_index 에서 현재 span과 같은 애를 찾고 index return
                        if gold_index.shape[0] == 1: # current span is gold
                            gold_labels.append(gold_span_label[gold_index].item())
                        else:  # current span isn't gold
                            gold_labels.append(-1)

                    gold_labels = torch.LongTensor(gold_labels).to(device)
                    g_is = torch.stack(g_is)
                    if use_pruning:
                        sense_embs = torch.stack([sense_emb] * n_span, dim=0)
                        scores = self.mention_score(torch.cat((g_is, sense_embs),dim=1)) # [n_span, 3072] -> [n_span, 1]
                        scores = scores.flatten()
                        is_gold = (gold_labels > -1).float()
                        probs = F.softmax(scores)
                        eps = 1e-8
                        # Negative marginal log-likelihood
                        span_loss += torch.log(torch.sum(torch.mul(probs, is_gold)).clamp_(eps, 1 - eps)) * -1

            span_loss, pair_loss, linking_loss = span_loss / input_ids.shape[0], pair_loss / input_ids.shape[0], linking_loss / input_ids.shape[0]

            if type(loss) == float:
                print()

            return span_loss, pair_loss, linking_loss

        else:  # is inference
            for i in range(input_ids.shape[0]):  # batch를 iter.
                utter_len, speaker_len = utter_lens[i], speaker_lens[i]
                pred_utter = None

                gold_sense = senses[i]  # [1]
                sense_emb = self.frame_embs(gold_sense)  # [768]
                target = targets[i]  # [3] utter_id, st, en
                target_uid = target[0]
                scene_ids, scene_speakers, special_token = input_ids[i][:utter_len], speakers[i][:utter_len], \
                                                           special_tokens[i][:utter_len]
                scene_maps = orig_tok_to_maps[i][:utter_len]

                top_spans = []  # scene에서 추출한 spans
                top_spans_label = []
                top_spans_gis = []
                top_spans_scores = []

                scene_token_type_ids = token_type_ids[i][:utter_len]
                scene_attention_mask = attention_mask[i][:utter_len]

                sequence_output, pooled_output = self.bert(scene_ids, token_type_ids=scene_token_type_ids,
                                                           attention_mask=scene_attention_mask)  # [n_utter, 256, 768], [n_utter, 768]
                ''' utterance-pair classification '''
                target_cls = pooled_output[target_uid]  # [768]
                pair_clses = torch.cat((torch.stack([target_cls] * utter_len.item()), pooled_output),
                                       dim=1)  # [n_utter, 768*2]
                pair_labels = torch.zeros(utter_len.item(), dtype=torch.long)  # [n_utter]
                pair_labels = pair_labels.cuda()
                is_same_spaker = (speakers[i][target_uid] == speakers[i]).to(device).long()
                speaker_emb = self.speaker(is_same_spaker)

                # TODO: speaker emb도 추가해야함. 같은지 다른지.
                dist = torch.LongTensor([self.boundary_filter(i - target_uid) + 7 for i in
                                         range(utter_len)]).cuda()  # -7 ~ 5 의 값으로 나오기 때문에 +7을해서 0 ~ 12로  [n_utter]
                dist_emb = self.distance(dist)  # [n_utter, 20]

                cur_pair_loss, pair_output = self.pair_score(pair_clses, pair_labels, dist_emb, sense_emb, speaker_emb)
                # pair_output : [#utter, 2]  0 == fe가 존재하지 않음, 1 == fe가 존재
                pred_pair = torch.argmax(pair_output, dim=1)
                unique_args_uid = (pred_pair > 0).nonzero().flatten()
                pred_utter = unique_args_uid

                if pred_utter.shape[0] == 0:  # 비었으면,
                    return None, None, None

                attns = self.attention(sequence_output)  # [2,256,1]

                st, en = target[1], target[2]
                if st.item() > 255 or en.item() > 255:
                    continue

                lu_emb = self.span_representation(sequence_output[target_uid], attns[target_uid], st, en)

                ''' FE span extractor '''
                span_candidates = self.extract_span(unique_args_uid, special_token, scene_maps)
                for u_idx, uid in enumerate(unique_args_uid):
                    cur_dist_emb = dist_emb[uid]
                    cur_speaker_emb = speaker_emb[uid]
                    utter_embs = sequence_output[uid]
                    utter_attns = attns[uid]
                    spans = span_candidates[u_idx].to(device)
                    n_span = len(spans)
                    g_is = []

                    if n_span == 0:
                        continue

                    for span in spans:
                        span_uid, st, en = span

                        if self.use_gi_variation:
                            g_i = self.span_representation(utter_embs, utter_attns, st, en)
                            g_i = torch.cat((g_i, lu_emb, cur_dist_emb, cur_speaker_emb), dim=0)
                        else:
                            span_attn, span_emb = utter_attns[st:en + 1], utter_embs[
                                                                          st:en + 1]  # [span_len,1], [span_len, 768]
                            attn_weights = F.softmax(span_attn, dim=1)  # [span_len, 1]
                            attn_embed = torch.sum(torch.mul(span_emb, attn_weights), dim=0)  # [768]
                            g_i = torch.cat((utter_embs[st], utter_embs[en], attn_embed, lu_emb), dim=0)  # [768*3]
                        g_is.append(g_i)
                        # g_i_lu = torch.cat((g_i, lu_emb), dim=0)  # [768*6]

                    g_is = torch.stack(g_is)

                    if use_pruning:
                        sense_embs = torch.stack([sense_emb] * n_span, dim=0)
                        scores = self.mention_score(torch.cat((g_is, sense_embs),dim=1)) # [n_span, 3072] -> [n_span, 1]
                        scores = scores.flatten()
                        top_spans_indices = torch.topk(scores.flatten(), min(self.k, spans.shape[
                            0])).indices  # top_spans.values, top_spans.indices
                        top_spans.append(spans[top_spans_indices])  # [k, 3]
                        top_spans_gis.append(g_is[top_spans_indices])

                def list_to_tensor(input):
                    tensors = None
                    for item in input:
                        if tensors is None:
                            tensors = item
                        else:
                            tensors = torch.cat((tensors, item))
                    return tensors

                top_spans = list_to_tensor(top_spans)  # [#unique_args_uid * k(가변), 3]
                top_spans_gis = list_to_tensor(top_spans_gis)  # [#unique_args_uid * k, 2304]
                pred_top_spans = top_spans

                return pred_utter, pred_top_spans, None

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

class thesis_spoken_model(BertPreTrainedModel):
    def __init__(self, config, num_senses=2, num_args=2, frargmap=None, masking=True,
                 return_pooled_output=False, original_loss=False, eval=False, use_gi_variation=False):
        super(thesis_spoken_model, self).__init__(config)
        self.frame_dim = 150 # config.hidden_size
        self.fe_dim = 150 # config.hidden_size
        self.distance_dim = 20
        self.masking = masking
        self.num_senses = num_senses  # total number of all frames
        self.num_args = num_args  # total number of all frame elements
        self.k = 10
        self.use_gi_variation = use_gi_variation

        self.mention_score = Score(3*config.hidden_size + self.frame_dim)
        self.pair_score = Pair_Score(config.hidden_size * 2 + self.frame_dim + self.distance_dim)
        self.role_score = Score(4*config.hidden_size)
        self.sr_score = Score(3*config.hidden_size + self.fe_dim)
        self.er_encoder = ffnn(self.frame_dim + self.fe_dim)
        self.linking_score = Score(6*config.hidden_size)
        self.distance = nn.Embedding(13, self.distance_dim)  # -7 ~ +5

        if use_gi_variation:
            self.fe_classifier = nn.Linear(2 * config.hidden_size + self.frame_dim + self.distance_dim, num_args + 1)
        else:
            self.fe_classifier = nn.Linear(3 * config.hidden_size + self.frame_dim + self.distance_dim, num_args + 1)

        if eval:
            self.bert = BertModel(config)
            self.attention = Score(config.hidden_size)
            self.frame_embs = nn.Embedding(num_senses, self.frame_dim)
            self.fe_embs = nn.Embedding(num_args+1, self.fe_dim)
        else:
            self.bert = None
            self.attention = None
            self.fe_embs = None

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
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
                targets=None, senses=None, position_ids=None, head_mask=None, gold_args=None, arg_lens=None,
                speakers=None, speaker_lens=None, train=True, special_tokens=None, use_gold_span=True, use_pruning=True, use_gi_variation=False):
        ''' token_type_ids : segment token indices [0,1] 0: sent A, 1: sent B '''

        loss = 0  # loss for sense id
        span_loss, pair_loss, linking_loss = 0, 0, 0

        if train:  # is train
            for i in range(input_ids.shape[0]):  # batch를 iter.
                utter_len, arg_len, speaker_len = utter_lens[i], arg_lens[i], speaker_lens[i]
                gold_sense = senses[i]  # [1]
                sense_emb = self.frame_embs(gold_sense)  # [frame_dim]
                gold_spans = gold_args[i][:arg_len]  # [4] utter_id, st, en, label
                target = targets[i]  # [3] utter_id, st, en
                scene_ids, scene_speakers, special_token = input_ids[i][:utter_len], speakers[i][:utter_len], special_tokens[i][:utter_len]
                scene_maps = orig_tok_to_maps[i][:utter_len]

                top_spans = []  # scene에서 추출한 spans
                top_spans_label = []
                top_spans_gis = []
                top_spans_scores = []

                scene_token_type_ids = token_type_ids[i][:utter_len]
                scene_attention_mask = attention_mask[i][:utter_len]

                sequence_output, pooled_output = self.bert(scene_ids, token_type_ids=scene_token_type_ids,
                                                           attention_mask=scene_attention_mask)  # [n_utter, 256, 768], [n_utter, 768]

                sequence_output = self.dropout(sequence_output)
                pooled_output = self.dropout(pooled_output)


                target_uid = target[0]
                args_uid = gold_spans[:,0]
                unique_args_uid = torch.unique(args_uid)  # FE가 존재하는 uids

                if not use_gold_span:
                    ''' utterance-pair classification '''
                    target_cls = pooled_output[target_uid]  # [768]
                    pair_clses = torch.cat((torch.stack([target_cls] * utter_len.item()), pooled_output), dim=1)  # [n_utter, 768*2]
                    pair_labels = torch.zeros(utter_len.item(), dtype=torch.long)  # [n_utter]
                    pair_labels[unique_args_uid] = 1  # [#utter]
                    pair_labels = pair_labels.cuda()

                    # TODO: speaker emb도 추가해야함. 같은지 다른지.
                    dist = torch.LongTensor([self.boundary_filter(i - target_uid) + 7 for i in range(utter_len)]).cuda()  # -7 ~ 5 의 값으로 나오기 때문에 +7을해서 0 ~ 12로  [n_utter]
                    dist_emb = self.distance(dist)  # [n_utter, 20]

                    cur_pair_loss, pair_output = self.pair_score(pair_clses, pair_labels, dist_emb, sense_emb)
                    pair_loss += cur_pair_loss
                    # pair_output : [#utter, 2]  0 == fe가 존재하지 않음, 1 == fe가 존재

                attns = self.attention(sequence_output)  # [2,256,1]

                st,en = target[1], target[2]
                if st.item() > 255 or en.item() > 255:
                    continue

                lu_emb = self.span_representation(sequence_output[target_uid], attns[target_uid], st, en)

                ''' FE span extractor '''
                frarg_mask = thesis_utils.get_masks([gold_sense], self.frargmap, num_label=self.num_args+1, masking=True).to(device)[0]
                loss_fct_arg = CrossEntropyLoss(weight=frarg_mask)

                if use_gold_span:
                    top_spans = gold_spans[:,:-1]
                    top_spans_label = gold_spans[:,-1]
                    top_spans_gis = []
                    for span in top_spans:
                        span_uid, st, en = span
                        utter_embs = sequence_output[span_uid]
                        utter_attns = attns[span_uid]

                        if self.use_gi_variation:
                            g_i = self.span_representation(utter_embs, utter_attns, st, en)
                            g_i = torch.cat((g_i, lu_emb), dim=0)

                        else:
                            span_attn, span_emb = utter_attns[st:en + 1], utter_embs[
                                                                          st:en + 1]  # [span_len,1], [span_len, 768]
                            attn_weights = F.softmax(span_attn, dim=1)  # [span_len, 1]
                            attn_embed = torch.sum(torch.mul(span_emb, attn_weights), dim=0)  # [768]
                            g_i = torch.cat((utter_embs[st], utter_embs[en], attn_embed, lu_emb), dim=0)  # [768*3]

                        top_spans_gis.append(g_i)
                    top_spans_gis = torch.stack(top_spans_gis)
                else:
                    span_candidates = self.extract_span(unique_args_uid, special_token, scene_maps)
                    gold_span_index = gold_spans[:,:-1]
                    gold_span_label = gold_spans[:,-1]

                    for u_idx, uid in enumerate(unique_args_uid):
                        utter_embs = sequence_output[uid]
                        utter_attns = attns[uid]
                        spans = span_candidates[u_idx].to(device)
                        n_span = len(spans)
                        g_is = []
                        gold_labels = []

                        if n_span == 0:
                            continue

                        for span in spans:
                            span_uid, st, en = span

                            if self.use_gi_variation:
                                g_i = self.span_representation(utter_embs, utter_attns, st, en)
                                g_i = torch.cat((g_i, lu_emb), dim=0)
                            else:
                                span_attn, span_emb = utter_attns[st:en + 1], utter_embs[
                                                                              st:en + 1]  # [span_len,1], [span_len, 768]
                                attn_weights = F.softmax(span_attn, dim=1)  # [span_len, 1]
                                attn_embed = torch.sum(torch.mul(span_emb, attn_weights), dim=0)  # [768]
                                g_i = torch.cat((utter_embs[st], utter_embs[en], attn_embed, lu_emb), dim=0)  # [768*3]
                            g_is.append(g_i)
                            # g_i_lu = torch.cat((g_i, lu_emb), dim=0)  # [768*6]

                            gold_index = ((torch.sum((gold_span_index == span), dim=1) == 3).nonzero()).flatten()  # gold_span_index 에서 현재 span과 같은 애를 찾고 index return
                            if gold_index.shape[0] == 1: # current span is gold
                                gold_labels.append(gold_span_label[gold_index].item())
                            else:  # current span isn't gold
                                gold_labels.append(-1)

                        gold_labels = torch.LongTensor(gold_labels).to(device)
                        g_is = torch.stack(g_is)
                        if use_pruning:
                            sense_embs = torch.stack([sense_emb] * n_span, dim=0)
                            scores = self.mention_score(torch.cat((g_is, sense_embs),dim=1)) # [n_span, 3072] -> [n_span, 1] TODO distance 추가?
                            scores = scores.flatten()
                            is_gold = (gold_labels > -1).float()
                            probs = F.softmax(scores)
                            eps = 1e-8
                            # Negative marginal log-likelihood
                            span_loss += torch.log(torch.sum(torch.mul(probs, is_gold)).clamp_(eps, 1 - eps)) * -1

                            top_spans_indices = torch.topk(scores.flatten(), min(self.k, spans.shape[0])).indices  # top_spans.values, top_spans.indices
                            top_spans.append(spans[top_spans_indices])  # [k, 3]
                            top_spans_label.append(gold_labels[top_spans_indices])
                            top_spans_gis.append(g_is[top_spans_indices])
                            top_spans_scores.append(scores[top_spans_indices])

                    def list_to_tensor(input):
                        tensors = None
                        for item in input:
                            if tensors is None:
                                tensors = item
                            else:
                                tensors = torch.cat((tensors, item))
                        return tensors

                    # if unique_args_uid.shape[0] > 1:
                    #     print()
                    top_spans = list_to_tensor(top_spans)  # [#unique_args_uid * k(가변), 3]
                    top_spans_label = list_to_tensor(top_spans_label)  # [#unique_args_uid * k]
                    top_spans_gis = list_to_tensor(top_spans_gis)  # [#unique_args_uid * k, 2304]
                    top_spans_scores = list_to_tensor(top_spans_scores)  # [#unique_args_uid * k]

                ''' FE slot filling '''
                try:
                    n_top_span = top_spans.shape[0]
                except:
                    continue
                if n_top_span == 0:
                    continue

                #top_spans_gis, sense_emb, top_spans, top_spans_dist, target_uid, top_spans_label
                top_spans_dist = top_spans[:,0] - target_uid  # target과의 거리
                top_spans_dist_emb = self.distance(top_spans_dist)  # [n_top_span, 20]
                sense_embs = torch.stack([sense_emb] * n_top_span, dim=0)
                features = torch.cat((top_spans_gis,top_spans_dist_emb, sense_embs), dim=1)  # [n_top_span, 768*3+150+20]
                fe_logits = self.fe_classifier(features)  # [n_top_span, n_fe+1]
                linking_loss += loss_fct_arg(fe_logits, top_spans_label)


            span_loss, pair_loss, linking_loss = span_loss / input_ids.shape[0], pair_loss / input_ids.shape[0], linking_loss / input_ids.shape[0]

            if type(loss) == float:
                print()

            return span_loss, pair_loss, linking_loss

        else:  # is inference
            for i in range(input_ids.shape[0]):  # batch를 iter.
                utter_len, speaker_len = utter_lens[i], speaker_lens[i]
                pred_utter = None

                if use_gold_span:
                    arg_len = arg_lens[i]
                    gold_spans = gold_args[i][:arg_len]

                gold_sense = senses[i]  # [1]
                sense_emb = self.frame_embs(gold_sense)  # [768]
                target = targets[i]  # [3] utter_id, st, en
                target_uid = target[0]
                scene_ids, scene_speakers, special_token = input_ids[i][:utter_len], speakers[i][:utter_len], \
                                                           special_tokens[i][:utter_len]
                scene_maps = orig_tok_to_maps[i][:utter_len]

                top_spans = []  # scene에서 추출한 spans
                top_spans_label = []
                top_spans_gis = []
                top_spans_scores = []

                scene_token_type_ids = token_type_ids[i][:utter_len]
                scene_attention_mask = attention_mask[i][:utter_len]

                sequence_output, pooled_output = self.bert(scene_ids, token_type_ids=scene_token_type_ids,
                                                           attention_mask=scene_attention_mask)  # [n_utter, 256, 768], [n_utter, 768]

                if not use_gold_span:
                    ''' utterance-pair classification '''
                    target_cls = pooled_output[target_uid]  # [768]
                    pair_clses = torch.cat((torch.stack([target_cls] * utter_len.item()), pooled_output),
                                           dim=1)  # [n_utter, 768*2]
                    pair_labels = torch.zeros(utter_len.item(), dtype=torch.long)  # [n_utter]
                    pair_labels = pair_labels.cuda()

                    # TODO: speaker emb도 추가해야함. 같은지 다른지.
                    dist = torch.LongTensor([self.boundary_filter(i - target_uid) + 7 for i in
                                             range(utter_len)]).cuda()  # -7 ~ 5 의 값으로 나오기 때문에 +7을해서 0 ~ 12로  [n_utter]
                    dist_emb = self.distance(dist)  # [n_utter, 20]

                    cur_pair_loss, pair_output = self.pair_score(pair_clses, pair_labels, dist_emb, sense_emb)
                    # pair_output : [#utter, 2]  0 == fe가 존재하지 않음, 1 == fe가 존재
                    pred_pair = torch.argmax(pair_output, dim=1)
                    unique_args_uid = (pred_pair > 0).nonzero().flatten()
                    pred_utter = unique_args_uid

                    if pred_utter.shape[0] == 0:  # 비었으면,
                        return None, None, None

                attns = self.attention(sequence_output)  # [2,256,1]

                st, en = target[1], target[2]
                if st.item() > 255 or en.item() > 255:
                    continue

                lu_emb = self.span_representation(sequence_output[target_uid], attns[target_uid], st, en)

                ''' FE span extractor '''
                if use_gold_span:
                    span_candidates = gold_spans
                    top_spans = gold_spans[:, :-1]
                    top_spans_label = gold_spans[:, -1]
                    top_spans_gis = []
                    for span in top_spans:
                        span_uid, st, en = span
                        utter_embs = sequence_output[span_uid]
                        utter_attns = attns[span_uid]

                        if self.use_gi_variation:
                            g_i = self.span_representation(utter_embs, utter_attns, st, en)
                            g_i = torch.cat((g_i, lu_emb), dim=0)
                        else:
                            span_attn, span_emb = utter_attns[st:en + 1], utter_embs[
                                                                          st:en + 1]  # [span_len,1], [span_len, 768]
                            attn_weights = F.softmax(span_attn, dim=1)  # [span_len, 1]
                            attn_embed = torch.sum(torch.mul(span_emb, attn_weights), dim=0)  # [768]
                            g_i = torch.cat((utter_embs[st], utter_embs[en], attn_embed, lu_emb), dim=0)  # [768*3]
                        top_spans_gis.append(g_i)
                    top_spans_gis = torch.stack(top_spans_gis)
                    pred_top_spans = top_spans
                else:
                    span_candidates = self.extract_span(unique_args_uid, special_token, scene_maps)
                    for u_idx, uid in enumerate(unique_args_uid):
                        utter_embs = sequence_output[uid]
                        utter_attns = attns[uid]
                        spans = span_candidates[u_idx].to(device)
                        n_span = len(spans)
                        g_is = []

                        if n_span == 0:
                            continue

                        for span in spans:
                            span_uid, st, en = span

                            if self.use_gi_variation:
                                g_i = self.span_representation(utter_embs, utter_attns, st, en)
                                g_i = torch.cat((g_i, lu_emb), dim=0)
                            else:
                                span_attn, span_emb = utter_attns[st:en + 1], utter_embs[
                                                                              st:en + 1]  # [span_len,1], [span_len, 768]
                                attn_weights = F.softmax(span_attn, dim=1)  # [span_len, 1]
                                attn_embed = torch.sum(torch.mul(span_emb, attn_weights), dim=0)  # [768]
                                g_i = torch.cat((utter_embs[st], utter_embs[en], attn_embed, lu_emb), dim=0)  # [768*3]
                            g_is.append(g_i)
                            # g_i_lu = torch.cat((g_i, lu_emb), dim=0)  # [768*6]

                        g_is = torch.stack(g_is)

                        if use_pruning:
                            sense_embs = torch.stack([sense_emb] * n_span, dim=0)
                            scores = self.mention_score(
                                torch.cat((g_is, sense_embs), dim=1))  # [n_span, 3072] -> [n_span, 1] TODO distance 추가?
                            scores = scores.flatten()
                            top_spans_indices = torch.topk(scores.flatten(), min(self.k, spans.shape[
                                0])).indices  # top_spans.values, top_spans.indices
                            top_spans.append(spans[top_spans_indices])  # [k, 3]
                            top_spans_gis.append(g_is[top_spans_indices])

                    def list_to_tensor(input):
                        tensors = None
                        for item in input:
                            if tensors is None:
                                tensors = item
                            else:
                                tensors = torch.cat((tensors, item))
                        return tensors

                    top_spans = list_to_tensor(top_spans)  # [#unique_args_uid * k(가변), 3]
                    top_spans_gis = list_to_tensor(top_spans_gis)  # [#unique_args_uid * k, 2304]
                    pred_top_spans = top_spans

                ''' FE slot filling '''
                n_top_span = top_spans.shape[0]
                if n_top_span == 0:
                    return pred_utter, pred_top_spans, None


                top_spans_dist = top_spans[:, 0] - target_uid  # target과의 거리
                top_spans_dist_emb = self.distance(top_spans_dist)  # [n_top_span, 20]
                sense_embs = torch.stack([sense_emb] * n_top_span, dim=0)
                features = torch.cat((top_spans_gis, top_spans_dist_emb, sense_embs),
                                     dim=1)  # [n_top_span, 768*3+150+20]
                fe_logits = self.fe_classifier(features) # [n_top_span, 1286(n_fe + 1)]
                pred_labels = torch.argmax(fe_logits, dim=1)
                pred_labels = [(span, pred_labels[ii]) for ii, span in enumerate(top_spans)]
                return pred_utter, pred_top_spans, pred_labels

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
