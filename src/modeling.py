from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from boltons.iterutils import windowed
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

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
        return to_cuda(torch.tensor([
            sum([True for i in self.bins if num >= i]) for num in lengths], requires_grad=False
        ))

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



class FrameTypeParser(BertPreTrainedModel):
    def __init__(self, config, num_senses=2, lufrmap=None, device=device, masking=True,
                 return_pooled_output=False, original_loss=False):
        super(FrameTypeParser, self).__init__(config)
        self.masking = masking
        self.num_senses = num_senses  # total number of all frames
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.sense_classifier = nn.Linear(config.hidden_size, num_senses)
        self.lufrmap = lufrmap  # mapping table for lu to its frame candidates, lu idx와 해당 lu가 속할 수 있는 frame indices.
        self.return_pooled_output = return_pooled_output
        self.original_loss = original_loss
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, lus=None, senses=None,
                using_gold_fame=False, position_ids=None, head_mask=None):
        # token_type_ids : segment token indices [0,1] 0: sent A, 1: sent B
        sequence_output, pooled_output = self.bert(input_ids, token_type_ids=token_type_ids,
                                                   attention_mask=attention_mask)
        # seq_output : [bsz, seq_size(256), hidden_size], pooled_output : [bsz, hidden_size]
        sequence_output = self.dropout(sequence_output)
        pooled_output = self.dropout(pooled_output)

        sense_logits = self.sense_classifier(pooled_output)  # [bsz, # of frame type]

        lufr_masks = utils.get_masks(lus, self.lufrmap, num_label=self.num_senses, masking=self.masking).to(
            device)  # [bsz, # of frame type] lu가 가질 수 있는 frame type이라면 1, 아니면 0

        sense_loss = 0  # loss for sense id

        if senses is not None:  # is train
            for i in range(len(sense_logits)):  # batch를 iter.
                sense_logit = sense_logits[i]  # [# of frame types]
                lufr_mask = lufr_masks[i]  # [# of frame types]
                gold_sense = senses[i]  # [1]
                # train sense classifier
                loss_fct_sense = CrossEntropyLoss(weight=lufr_mask)
                loss_per_seq_for_sense = loss_fct_sense(sense_logit.view(-1, self.num_senses), gold_sense.view(-1))
                sense_loss += loss_per_seq_for_sense


            total_loss = sense_loss
            loss = total_loss / len(sense_logits)

            if self.return_pooled_output:
                return pooled_output, loss
            else:
                return loss
        else:  # is inference
            if self.return_pooled_output:
                return pooled_output, sense_logits
            else:
                return sense_logits

class ArgsParser(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        """
        return_dict = False

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return total_loss

        # return QuestionAnsweringModelOutput(
        #     loss=total_loss,
        #     start_logits=start_logits,
        #     end_logits=end_logits,
        #     hidden_states=outputs.hidden_states,
        #     attentions=outputs.attentions,
        # )

class SpanArgsParser(BertPreTrainedModel):
    """ e2e span extract & role labeler """
    def __init__(self, config, num_senses=-1, num_args=-1, frargmap=None):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.attention = Score(config.hidden_size)
        self.width = Distance(20)
        self.frame_embs = nn.Embedding(num_senses, 20)
        self.score = Score(3*config.hidden_size + 20*2)

        self.distance = Distance(20)
        self.fe_embs = nn.Embedding(num_args + 1, config.hidden_size)  # 1285번째 idx는 None으로
        self.pair_score = Score(4*config.hidden_size + 20*2 + 20*1)

        self.frargmap = frargmap


        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        input_orig_tok_to_maps=None,
        lus=None,
        senses=None,
        args=None,
        gold_spans=None,
        span_pad=None,
        invalid_pos=None   # [cls, tgt, \tgt, sep의 bert tok indices가 순서대로 들어있음.]
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        """
        return_dict = False



        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output, pooled_output = outputs[0], outputs[1]  # token마다, cls
        bsz = pooled_output.shape[0]
        hidden_size = pooled_output.shape[1]

        # "O": 0, "X": 1

        sequence_output = self.dropout(sequence_output)  # [6, 256, 768]
        pooled_output = self.dropout(pooled_output)
        attns = self.attention(sequence_output)

        b_spans, b_probs = [], []

        if senses is not None:  # is train
            for i in range(bsz):  # batch를 iter.
                gold_sense = senses[i]

                attn = attns[i]  # [256, 1]
                seq_output = sequence_output[i]
                # extract span candidates indices.
                invalid = invalid_pos[i]
                input_orig_tok_to_map = input_orig_tok_to_maps[i]
                span_candidates = self.extract_span_candidates(invalid, input_orig_tok_to_map)
                g_is = []
                frame_emb = self.frame_embs(gold_sense)

                for candidate in span_candidates:
                    st, en = candidate.st, candidate.en
                    span_attn, span_emb = attn[st:en+1], seq_output[st:en+1]  # [span_len,1], [span_len, 768]
                    attn_weights = F.softmax(span_attn, dim=1)  # [span_len, 1]
                    attn_embed = torch.sum(torch.mul(span_emb, attn_weights), dim=0)  # [768]
                    width = self.width([st-en+1]) # [1, 20]
                    g_i = torch.cat((seq_output[st], seq_output[en], attn_embed, torch.squeeze(frame_emb), torch.squeeze(width)))  # [768*3 + 20*2]
                    g_is.append(g_i)

                if len(g_is) == 0:
                    # b_spans.append([])
                    # b_probs.append(torch.tensor([]))
                    continue

                g_is = torch.stack(g_is)
                mention_scores = self.score(g_is)  # [num_span, 1]

                for x,y,z in zip(span_candidates, g_is, mention_scores.detach()):
                    x.g_i = y
                    x.score = z

                spans = self.prune(span_candidates, 3)  ##TODO 현재 frame의 element 개수를 구하고, 그 개수 span만 뽑아내기.
                n_span = len(spans)
                fe_indices = to_cuda(torch.tensor(self.frargmap[str(gold_sense.item())]))  #[num_fe]
                fe_embs = self.fe_embs(fe_indices)  # [num_fe, 20]
                n_fe = len(self.frargmap[str(gold_sense.item())])
                st_tgt, en_tgt = invalid_pos[i][1], invalid_pos[i][2]
                distances = self.distance(self.calc_distance(spans, st_tgt, en_tgt)) # [span_len, 20]
                g_is = to_cuda(torch.stack([s.g_i for s in spans]))  # [span_len, 2344]
                fe_embs = torch.stack([fe_embs] * n_span, dim=0)  # [len_span num_fe 20]
                distances = torch.stack([distances] * n_fe, dim=1)  # [len_span num_fe 20]
                g_is = torch.stack([g_is] * n_fe, dim=1)  # [span_len num_fe 2344]

                # print(g_is.shape, fe_embs.shape, distances.shape)
                features = torch.cat((g_is, fe_embs, distances), dim=2)  # [span_len, num_fe, 2384]

                pair_scores = self.pair_score(features)  # [span_len, num_fe, 1]
                mention_scores = to_cuda(torch.stack([s.score for s in spans])) #[span_len, 1]
                stacked_mention_scores = torch.stack([mention_scores]*n_fe, dim=1)  # [span, fe, 1]
                scores = torch.cat((pair_scores, stacked_mention_scores), dim=2)  # [span, fe, 2]
                scores = torch.sum(scores, dim=2)  # [span, fe]
                epsilon = to_var(torch.zeros(n_span, 1))  # [span, 1]  not ne case
                scores_with_epsilon = torch.cat((scores, epsilon), dim=1)  # [span, fe+1]

                probs = [F.softmax(score) for score in scores_with_epsilon]
                probs = torch.stack(probs)  # [span, fe+1]

                b_spans.append(spans)
                b_probs.append(probs)

        return b_spans, b_probs




    def calc_distance(self, spans, st_tgt, en_tgt):
        distances = []
        for s in spans:
            if en_tgt < s.st:
                d = s.st - en_tgt
            elif s.en < st_tgt:
                d = st_tgt - s.en
            distances.append(d)
        return distances

    def prune(self, candidates, k=3):
        candidates = sorted(candidates, key=lambda s: s.score, reverse=True)
        candidates = self.remove_overlapping(candidates)
        pruned_spans = candidates[:k]
        return pruned_spans

    def remove_overlapping(self, sorted_spans):
        nonoverlapping, seen = [], set()
        for s in sorted_spans:
            indexes = range(s.st, s.en + 1)
            taken = [i in seen for i in indexes]
            if len(set(taken)) == 1 or (taken[0] == taken[-1] == False):
                nonoverlapping.append(s)
                seen.update(indexes)

        return nonoverlapping

    def extract_span_candidates(self, special_tokens, map):
        max_len = special_tokens[-1] # [sep] 위치
        invalid = special_tokens[:-1]
        span_candidates = [Span(id, i[0], i[-1]) for id, i in enumerate(self.compute_idx_spans(max_len))]
        del_list = set()
        invalid = set([x for x in range(invalid[1], invalid[2] + 1)])

        start_valid = [int(x) for x in map if x != -1]
        end_valid = [int(x) - 1 for x in start_valid[1:]] + [int(max_len) - 1]

        for ii, candidate in enumerate(span_candidates):
            if len(candidate) == 1:
                indices = set([candidate.st])
            else:
                indices = set([x for x in range(candidate.st, candidate.en + 1)])

            if len(indices.intersection(invalid)) > 0:
                del_list.add(ii)
            if candidate.st not in start_valid:
                del_list.add(ii)
            if candidate.en not in end_valid:
                del_list.add(ii)
        del_list = list(del_list)
        del_list.sort()
        del_list.reverse()
        for ii in del_list:
            del span_candidates[ii]
        return span_candidates

    def compute_idx_spans(self, max_len, L=10):
        """ Compute span indexes for all possible spans up to length L in each
        sentence """

        def flatten(alist):
            """ Flatten a list of lists into one list """
            return [item for sublist in alist for item in sublist]

        idx_spans, shift = [], 0
        sent_spans = flatten([windowed(range(shift, max_len + shift), length)
                              for length in range(1, L)])

        return sent_spans

class BertForJointShallowSemanticParsing(BertPreTrainedModel):
    def __init__(self, config, num_senses=2, num_args=2, lufrmap=None, frargmap=None, masking=True, return_pooled_output=False, original_loss=False):
        super(BertForJointShallowSemanticParsing, self).__init__(config)
        self.masking = masking
        self.num_senses = num_senses # total number of all frames
        self.num_args = num_args # total number of all frame elements
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.sense_classifier = nn.Linear(config.hidden_size, num_senses)
        self.arg_classifier = nn.Linear(config.hidden_size, num_args) # all FEs
        self.lufrmap = lufrmap # mapping table for lu to its frame candidates, lu idx와 해당 lu가 속할 수 있는 frame indices.
        self.frargmap = frargmap # mapping table for frame to its frame element candidates
        self.return_pooled_output = return_pooled_output
        self.original_loss = original_loss
        
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, lus=None, senses=None, args=None, using_gold_fame=False, position_ids=None, head_mask=None):
        # token_type_ids : segment token indices [0,1] 0: sent A, 1: sent B
        sequence_output, pooled_output = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        # seq_output : [bsz, seq_size(256), hidden_size], pooled_output : [bsz, hidden_size]
        sequence_output = self.dropout(sequence_output)
        pooled_output = self.dropout(pooled_output)
        
        sense_logits = self.sense_classifier(pooled_output) # [bsz, # of frame type]
        arg_logits = self.arg_classifier(sequence_output)  # [bsz, seq_size(256), # of FEs]

        lufr_masks = utils.get_masks(lus, self.lufrmap, num_label=self.num_senses, masking=self.masking).to(device) # [bsz, # of frame type] lu가 가질 수 있는 frame type이라면 1, 아니면 0
        
        sense_loss = 0 # loss for sense id
        arg_loss = 0 # loss for arg id
        
        if senses is not None:  # is train
            for i in range(len(sense_logits)):  # batch를 iter.
                sense_logit = sense_logits[i] # [# of frame types]
                arg_logit = arg_logits[i]  # [seq_size(256), # of FEs]

                lufr_mask = lufr_masks[i]  # [# of frame types]
                    
                gold_sense = senses[i]  # [1]
                gold_arg = args[i]  # [seq_size(256)]
                
                #train sense classifier
                loss_fct_sense = CrossEntropyLoss(weight = lufr_mask)
                loss_per_seq_for_sense = loss_fct_sense(sense_logit.view(-1, self.num_senses), gold_sense.view(-1))
                sense_loss += loss_per_seq_for_sense
                
                #train arg classifier
                masked_sense_logit = utils.masking_logit(sense_logit, lufr_mask)
                pred_sense, sense_score = utils.logit2label(masked_sense_logit)  # arg max

                frarg_mask = utils.get_masks([pred_sense], self.frargmap, num_label=self.num_args, masking=True).to(device)[0]                
                loss_fct_arg = CrossEntropyLoss(weight = frarg_mask)

                
                # only keep active parts of loss
                if attention_mask is not None:
                    active_loss = attention_mask[i].view(-1) == 1  # attention_mask[i] == [256] -> [256]
                    active_logits = arg_logit.view(-1, self.num_args)[active_loss]  # arg_logit == [256,num_args], arg_logit.view(-1, self.num_args) == [256,num_args], result = [69(active 된 애들),num_args]
                    active_labels = gold_arg.view(-1)[active_loss]  # gold_arg == [256], gold_arg.view(-1) == [256], result = [69]
                    loss_per_seq_for_arg = loss_fct_arg(active_logits, active_labels)
                else:
                    loss_per_seq_for_arg = loss_fct_arg(arg_logit.view(-1, self.num_args), gold_arg.view(-1))
                arg_loss += loss_per_seq_for_arg

            total_loss = 0.5*sense_loss + 0.5*arg_loss
            loss = total_loss / len(sense_logits)
            
            if self.return_pooled_output:
                return pooled_output, loss
            else:
                return loss
        else:  # is inference
            if self.return_pooled_output:
                return pooled_output, sense_logits, arg_logits
            else:
                return sense_logits, arg_logits
