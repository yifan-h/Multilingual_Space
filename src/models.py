import os
import copy
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
import transformers.adapters.composition as ac


class MLKGLM(nn.Module):
    """docstring for ClassName"""
    def __init__(self, args):
        super(MLKGLM, self).__init__()
        # load pretrained MLLM
        self.MLLM = AutoModel.from_pretrained(args.model_dir, 
                                                return_dict=True,
                                                output_hidden_states=True)
        hidden_num = self.MLLM.get_input_embeddings().embedding_dim
        self.training = True
        self.lm_mask_token_id = args.lm_mask_token_id
        # set three extra modules
        self.knowledge_mapping = nn.Sequential(nn.Linear(hidden_num, int(hidden_num / 4)),
                                                nn.ELU(),
                                                nn.Dropout(0.1),  # project down
                                                nn.Linear(int(hidden_num / 4), hidden_num),
                                                nn.ELU(),
                                                nn.LayerNorm(4*hidden_num, eps=1e-12),
                                                nn.Dropout(0.1),  # project up
                                                nn.Linear(hidden_num, 4*hidden_num),
                                                nn.ELU(),
                                                nn.Dropout(0.1),  # project up
                                                nn.Linear(4*hidden_num, hidden_num),
                                                nn.ELU(),
                                                nn.LayerNorm(hidden_num, eps=1e-12),
                                                nn.Dropout(0.1),  # project down
                                                nn.Linear(hidden_num, hidden_num),
                                                nn.Tanh(),
                                                nn.Dropout(0.1))
        # for testing
        self.all_aggregator = nn.Linear(2*hidden_num, hidden_num, bias=False)
        self.all_aggregator.weight.data = self.weight_init_sum(self.all_aggregator.weight.data)

    def weight_init_sum(self, t):
        hidden_num = int(t.shape[-1]/2)
        nn.init.xavier_normal_(t)
        return t*0.05 + torch.cat((0.5*torch.eye(hidden_num,hidden_num),
                                    0.5*torch.eye(hidden_num,hidden_num)),dim=1)

    def forward(self, **inputs):
        # get MLLM output
        outputs_MLLM = self.MLLM(**inputs).hidden_states
        # take last layer hidden state: (batch_size, sequence_length, hidden_size)
        outputs_MLLM = outputs_MLLM[-1]
        # add adversarial noise
        if self.training:
            outputs_MLLM = outputs_MLLM + 0.05*torch.randn_like(outputs_MLLM)
        outputs_both = self.knowledge_mapping(outputs_MLLM)
        if self.training:
            return (outputs_both + outputs_MLLM) / 2
        else:
            outputs_both = self.all_aggregator(torch.cat((outputs_MLLM, outputs_MLKGLM), dim=-1))
            return outputs_both


def loss_universal(args, outputs, lossfcn, input_ids=None, el2=True):
    # transform set-level to sample-level
    # outputs = torch.mean(outputs, dim=1)
    outputs_pos = outputs[:int(outputs.shape[0]/2)]
    if input_ids is not None:
        outputs_pos = get_mask(outputs_pos, input_ids, args.lm_mask_token_id)
    outputs_neg = outputs[int(outputs.shape[0]/2):]
    # average
    outputs_pos = torch.mean(outputs_pos, dim=1)
    outputs_neg = torch.mean(outputs_neg, dim=1)
    idx_query, idx_pos = [], []
    for i in range(int(outputs.shape[0]/2)):
        for j in range(int(outputs.shape[0]/2)):
            if i > j:
                idx_query.append(i)
                idx_pos.append(j)
    '''
    if len(idx_query) > args.batch_num:
        idx_all = [i for i in range(len(idx_query))]
        idx_random = random.sample(idx_all, args.batch_num)
        idx_query = [idx_query[i] for i in idx_random]
        idx_pos = [idx_pos[i] for i in idx_random]
    '''
    loss_dp = lossfcn(outputs_pos[idx_query], outputs_pos[idx_pos], outputs_neg)
    if el2 == True:
        lossfcn_el2 = nn.SmoothL1Loss()
        loss_el2 = lossfcn_el2(outputs_pos[idx_query], outputs_pos[idx_pos]) - lossfcn_el2(outputs_pos[idx_query], outputs_neg)
        return loss_dp + loss_el2
    else:
        return loss_dp


def loss_triple(args, outputs, lossfcn, input_ids=None, el2=True):
    # transform set-level to sample-level
    # outputs = torch.mean(outputs, dim=1)
    outputs_query = outputs[:int(outputs.shape[0]/3)]
    if input_ids is not None:
        outputs_query = get_mask(outputs_query, input_ids, args.lm_mask_token_id)
    outputs_pos = outputs[int(outputs.shape[0]/3):int(outputs.shape[0]/3*2)]
    outputs_neg = outputs[int(outputs.shape[0]/3*2):]
    # average
    outputs_query = torch.mean(outputs_query, dim=1)
    outputs_pos = torch.mean(outputs_pos, dim=1)
    outputs_neg = torch.mean(outputs_neg, dim=1)
    loss_dp = lossfcn(outputs_query, outputs_pos, outputs_neg)
    if el2 == True:
        lossfcn_el2 = nn.SmoothL1Loss()
        loss_el2 = lossfcn_el2(outputs_query, outputs_pos) - lossfcn_el2(outputs_query, outputs_neg)
        return loss_dp + loss_el2
    else:
        return loss_dp

def get_mask(outputs, input_ids, lm_mask_token_id):
    tmp_batch_num = input_ids.shape[0]
    for i in range(int(tmp_batch_num)):
        if lm_mask_token_id not in input_ids[i]: continue
        mask_idx = ((input_ids[i] == lm_mask_token_id).nonzero(as_tuple=True)[0])
        if len(mask_idx) == 1:
            outputs[i,mask_idx[0]:,:] = 0
        else: ## len(mask_idx) == 2
            outputs[i,:mask_idx[0],:] = 0
            outputs[i,mask_idx[1]:,:] = 0
    return outputs

'''
class MLKGLM(nn.Module):
    """docstring for ClassName"""
    def __init__(self, args):
        super(MLKGLM, self).__init__()
        # load pretrained MLLM
        self.MLLM = AutoModel.from_pretrained(args.model_dir, 
                                            return_dict=True,
                                            output_hidden_states=True)
        hidden_num = self.MLLM.get_input_embeddings().embedding_dim
        self.training = False
        # set three extra modules
        self.universal_mapping = nn.Sequential(Conv1D(hidden_num, hidden_num),
                                                    nn.ELU(),
                                                    nn.LayerNorm(hidden_num, eps=1e-12),
                                                    nn.Dropout(0.1),
                                                    Conv1D(4*hidden_num, hidden_num),
                                                    nn.ELU(),
                                                    nn.LayerNorm(4*hidden_num, eps=1e-12),
                                                    nn.Dropout(0.1),
                                                    Conv1D(hidden_num, 4*hidden_num),
                                                    nn.ELU(),
                                                    nn.LayerNorm(hidden_num, eps=1e-12),
                                                    nn.Dropout(0.1),
                                                    Conv1D(hidden_num, hidden_num),
                                                    nn.ELU(),
                                                    nn.LayerNorm(hidden_num, eps=1e-12),
                                                    nn.Dropout(0.1))
        self.universal_aggregator = nn.Sequential(Conv1D(hidden_num, 2*hidden_num),
                                                    nn.LayerNorm(hidden_num, eps=1e-12),
                                                    nn.Dropout(0.1))
        self.triple_mapping = nn.Sequential(Conv1D(hidden_num, hidden_num),
                                                nn.ELU(),
                                                nn.LayerNorm(hidden_num, eps=1e-12),
                                                nn.Dropout(0.1),
                                                Conv1D(4*hidden_num, hidden_num),
                                                nn.ELU(),
                                                nn.LayerNorm(4*hidden_num, eps=1e-12),
                                                nn.Dropout(0.1),
                                                Conv1D(hidden_num, 4*hidden_num),
                                                nn.ELU(),
                                                nn.LayerNorm(hidden_num, eps=1e-12),
                                                nn.Dropout(0.1),
                                                Conv1D(hidden_num, hidden_num),
                                                nn.ELU(),
                                                nn.LayerNorm(hidden_num, eps=1e-12),
                                                nn.Dropout(0.1))
        self.triple_aggregator = nn.Sequential(Conv1D(hidden_num, 2*hidden_num),
                                                nn.LayerNorm(hidden_num, eps=1e-12),
                                                nn.Dropout(0.1))
        # for testing
        self.all_aggregator = nn.Sequential(nn.Linear(3*hidden_num, hidden_num, bias=False),
                                            nn.Dropout(0.2))
        # self.all_aggregator.weight.data = self.weight_init_sum(self.all_aggregator.weight.data)

    def weight_init_sum(self, t):
        hidden_num = int(t.shape[-1]/3)
        return 0.0003*torch.randn(3*hidden_num, ) + torch.cat((0.333*torch.eye(hidden_num,hidden_num),
                                    0.333*torch.eye(hidden_num,hidden_num),
                                    0.333*torch.eye(hidden_num,hidden_num)),dim=1)

    def forward(self, **inputs):
        # get MLLM output
        outputs_MLLM = self.MLLM(**inputs).hidden_states
        # take last layer hidden state: (batch_size, sequence_length, hidden_size)
        outputs_MLLM = outputs_MLLM[-1]
        # objective 1: universal space
        outputs_universal = self.universal_mapping(outputs_MLLM)
        outputs_universal = self.universal_aggregator(torch.cat((outputs_MLLM, outputs_universal), dim=-1))
        # objective 2: transformer layers
        outputs_MLKGLM = self.triple_mapping(outputs_universal)
        outputs_MLKGLM = self.triple_aggregator(torch.cat((outputs_MLLM, outputs_MLKGLM), dim=-1))
        return self.all_aggregator(torch.cat((outputs_MLLM, outputs_universal, outputs_MLKGLM), dim=-1))
        # return (outputs_MLLM + outputs_universal + outputs_MLKGLM)/3
'''
