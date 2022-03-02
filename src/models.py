import os
import copy
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, Conv1D
import transformers.adapters.composition as ac


class MLKGLM(nn.Module):
    """docstring for ClassName"""
    def __init__(self, args):
        super(MLKGLM, self).__init__()
        # load pretrained MLLM
        self.MLLM = AutoModel.from_pretrained(args.model_dir, 
                                            return_dict=True,
                                            output_hidden_states=True)
        '''
        # set and activate adapters
        adapters = []
        for i in range(args.adapter_num):
            adapters.append("adapter"+str(i+1))
            self.MLLM.add_adapter(adapters[i])
        self.MLLM.add_adapter_fusion(adapters)
        self.MLLM.active_adapters = ac.Fuse(*adapters)
        '''
        hidden_num = self.MLLM.get_input_embeddings().embedding_dim
        self.lm_mask_token_id = args.lm_mask_token_id
        # set mask status
        self.training_mask = False
        # set three extra modules
        self.entity_masking = nn.Sequential(Conv1D(hidden_num, 2))
        self.universal_mapping = nn.Sequential(Conv1D(hidden_num, hidden_num),
                                                nn.LayerNorm(hidden_num, eps=1e-12),
                                                nn.Dropout(0.1),
                                                Conv1D(4*hidden_num, hidden_num),
                                                Conv1D(hidden_num, 4*hidden_num),
                                                nn.LayerNorm(hidden_num, eps=1e-12),
                                                nn.Dropout(0.1))
        self.triple_mapping = nn.Sequential(Conv1D(hidden_num, hidden_num),
                                            nn.LayerNorm(hidden_num, eps=1e-12),
                                            nn.Dropout(0.1),
                                            Conv1D(4*hidden_num, hidden_num),
                                            Conv1D(hidden_num, 4*hidden_num),
                                            nn.LayerNorm(hidden_num, eps=1e-12),
                                            nn.Dropout(0.1))

    def forward(self, **inputs):
        # get MLLM output
        outputs_MLLM = self.MLLM(**inputs).hidden_states
        # take last layer hidden state: (batch_size, sequence_length, hidden_size)
        outputs_MLLM = outputs_MLLM[-1]
        outputs_mask_logit = outputs_MLLM*0
        outputs_mask = outputs_MLLM*0
        # objective 0: get entity mask
        if self.training_mask:  # masking training
            outputs_mask_logit = F.sigmoid(self.entity_masking(outputs_MLLM))
            outputs_mask = torch.where(outputs_mask_logit<0.5, outputs_mask, outputs_mask+1)
            outputs_MLLM = outputs_mask*outputs_MLLM
        else:  # entity/triple training
            if self.lm_mask_token_id in inputs["input_ids"]:  # triple: mask relation
                input_ids = inputs["input_ids"] # identical: (batch_size, sequence_length, hidden_size) 
                input_ids = torch.transpose(torch.transpose(input_ids.expand(outputs_MLLM.shape[2],
                                                                    outputs_MLLM.shape[0],
                                                                    outputs_MLLM.shape[1]),0,1),1,2)
                outputs_mask = torch.where(input_ids==self.lm_mask_token_id, outputs_mask, outputs_mask+1)
                outputs_MLLM = outputs_mask*outputs_MLLM
        # objective 1: universal space
        outputs_universal = self.universal_mapping(outputs_MLLM)
        outputs_universal = (outputs_MLLM+outputs_universal) / 2
        # objective 2: transformer layers
        outputs_MLKGLM = self.triple_mapping(outputs_universal)
        outputs_MLKGLM = (outputs_universal+outputs_MLKGLM) / 2
        return outputs_mask_logit, outputs_universal, outputs_MLKGLM


def loss_universal(args, outputs, lossfcn):
    # transform set-level to sample-level
    outputs = torch.mean(outputs, dim=1)
    outputs_pos = outputs[:int(outputs.shape[0]/2)]
    outputs_neg = outputs[int(outputs.shape[0]/2):]
    idx_query, idx_pos = [], []
    for i in range(int(outputs.shape[0]/2)):
        for j in range(int(outputs.shape[0]/2)):
            if i > j:
                idx_query.append(i)
                idx_pos.append(j)
    if len(idx_query) > args.batch_num:
        idx_all = [i for i in range(len(idx_query))]
        idx_random = random.sample(idx_all, args.batch_num)
        idx_query = [idx_query[i] for i in idx_random]
        idx_pos = [idx_pos[i] for i in idx_random]
    return lossfcn(outputs_pos[idx_query], outputs_pos[idx_pos], outputs_neg)


def loss_triple(outputs, lossfcn):
    # transform set-level to sample-level
    outputs = torch.mean(outputs, dim=1)
    outputs_query = outputs[:int(outputs.shape[0]/2)]
    outputs_pos = outputs[int(outputs.shape[0]/2):]
    return lossfcn(outputs_query, outputs_pos)
