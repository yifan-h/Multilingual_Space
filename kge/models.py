import os
import copy
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, Conv1D
from info_nce import InfoNCE

from utils import load_model

seed = 123
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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
        # self.lm_pad_token_id = args.lm_pad_token_id
        # set three extra modules
        self.knowledge_mapping = nn.Sequential(nn.Linear(hidden_num, int(hidden_num / 2)),
                                                nn.ELU(),
                                                nn.Dropout(0.1),  # project down
                                                nn.Linear(int(hidden_num / 2), hidden_num),
                                                nn.ELU(),
                                                nn.LayerNorm(hidden_num, eps=1e-12),
                                                nn.Dropout(0.1),  # project up
                                                nn.Linear(hidden_num, 4*hidden_num),
                                                nn.ELU(),
                                                nn.Dropout(0.1),  # project up
                                                nn.Linear(4*hidden_num, hidden_num),
                                                nn.ELU(),
                                                nn.LayerNorm(hidden_num, eps=1e-12),
                                                nn.Dropout(0.1),  # project down
                                                nn.Linear(hidden_num, hidden_num),
                                                nn.ELU(),
                                                nn.LayerNorm(hidden_num, eps=1e-12),
                                                nn.Dropout(0.1))
        if not self.training:
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
            outputs_MLLM = outputs_MLLM + 0.1*torch.abs(outputs_MLLM).mean()*torch.randn_like(outputs_MLLM)
        outputs_both = self.knowledge_mapping(outputs_MLLM)
        if self.training:
            return (outputs_both + outputs_MLLM) / 2, outputs_MLLM.clone()
        else:
            outputs_both = self.all_aggregator(torch.cat((outputs_MLLM, outputs_both), dim=-1))
            return outputs_both


class KGLM(nn.Module):
    """docstring for ClassName"""
    def __init__(self, args):
        super(KGLM, self).__init__()
        # load pretrained MLLM
        self.model_name = args.model_name
        if args.model_name[-2:] == "KG":
            self.base_model = MLKGLM(args)
            load_model(self.base_model, args.modelkg_dir)
        else:
            self.base_model = AutoModel.from_pretrained(args.model_dir, 
                                                        return_dict=True,
                                                        output_hidden_states=True)
        # 64 for EA
        # self.new_all_aggregator = nn.Sequential(nn.Linear(768, 128), nn.Dropout(0.2))

    def forward(self, **inputs):
        if self.model_name[-2:] == "KG":
            outputs = self.base_model(**inputs)
        else:
            outputs = self.base_model(**inputs).hidden_states[-1]
        # outputs = torch.tanh(self.new_all_aggregator(outputs))
        outputs = torch.mean(outputs, dim=1)
        return outputs


def lossfcn(output_src, output_dst, output_neg):
    lossfcn = InfoNCE(negative_mode='unpaired')
    lossfcn_el2 = nn.MSELoss()
    lossfcn_re = nn.MSELoss()
    outputs_query = output_src
    outputs_pos = output_dst
    outputs_neg = output_neg
    # print(outputs_query.shape, outputs_pos.shape, outputs_neg.shape)
    # average
    # outputs_query = torch.mean(outputs_query, dim=1)
    # outputs_pos = torch.mean(outputs_pos, dim=1)
    # outputs_neg = torch.mean(outputs_neg, dim=1)
    # cosine loss
    loss_dp = lossfcn(outputs_query, outputs_pos, outputs_neg)
    # l2-norm loss
    loss_el2 = lossfcn_el2(outputs_query, outputs_pos) / (lossfcn_el2(outputs_query, outputs_pos) + lossfcn_el2(outputs_query, outputs_neg))
    return loss_dp + loss_el2