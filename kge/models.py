import os
import copy
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, Conv1D

from utils import load_model

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
        self.all_aggregator = nn.Linear(3*hidden_num, hidden_num, bias=False)
        self.all_aggregator.weight.data = self.weight_init_sum(self.all_aggregator.weight.data)

    def weight_init_sum(self, t):
        hidden_num = int(t.shape[-1]/3)
        return 0.0003 + torch.cat((0.333*torch.eye(hidden_num,hidden_num),
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
        # return self.all_aggregator(torch.cat((outputs_MLLM, outputs_universal, outputs_MLKGLM), dim=-1))
        return outputs_MLLM + outputs_universal + outputs_MLKGLM


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
        self.new_all_aggregator = nn.Linear(768, 128)

    def forward(self, **inputs):
        if self.model_name[-2:] == "KG":
            outputs = self.base_model(**inputs)
        else:
            outputs = self.base_model(**inputs).hidden_states[-1]
        outputs = torch.mean(outputs, dim=1)
        outputs = F.elu(self.new_all_aggregator(outputs))
        return outputs
