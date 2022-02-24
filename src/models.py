import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, Conv1D
import transformers.adapters.composition as ac
from info_nce import InfoNCE


class MLKGLM(nn.Module):
    """docstring for ClassName"""
    def __init__(self, args):
        super(MLKGLM, self).__init__()
        # load pretrained MLLM
        MLLM = AutoModel.from_pretrained(args.model_dir, 
                                            return_dict=True,
                                            output_hidden_states=True)
        # set pretrained MLLM
        self.obj = 2
        self.MLLM = MLLM
        hidden_num = MLLM.get_input_embeddings().embedding_dim
        # set two extra modules
        self.universal_mapping = Conv1D(hidden_num, hidden_num)
        self.triple_encoder = copy.deepcopy(MLLM.encoder.layer[-1])
        # set and activate adapters
        adapters = []
        for i in range(args.adapter_num):
            adapters.append("adapter"+str(i+1))
            MLLM.add_adapter(adapters[i])
        MLLM.add_adapter_fusion(adapters)
        MLLM.active_adapters = ac.Fuse(*adapters)
        # set loss function
        self.lossfcn_universal = InfoNCE(negative_mode='unpaired')
        self.lossfcn_triple = InfoNCE()

    def forward(self, **inputs):
        # get MLLM output
        outputs_MLLM = self.MLLM(**inputs).hidden_states
        # take average of each layer: (batch_size, sequence_length, hidden_size)
        outputs_MLLM = sum(outputs_MLLM) / len(outputs_MLLM)
        # objective 1: universal space
        outputs_universal = self.universal_mapping(outputs_MLLM)
        # objective 2: transformer layers
        if self.obj == 2:
            print("test obj2...")
            outputs_MLLM = self.triple_encoder(outputs_universal)
        return outputs_universal, outputs_MLLM

    def loss_universal(self, outputs):
        # transform set-level to sample-level
        outputs = torch.mean(outputs, dim=1)
        outputs_pos = outputs[:int(outputs.shape[0]/2)]
        outputs_neg = outputs[int(outputs.shape[0]/2):]
        idx_query, idx_pos = [], []
        for i in range(int(outputs.shape[0]/2)):
            for j in range(int(outputs.shape[0]/2)):
                if i != j:
                    idx_query.append(i)
                    idx_pos.append(j)
        return self.lossfcn_universal(outputs_pos[idx_query], outputs_pos[idx_pos], outputs_neg)

    def loss_triple(self):
        return

    def grad_parameters(self, model, freeze=True):
        for name, param in model.named_parameters():
            param.requires_grad = freeze
        return

    def grad_adapters(self, model, freeze=True):
        for name, param in model.named_parameters():
            if "adapter" in name or "universal" in name:
                param.requires_grad = freeze
        return

    def grad_triple_encoder(self, model, freeze=True):
        for name, param in model.named_parameters():
            if "triple_encoder" in name:
                param.requires_grad = freeze
        return
