import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, Conv1D
import transformers.adapters.composition as ac
from info_nce import InfoNCE


class MLKGLM(nn.Module):
    """docstring for ClassName"""
    def __init__(self, args):
        super(MLKGLM, self).__init__()
        # load pretrained MLLM
        MLLM = AutoModelForMaskedLM.from_pretrained(args.model_dir, 
                                                    return_dict=True,
                                                    output_hidden_states=True)
        # set and activate adapters
        adapters = []
        for i in range(args.adapter_num):
            adapters.append("adapter"+str(i+1))
            MLLM.add_adapter(adapters[i])
        MLLM.add_adapter_fusion(adapters)
        MLLM.active_adapters = ac.Fuse(adapters)
        self.MLLM = MLLM
        hidden_num = MLLM.get_input_embeddings().embedding_dim
        self.uni_mapping = Conv1D(hidden_num, hidden_num)

    def forward(self, inputs):
        # get MLLM output
        outputs_MLLM = self.MLLM(inputs).hidden_states
        # take average of each layer: (batch_size, sequence_length, hidden_size)
        outputs_MLLM = sum(outputs_MLLM) / len(outputs_MLLM)
        # objective 1: universal space
        outputs_uni = self.uni_mapping(outputs_MLLM)
        # objective 2: transformer layers
        return outputs_uni