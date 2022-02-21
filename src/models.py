import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM
import transformers.adapters.composition as ac


class MLKGLM(AutoModelForMaskedLM):
    """docstring for ClassName"""
    def __init__(self, args):
        super(MLKGLM, self).__init__()
        if len(args.model_dir): 
            MLLM_path = os.path.join(args.model_dir, args.simulate_model)
        else:
            MLLM_path = args.simulate_model
        # load pretrained MLLM
        MLLM = AutoModelForMaskedLM.from_pretrained(MLLM_path, 
                                                    return_dict=True,
                                                    output_hidden_states=True)
        # objective 1: set and activate adapters
        adapters = []
        for i in range(args.adapter_num):
            adapters.append("adapter"+str(i+1))
            MLLM.add_adapter(adapters[i])
        MLLM.add_adapter_fusion(adapters)
        MLLM.active_adapters = ac.Fuse(adapters)
        # objective 2: set transformer layers
        