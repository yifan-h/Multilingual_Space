import os
import json
import random
import torch
import torch.utils.data as Data
from tqdm import tqdm
from transformers import AutoTokenizer


class EntityLoader(Data.Dataset):
    def __init__(self, args):
        # load entity dictionary
        num_e = 0
        entity_dict = {}
        # load entity dict
        with open(os.path.join(args.data_dir, "entity.json"), "r") as f:
            # for line in tqdm(f, desc="load entity data"):
            for line in f:
                num_e += 1
                tmp_data = json.loads(line)
                entity_dict[tmp_data["id"]] = []
                for k, v in tmp_data["labels"].items():
                    entity_dict[tmp_data["id"]].append(v["value"])
        self.num_e = num_e
        self.neg_num = args.neg_num
        self.entity_dict = entity_dict
        self.entity_pool = set(entity_dict.keys())
        self.fopen = open(os.path.join(args.data_dir, "entity.json"), "r")
        # set tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
        self.lm_mask_token = self.tokenizer.pad_token
        self.lm_mask_token_id = self.tokenizer.pad_token_id

    def __len__(self):
        return self.num_e

    def __getitem__(self, index):
        line = self.fopen.__next__()
        return self.cleaning(line)

    def negative_sampler(self, neg_num):
        qids = random.sample(self.entity_pool, neg_num)
        return [random.choice(self.entity_dict[qid]) for qid in qids]

    def cleaning(self, tmp_data):
        tmp_data = json.loads(tmp_data)
        inputs_pos = [v["value"] for k, v in tmp_data["labels"].items()]
        # inputs_neg = self.negative_sampler(self.neg_num)
        inputs_neg = self.negative_sampler(len(inputs_pos))
        return self.tokenizer(inputs_pos+inputs_neg, padding=True, return_tensors="pt")


class TripleLoader(Data.Dataset):
    def __init__(self, args, entity_dict, triple_context=True):
        # load relation dict
        relation_dict = {}
        with open(os.path.join(args.data_dir, "relation.json"), "r") as f:
            # get the number of entity
            for line in f: 
                tmp_data = json.loads(line)
                if len(tmp_data["labels"]) < 1: continue
                relation_dict[tmp_data["id"]] = []
                for k, v in tmp_data["labels"].items():
                    relation_dict[tmp_data["id"]].append(v["value"])
        # load triple
        num_t = 0
        with open(os.path.join(args.data_dir, "triple.txt"), "r") as f:
            # for line in tqdm(f, desc="load triple data"):
            for line in f:
                triple_list = line[:-1].split("\t")
                if len(triple_list) != 3: continue
                num_t += 1
        self.triple_batch = args.batch_num
        self.num_t = num_t
        self.relation_dict = relation_dict
        self.relation_pool = set(relation_dict.keys())
        self.entity_dict = entity_dict
        self.entity_pool = set(entity_dict.keys())
        self.fopen = open(os.path.join(args.data_dir, "triple.txt"), "r")
        # set tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
        self.lm_mask_token = self.tokenizer.pad_token
        self.lm_mask_token_id = self.tokenizer.pad_token_id
        # set context
        self.triple_context = triple_context

    def __len__(self):
        return int(self.num_t/self.triple_batch)

    def __getitem__(self, index):
        c_list, o_list = [], []
        for i in range(self.triple_batch):
            line = self.fopen.__next__()
            triple_list = line[:-1].split("\t")
            if len(triple_list) != 3: continue
            s, p, o = triple_list
            if p in self.relation_dict:
                if self.triple_context:
                    c_list.append(random.choice(self.entity_dict[s])+" "+self.lm_mask_token+" "+\
                                    random.choice(self.relation_dict[p]))
                else:
                    c_list.append(random.choice(self.entity_dict[s]))
            else:
                c_list.append(random.choice(self.entity_dict[s]))
            o_list.append(random.choice(self.entity_dict[o]))
        return self.cleaning(c_list, o_list)

    def negative_sampler(self, neg_num):
        qids = random.sample(self.entity_pool, neg_num)
        return [random.choice(self.entity_dict[qid]) for qid in qids]

    def cleaning(self, c_list, o_list):
        inputs_neg = self.negative_sampler(len(c_list))
        input_tokens = self.tokenizer(c_list+o_list+inputs_neg, padding=True, return_tensors="pt")
        # set separation token [MASK] <mask> attention mask as 0
        tmp_attn_mask = torch.where(input_tokens["input_ids"]==self.lm_mask_token_id, 0, input_tokens["attention_mask"])
        input_tokens["attention_mask"] = tmp_attn_mask
        return input_tokens


def grad_parameters(model, freeze=True):
    for name, param in model.named_parameters():
        param.requires_grad = freeze
    return

def grad_universal(model, freeze=True):
    for name, param in model.named_parameters():
        if "universal" in name:
            param.requires_grad = freeze
    return

def grad_triple_encoder(model, freeze=True):
    for name, param in model.named_parameters():
        if "triple" in name:
            param.requires_grad = freeze
    return

def save_model(model, accelerator, path):
    accelerator.save(model.state_dict(), path)
    return

def load_model(model, path):
    model.load_state_dict(torch.load(path, map_location='cpu'), strict=False)
    return