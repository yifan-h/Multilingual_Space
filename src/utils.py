import os
import json
import random
import torch
import torch.utils.data as Data
from tqdm import tqdm
from transformers import AutoTokenizer

seed = 123
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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
        # self.neg_num = args.neg_num
        self.entity_dict = entity_dict
        self.entity_pool = set(entity_dict.keys())
        self.fopen = open(os.path.join(args.data_dir, "entity.json"), "r")
        # set tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
        self.lm_mask_token = self.tokenizer.mask_token
        self.lm_mask_token_id = self.tokenizer.mask_token_id

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
        inputs_neg = self.negative_sampler(len(inputs_pos)+1)
        return self.tokenizer(inputs_pos+inputs_neg, padding=True, truncation=True, max_length=500, return_tensors="pt")


class TripleLoader(Data.Dataset):
    def __init__(self, args, entity_dict):
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
        self.lm_mask_token = self.tokenizer.mask_token
        self.lm_mask_token_id = self.tokenizer.mask_token_id

    def __len__(self):
        return int(self.num_t/self.triple_batch)

    def __getitem__(self, index):
        c_list, o_list = [], []
        for i in range(self.triple_batch):
            line = self.fopen.__next__()
            triple_list = line[:-1].split("\t")
            if len(triple_list) != 3: continue
            s, p, o = triple_list
            c_list.append(random.choice(self.entity_dict[s]))
            o_list.append(random.choice(self.entity_dict[o]))
        return self.cleaning(c_list, o_list)

    def negative_sampler(self, neg_num):
        qids = random.sample(self.entity_pool, neg_num)
        return [random.choice(self.entity_dict[qid]) for qid in qids]

    def cleaning(self, c_list, o_list):
        inputs_neg = self.negative_sampler(len(c_list))
        input_tokens = self.tokenizer(c_list+o_list+inputs_neg, padding=True, truncation=True, max_length=500, return_tensors="pt")
        return input_tokens


class MixLoader(Data.Dataset):
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
        num_t, num_s = 0, 0
        with open(os.path.join(args.data_dir, "triple.txt"), "r") as f:
            # for line in tqdm(f, desc="load triple data"):
            for line in f:
                triple_list = line[:-1].split("\t")
                if len(triple_list) != 3: continue
                num_t += 1
        with open(os.path.join(args.data_dir, "triple_des.txt"), "r") as f:
            # for line in tqdm(f, desc="load triple data"):
            for line in f:
                triple_list = line[:-1].split("\t")
                if len(triple_list) != 3: continue
                num_s += 1
        # load description
        self.des_dict = {}
        if not triple_context:
            with open(os.path.join(args.data_dir, "description.json"), "r") as f:
                # for line in tqdm(f, desc="load triple data"):
                for line in f:
                    tmp_data = json.loads(line)
                    self.des_dict[tmp_data["id"]] = tmp_data
            self.fopen = open(os.path.join(args.data_dir, "triple_des.txt"), "r")
            self.num = num_s
        else:
            self.fopen = open(os.path.join(args.data_dir, "triple.txt"), "r")
            self.num = num_t
        self.triple_batch = args.batch_num
        self.relation_dict = relation_dict
        self.entity_dict = entity_dict
        self.relation_pool = set(relation_dict.keys())
        self.entity_pool = set(entity_dict.keys())
        # set tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
        self.lm_mask_token = self.tokenizer.mask_token
        self.lm_mask_token_id = self.tokenizer.mask_token_id
        # set context: triple context or sentence context
        self.triple_context = triple_context

    def __len__(self):
        return int(self.num/self.triple_batch)

    def __getitem__(self, index):
        c_list, cl_list, o_list = [], [], []
        for i in range(self.triple_batch):
            line = self.fopen.__next__()
            triple_list = line[:-1].split("\t")
            if len(triple_list) != 3: continue
            s, p, o = triple_list
            # get predicate text
            if p in self.relation_dict:
                p_text = random.choice(self.relation_dict[p])
            else:
                p_text = ""
            # add random object label
            p_text += random.choice(self.entity_dict[random.sample(self.entity_pool, 1)[0]])
            # get context
            if self.triple_context:  # triple as context
                c_list.append(random.choice(self.entity_dict[s])+" "+self.lm_mask_token+" "+p_text)
            else:  # sentence as context
                tmp_lang = random.choice([k for k,v in self.des_dict[s]["descriptions"].items()])
                tmp_sent = self.des_dict[s]["descriptions"][tmp_lang]
                tmp_label = self.des_dict[s]["labels"][tmp_lang]
                tmp_idx = tmp_sent.find(tmp_label)
                c_list.append(tmp_sent[:tmp_idx]+" "+self.lm_mask_token+" "+tmp_sent[tmp_idx:tmp_idx+len(tmp_label)]\
                                +" "+self.lm_mask_token+" "+tmp_sent[tmp_idx+len(tmp_label):])
            cl_list.append(random.choice(self.entity_dict[s]))
            o_list.append(random.choice(self.entity_dict[o]))
        return self.cleaning(c_list, cl_list, o_list)

    def negative_sampler(self, neg_num):
        qids = random.sample(self.entity_pool, neg_num)
        return [random.choice(self.entity_dict[qid]) for qid in qids]

    def cleaning(self, c_list, cl_list, o_list):
        # entity
        inputs_neg = self.negative_sampler(len(cl_list))
        input_tokens_e = self.tokenizer(c_list+cl_list+inputs_neg, padding=True, truncation=True, max_length=500, return_tensors="pt")
        # set separation token [MASK] <mask> attention mask as 0
        tmp_attn_mask = torch.where(input_tokens_e["input_ids"]==self.lm_mask_token_id, 0, input_tokens_e["attention_mask"])
        input_tokens_e["attention_mask"] = tmp_attn_mask
        # triple
        inputs_neg = self.negative_sampler(len(c_list))
        input_tokens_t = self.tokenizer(c_list+o_list+inputs_neg, padding=True, truncation=True, max_length=500, return_tensors="pt")
        # set separation token [MASK] <mask> attention mask as 0
        tmp_attn_mask = torch.where(input_tokens_t["input_ids"]==self.lm_mask_token_id, 0, input_tokens_t["attention_mask"])
        input_tokens_t["attention_mask"] = tmp_attn_mask
        return input_tokens_e, input_tokens_t


class WOCLoader(Data.Dataset):
    def __init__(self, args):
        # load relation dict
        relation_dict = {}
        with open(os.path.join(args.data_dir, "relation.json"), "r") as f:
            for line in f: 
                tmp_data = json.loads(line)
                if len(tmp_data["labels"]) < 1: continue
                relation_dict[tmp_data["id"]] = []
                for k, v in tmp_data["labels"].items():
                    relation_dict[tmp_data["id"]].append(v["value"])
        entity_dict = {}
        # load entity dict
        with open(os.path.join(args.data_dir, "entity.json"), "r") as f:
            for line in f:
                tmp_data = json.loads(line)
                entity_dict[tmp_data["id"]] = []
                for k, v in tmp_data["labels"].items():
                    entity_dict[tmp_data["id"]].append(v["value"])
        # count triple number
        num_triple = 0
        with open(os.path.join(args.data_dir, "triple.txt"), "r") as f:
            for line in f:
                triple_list = line[:-1].split("\t")
                if len(triple_list) != 3: continue
                num_triple += 1
        self.num_triple = num_triple
        self.triple_batch = args.batch_num
        self.relation_dict = relation_dict
        self.entity_dict = entity_dict
        self.relation_pool = set(relation_dict.keys())
        self.entity_pool = set(entity_dict.keys())
        self.fopen = open(os.path.join(args.data_dir, "triple.txt"), "r")
        # set tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
        self.lm_mask_token = self.tokenizer.mask_token
        self.lm_mask_token_id = self.tokenizer.mask_token_id

    def __len__(self):
        return int(self.num_triple / self.triple_batch)

    def __getitem__(self, index):
        c_list, o_list = [], []  # length = batch_size
        e1_list, e2_list = [], []  # length = 2 * batch_size
        for i in range(self.triple_batch):
            # get a triple
            line = self.fopen.__next__()
            triple_list = line[:-1].split("\t")
            if len(triple_list) != 3: continue
            s, p, o = triple_list
            # get triple data
            p_text = ""
            if p in self.relation_dict:
                p_text += random.choice(self.relation_dict[p])
            c_list.append(random.choice(self.entity_dict[s])+" "+p_text)
            o_list.append(random.choice(self.entity_dict[o]))
            # get entity data
            s_pair = random.sample(self.entity_dict[s], 2)
            e1_list.append(s_pair[0])
            e2_list.append(s_pair[1])
            o_pair = random.sample(self.entity_dict[o], 2)
            e1_list.append(o_pair[0])
            e2_list.append(o_pair[1])
        return self.cleaning(c_list, o_list, e1_list, e2_list)

    def negative_sampler(self, neg_num):
        qids = random.sample(self.entity_pool, neg_num)
        return [random.choice(self.entity_dict[qid]) for qid in qids]

    def cleaning(self, c_list, o_list, e1_list, e2_list):
        # entity
        # inputs_neg = self.negative_sampler(len(e1_list))
        # input_e = self.tokenizer(e1_list+e2_list+inputs_neg, padding=True, truncation=True, max_length=128, return_tensors="pt")
        input_e1 = self.tokenizer(e1_list, padding=True, truncation=True, max_length=32, return_tensors="pt")
        input_e2 = self.tokenizer(e2_list, padding=True, truncation=True, max_length=32, return_tensors="pt")
        # triple
        # inputs_neg = self.negative_sampler(len(c_list))
        # input_t = self.tokenizer(c_list+o_list+inputs_neg, padding=True, truncation=True, max_length=128, return_tensors="pt")
        input_t1 = self.tokenizer(c_list, padding=True, truncation=True, max_length=32, return_tensors="pt")
        input_t2 = self.tokenizer(o_list, padding=True, truncation=True, max_length=32, return_tensors="pt")
        return input_e1, input_e2, input_t1, input_t2


class WCLoader(Data.Dataset):
    def __init__(self, args):
        # load relation dict
        relation_dict = {}
        with open(os.path.join(args.data_dir, "relation.json"), "r") as f:
            for line in f: 
                tmp_data = json.loads(line)
                if len(tmp_data["labels"]) < 1: continue
                relation_dict[tmp_data["id"]] = []
                for k, v in tmp_data["labels"].items():
                    relation_dict[tmp_data["id"]].append(v["value"])
        entity_dict = {}
        # load entity dict
        with open(os.path.join(args.data_dir, "entity.json"), "r") as f:
            for line in f:
                tmp_data = json.loads(line)
                entity_dict[tmp_data["id"]] = []
                for k, v in tmp_data["labels"].items():
                    entity_dict[tmp_data["id"]].append(v["value"])
        # load description
        des_dict = {}
        with open(os.path.join(args.data_dir, "description.json"), "r") as f:
            for line in f:
                tmp_data = json.loads(line)
                des_dict[tmp_data["id"]] = tmp_data
        # count triple number
        num_triple = 0
        with open(os.path.join(args.data_dir, "triple_en_des.json"), "r") as f:
            for line in f:
                tmp_data = json.loads(line)
                if len(tmp_data): num_triple += 1
        self.num_triple = num_triple
        self.fopen = open(os.path.join(args.data_dir, "triple_en_des.json"), "r")
        self.triple_batch = int(args.batch_num/4)
        self.relation_dict = relation_dict
        self.entity_dict = entity_dict
        self.des_dict = des_dict
        self.relation_pool = set(relation_dict.keys())
        self.entity_pool = set(des_dict.keys())
        # set tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
        self.lm_mask_token = self.tokenizer.mask_token
        self.lm_mask_token_id = self.tokenizer.mask_token_id

    def __len__(self):
        return int(self.num_triple / self.triple_batch)

    def __getitem__(self, index):
        st_list, o_list = [], []
        c_list, e_list = [], []
        for i in range(self.triple_batch):
            line = self.fopen.__next__()
            tmp_data = json.loads(line)
            s, o = tmp_data["subj_label"], tmp_data["obj_label"]
            # get triple data
            tmp_sl = tmp_data["token"]
            s_text = " ".join(tmp_sl[tmp_data["subj_start"]:tmp_data["subj_end"]+1])
            tmp_sent = " ".join(tmp_sl[:tmp_data["obj_start"]]+tmp_sl[tmp_data["obj_end"]:])
            o_text = random.choice(self.entity_dict[o])
            st_list.append(tmp_sent)
            o_list.append(o_text)
            # get entity data
            tmp_lang = random.choice([k for k,v in self.des_dict[s]["descriptions"].items()])
            tmp_label = self.des_dict[s]["labels"][tmp_lang]
            '''
            tmp_sent = self.des_dict[s]["descriptions"][tmp_lang].replace(tmp_label, \
                                                                    self.lm_mask_token+" "+tmp_label+" "+self.lm_mask_token)
            '''
            tmp_sent = self.des_dict[s]["descriptions"][tmp_lang]
            c_list.append(tmp_sent)
            e_list.append(random.choice(self.entity_dict[s]))
        return self.cleaning(c_list, e_list, st_list, o_list)

    def negative_sampler(self, neg_num):
        qids = random.sample(self.entity_pool, neg_num)
        return [random.choice(self.entity_dict[qid]) for qid in qids]

    def cleaning(self, c_list, e_list, st_list, o_list):
        '''
        # entity
        # inputs_neg = self.negative_sampler(len(c_list))
        # input_e = self.tokenizer(c_list+e_list+inputs_neg, padding=True, truncation=True, max_length=256, return_tensors="pt")
        input_e = self.tokenizer(c_list+e_list, padding=True, truncation=True, max_length=128, return_tensors="pt")
        # set separation token [MASK] <mask> attention mask as 0
        tmp_attn_mask = torch.where(input_e["input_ids"]==self.lm_mask_token_id, 0, input_e["attention_mask"])
        input_e["attention_mask"] = tmp_attn_mask
        # triple
        # inputs_neg = self.negative_sampler(len(st_list))
        # input_t = self.tokenizer(st_list+o_list+inputs_neg, padding=True, truncation=True, max_length=256, return_tensors="pt")
        input_t = self.tokenizer(st_list+o_list, padding=True, truncation=True, max_length=128, return_tensors="pt")
        # set separation token [MASK] <mask> attention mask as 0
        tmp_attn_mask = torch.where(input_t["input_ids"]==self.lm_mask_token_id, 0, input_t["attention_mask"])
        input_t["attention_mask"] = tmp_attn_mask
        return input_e, input_t
        '''
        input_e1 = self.tokenizer(c_list, padding=True, truncation=True, max_length=128, return_tensors="pt")
        input_e2 = self.tokenizer(e_list, padding=True, truncation=True, max_length=128, return_tensors="pt")
        # triple
        # inputs_neg = self.negative_sampler(len(c_list))
        # input_t = self.tokenizer(c_list+o_list+inputs_neg, padding=True, truncation=True, max_length=128, return_tensors="pt")
        input_t1 = self.tokenizer(st_list, padding=True, truncation=True, max_length=128, return_tensors="pt")
        input_t2 = self.tokenizer(o_list, padding=True, truncation=True, max_length=128, return_tensors="pt")
        return input_e1, input_e2, input_t1, input_t2


def grad_parameters(model, stage, fuse, free=True):
    # freeze all parameters
    # for name, param in model.named_parameters(): param.requires_grad = False
    # set adapter parameters
    if fuse: 
        model.module.MLLM.train_adapter_fusion(model.module.MLLM.active_adapters)
    else:
        model.module.MLLM.train_adapter(stage)
    return
    '''
    # set linear mapping
    for name, param in model.named_parameters():
        if "M"+model.module.stage in name:
            param.requires_grad = True
    return
    '''

def grad_universal(model, free=True):
    for name, param in model.named_parameters():
        if "universal" in name:
            param.requires_grad = free
    return

def grad_triple_encoder(model, free=True):
    for name, param in model.named_parameters():
        if "triple" in name:
            param.requires_grad = free
    return

def grad_kgencoder(model, free=True):
    for name, param in model.named_parameters():
        if "knowledge_mapping" in name:
            param.requires_grad = free
    return

def save_model(model, accelerator, path, fusion=False):
    model = accelerator.unwrap_model(model)
    if accelerator.state.local_process_index == 0:
        if fusion:
            model.MLLM.save_adapter_fusion(path, "ep,tp,es,ts")
        else:
            model.MLLM.save_all_adapters(path)
    # accelerator.save(model.state_dict(), path)
    return

def load_model(model, path, fusion=False, simple=False):
    if simple:
        model.MLLM.load_adapter(os.path.join(path, "baseline"))
    else:
        model.MLLM.load_adapter(os.path.join(path, "ep"))
        model.MLLM.load_adapter(os.path.join(path, "tp"))
        model.MLLM.load_adapter(os.path.join(path, "es"))
        model.MLLM.load_adapter(os.path.join(path, "ts"))
    if fusion:
        model.MLLM.load_adapter_fusion(path, "ep,tp,es,ts")
    return