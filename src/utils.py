import os
import json
import random
import torch.utils.data as Data
from tqdm import tqdm


class EntityLoader(Data.Dataset):
    def __init__(self, filepath):
        # load entity dictionary
        num_e = 0
        entity_dict = {}
        # load entity dict
        avg_count = 0
        with open(os.path.join(filepath, "entity.json"), "r") as f:
            for line in tqdm(f, desc="load entity data"): 
                num_e += 1
                tmp_data = json.loads(line)
                entity_dict[tmp_data["id"]] = []
                for k, v in tmp_data["labels"].items():
                    entity_dict[tmp_data["id"]].append(v["value"])
                avg_count += len(tmp_data["labels"])
        print(avg_count / num_e)
        self.num_e = num_e
        self.avg_count = avg_count / num_e
        self.entity_dict = entity_dict
        self.entity_pool = set(entity_dict.keys())
        self.fopen = open(os.path.join(filepath, "entity.json"), "r")

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
        tmp_label = [v["value"] for k, v in tmp_data["labels"].items()]
        return tmp_label


class TripleLoader(Data.Dataset):
    def __init__(self, filepath):
        # load entity dictionary
        num_t = 0
        relation_dict = {}
        # load relation dict
        with open(os.path.join(filepath, "relation.json"), "r") as f:
            # get the number of entity
            for line in tqdm(f, desc="load relation data"): 
                num_t += 1
                tmp_data = json.loads(line)
                if len(tmp_data["labels"]) < 1: continue
                relation_dict[tmp_data["id"]] = []
                for k, v in tmp_data["labels"].items():
                    relation_dict[tmp_data["id"]].append(v["value"])
        self.num_t = num_t
        self.relation_dict = relation_dict
        self.relation_pool = set(relation_dict.keys())
        self.fopen = open(os.path.join(filepath, "triple.txt"), "r")

    def __len__(self):
        return self.num_t

    def __getitem__(self, index):
        line = self.fopen.__next__()
        return self.cleaning(line)

    def negative_sampler(self, neg_num):
        qids = random.sample(self.relation_pool, neg_num)
        return [random.choice(self.relation_dict[qid]) for qid in qids]

    def cleaning(self, tmp_data):
        return tmp_data[:-1].split("\t")
