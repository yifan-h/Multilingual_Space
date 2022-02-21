import os
import json
import random
import torch.utils.data as Data
from tqdm import tqdm


class EntityLoader(Data.Dataset):
    def __init__(self, filepath):
        # load entity dictionary
        num_e = 0
        entity_dict, relation_dict = {}, {}
        # load entity dict
        with open(os.path.join(filepath, "data.json"), "r") as f:
            for line in tqdm(f, desc="load entity data"): 
                num_e += 1
                tmp_data = json.loads(line)
                entity_dict[tmp_data["id"]] = []
                for k, v in tmp_data["labels"].items():
                    entity_dict[tmp_data["id"]].append(v["value"])
        # load relation dict
        with open(os.path.join(filepath, "relation.json"), "r") as f:
            # get the number of entity
            for line in tqdm(f, desc="load relation data"): 
                tmp_data = json.loads(line)
                relation_dict[tmp_data["id"]] = []
                for k, v in tmp_data["labels"].items():
                    relation_dict[tmp_data["id"]].append(v["value"])
        self.num_e = num_e
        self.relation_dict = relation_dict
        self.entity_dict = entity_dict
        self.entity_pool = set(entity_dict.keys())
        # check if data files are merged
        self.preprocess_data(filepath)
        self.fopen = open(os.path.join(filepath, "data.json"), "r")

    def __len__(self):
        return self.num_e

    def __getitem__(self, index):
        line = self.fopen.__next__()
        return line

    def negative_sampler(self, neg_num):
        qids = random.sample(self.entity_pool, neg_num)
        return [random.choice(self.entity_dict[qid]) for qid in qids]

    def preprocess_data(self, filepath):
        if os.path.exists(os.path.join(filepath, "data.json")): return
        # if data is not merged, merge to get data.json
        max_triple_count, empty_subj_count = 0, 0
        triple_dict = {}
        with open(os.path.join(filepath, "triple.txt"), "r") as f_t:
            for line in tqdm(f_t, desc="merging: load triple data..."):
                if line.split("\t")[0] not in triple_dict:
                    triple_dict[line.split("\t")[0]] = []
                triple_dict[line.split("\t")[0]].append(line[:-1])
        with open(os.path.join(filepath, "entity.json"), "r") as f_e:
            with open(os.path.join(filepath, "data.json"), "w") as f_w:
                for line in tqdm(f_e, desc="merging: load entity data..."):
                    tmp_e = json.loads(line)
                    if tmp_e["id"] not in triple_dict: continue
                    tmp_e["triple"] = triple_dict[tmp_e["id"]]
                    max_triple_count = max(max_triple_count, len(tmp_e["triple"]))
                    f_w.write(json.dumps(tmp_e))
                    f_w.write("\n")
        print(max_triple_count, empty_subj_count)

    def cleaning(self, tmp_data):
        tmp_data = json.loads(tmp_data)
        tmp_label = [v["value"] for k, v in tmp_data["labels"].items()]
        tmp_triple = tmp_data["triple"]
        return tmp_label, tmp_triple
