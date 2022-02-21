import os
import json
import torch.utils.data as Data
from tqdm import tqdm


class EntityLoader(Data.Dataset):
    def __init__(self, filepath):
        # check if data files are merged
        self.merge_data(filepath)
        count_e = 0
        with open(os.path.join(filepath, "data.json"), "r") as f:
            # get the number of entity
            for _ in tqdm(f, desc="load entity data"): count_e += 1
        self.num_e = count_e
        self.fopen = open(os.path.join(filepath, "data.json"), "r")

    def __len__(self):
        return self.num_e

    def __getitem__(self, index):
        line = self.fopen.__next__()
        # preprocess entity data
        data = self.preprocess(line)
        return data

    def merge_data(self, filepath):
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

    def preprocess(self, line):
        tmp_data = json.loads(line)

