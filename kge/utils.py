import os
import json
import random
import torch
from tqdm import tqdm

def load_data(args, data_name):
    data_path = os.path.join(args.data_dir, data_name)
    if data_name == "dbp5l":
        # load relation
        relation_path = os.path.join(data_path, "relations.txt")
        relation_dict = {}
        count = 0
        with open(relation_path, "r") as f:
            for line in f:
                tmp_rlabel = line[:-1].split("/")[-1]
                relation_dict[count] = tmp_rlabel
                count += 1
        # load entities
        entities_path = os.path.join(data_path, "entity")
        langs = ["de", "el", "en", "es", "fr", "ja"]
        entities = {}
        for l in langs:
            tmp_edict = {}
            entity_path = os.path.join(entities_path, l+".tsv")
            count = 0
            with open(entity_path, "r") as f:
                for line in f:
                    tmp_elabel = line[:-1].split("/")[-1]
                    tmp_edict[count] = tmp_elabel
                    count += 1
            entities[l] = tmp_edict
        # load KG
        kgs_path = os.path.join(data_path, "kg")
        langs = ["el", "en", "es", "fr", "ja"]
        kgs = {}
        for l in langs:
            tmp_kgdict = {}
            kg_path_train = os.path.join(kgs_path, l+"-train.tsv")
            kg_path_val = os.path.join(kgs_path, l+"-val.tsv")
            kg_path_test = os.path.join(kgs_path, l+"-test.tsv")
            tmp_list = []
            with open(kg_path_train, "r") as f:
                for line in f:
                    tmp_list.append(line[:-1])
            tmp_kgdict["train"] = tmp_list
            tmp_list = []
            with open(kg_path_val, "r") as f:
                for line in f:
                    tmp_list.append(line[:-1])
            tmp_kgdict["val"] = tmp_list
            tmp_list = []
            with open(kg_path_test, "r") as f:
                for line in f:
                    tmp_list.append(line[:-1])
            tmp_kgdict["test"] = tmp_list
            kgs[l] = tmp_kgdict
        # load alignment
        align_path = os.path.join(data_path, "seed_alignlinks")
        aligns = {}
        files = ["el-en.tsv", "el-es.tsv", "el-fr.tsv", "el-ja.tsv", "en-fr.tsv", \
                    "es-en.tsv", "es-fr.tsv", "ja-en.tsv", "ja-es.tsv", "ja-fr.tsv"]
        for file in files:
            file_path = os.path.join(align_path, file)
            align_list = []
            with open(file_path, "r") as f:
                for line in f:
                    s, t = line[:-1].split("\t")
                    align_list.append(str(int(float(s)))+"\t"+str(int(float(t))))
            aligns[file[:-4]] = align_list
        return entities, relation_dict, kgs, aligns
    else:  # wk3l60
        data_path = os.path.join(args.data_dir, "wk3l60")
        # load alignments
        files = ["en_de_60k_test75.csv", "en_de_60k_train25.csv", "en_fr_60k_test75.csv", "en_fr_60k_train25.csv"]
        aligns = {}
        for file in files:
            tmp_align = []
            with open(os.path.join(data_path, "alignment", file), "r") as f:
                for line in f:
                    tmp_align.append(line[:-1])
            # get language
            if "en_de" in file:
                k = "en_de"
            else:
                k = "en_fr"
            if k not in aligns:
                aligns[k]={}
            # get train/test
            if "train" in file:
                kk = "train"
            else:
                kk = "test"
            aligns[k][kk] = tmp_align
        return aligns

def grad_parameters(model, free=True):
    for name, param in model.named_parameters():
        param.requires_grad = free
    return

def grad_aggregator(model, free=True):
    for name, param in model.named_parameters():
        if "aggregator" in name:
            param.requires_grad = free
    return

def save_model(model, accelerator, path):
    model = accelerator.unwrap_model(model)
    accelerator.save(model.state_dict(), path)
    return

def load_model(model, path):
    model.load_state_dict(torch.load(path, map_location='cpu'), strict=False)
    return