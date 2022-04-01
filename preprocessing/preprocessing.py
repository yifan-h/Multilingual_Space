import os
import sys
import bz2
import json
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import time
from threading import Thread


def extend_kgdataset(args):
    entity_path = os.path.join(args.data_dir, "latest_all_clean.json")
    data_path = os.path.join(args.kg_dir, "alignment")
    # get all WK entity set
    entity_set_wk = set()
    with open(entity_path, "r") as f:
        for line in tqdm(f):
            tmp_data = json.loads(line)
            if "en" in tmp_data["labels"]:
                entity_set_wk.add(tmp_data["labels"]["en"]["value"].lower())
    # get all kg entity
    entity_set_kg_train, entity_set_train = set(), set()
    with open(os.path.join(data_path, "en_de_60k_train25.csv"), "r") as f:
        for line in f:
            tmp_e = line[:-1].split("@@@")[0]
            entity_set_train.add(tmp_e)
            if tmp_e in entity_set_wk:
                entity_set_kg_train.add(tmp_e)
    with open(os.path.join(data_path, "en_fr_60k_train25.csv"), "r") as f:
        for line in f:
            tmp_e = line[:-1].split("@@@")[0]
            entity_set_train.add(tmp_e)
            if tmp_e in entity_set_wk:
                entity_set_kg_train.add(tmp_e)
    entity_set_kg_test, entity_set_test = set(), set()
    with open(os.path.join(data_path, "en_de_60k_test75.csv"), "r") as f:
        for line in f:
            tmp_e = line[:-1].split("@@@")[0]
            entity_set_test.add(tmp_e)
            if tmp_e in entity_set_wk:
                entity_set_kg_test.add(tmp_e)
    with open(os.path.join(data_path, "en_fr_60k_test75.csv"), "r") as f:
        for line in f:
            tmp_e = line[:-1].split("@@@")[0]
            entity_set_test.add(tmp_e)
            if tmp_e in entity_set_wk:
                entity_set_kg_test.add(tmp_e)
    print("The aligned entity number for training and testing is: ", \
                                len(entity_set_train), " ", len(entity_set_kg_train), " and ", 
                                len(entity_set_test), " ", len(entity_set_kg_test))
    # get language statistics
    entity_count_train, entity_count_test = {}, {}
    with open(entity_path, "r") as f:
        for line in tqdm(f):
            tmp_data = json.loads(line)
            if "en" in tmp_data["labels"]:
                tmp_label = tmp_data["labels"]["en"]["value"].lower()
                if tmp_label in entity_set_train:
                    for k, v in tmp_data["labels"].items():
                        if k not in entity_count_train:
                            entity_count_train[k] = 1
                        else:
                            entity_count_train[k] = entity_count_train[k]+1
                if tmp_label in entity_set_test:
                    for k, v in tmp_data["labels"].items():
                        if k not in entity_count_test:
                            entity_count_test[k] = 1
                        else:
                            entity_count_test[k] = entity_count_test[k]+1
    print("The statistics of entity label (test) language is: ", entity_count_train, "\n", entity_count_test)
    # start to get alignment data
    lang_set = {'el': 11334, 'lt': 10744, 'sk': 11659, 'uk': 16295, 'es': 26697, 'ga': 12556, 'oc': 11648, \
                'zh': 19979, 'nl': 27907, 'ast': 18061, 'sl': 15401, 'ms': 11434, 'ro': 14583, 'da': 19085, \
                'ca': 22375, 'ar': 16733, 'pt': 22427, 'ru': 20926, 'hy': 10519, 'eo': 14016, 'et': 11556, \
                'nn': 15796, 'gl': 12487, 'cy': 13311, 'fi': 19377, 'id': 14916, 'vi': 13141, 'nb': 20400, \
                'sv': 22315, 'bg': 11029, 'eu': 16161, 'it': 25146, 'he': 14961, 'sh': 10143, 'hu': 19604, \
                'tr': 15953, 'fa': 16898, 'pl': 21628, 'sq': 13511, 'arz': 11722, 'cs': 17457, 'hr': 11081, \
                'ja': 19409, 'ko': 14009, 'sr': 11396}  # 45 languages
    # construct alignment dataset
    for k, _ in lang_set.items():
        data_path_train = os.path.join(data_path, "en_"+k+"_60k_train25.csv")
        data_path_test = os.path.join(data_path, "en_"+k+"_60k_test75.csv")
        with open(data_path_train, "w") as f_train:
            with open(data_path_test, "w") as f_test:
                with open(entity_path, "r") as f:
                    for line in tqdm(f):
                        tmp_data = json.loads(line)
                        if "en" in tmp_data["labels"]:
                            if k in tmp_data["labels"]:
                                tmp_label = tmp_data["labels"]["en"]["value"].lower()
                                if tmp_label in entity_set_train:
                                    f_train.write(tmp_label+"@@@"+tmp_data["labels"][k]["value"])
                                    f_train.write("\n")
                                elif tmp_label in entity_set_test:
                                    f_test.write(tmp_label+"@@@"+tmp_data["labels"][k]["value"])
                                    f_test.write("\n")
    return




def preprocess_des(args):
    '''
    pretrain_langs = set(["af", "an", "ar", "ast", "az", "bar", "be", "bg", "bn", "br", "bs", "ca", "ceb",
                            "cs", "cy", "da", "de", "el", "en", "es", "et", "eu", "fa", "fi", "fr", "fy",
                            "ga", "gl", "gu", "he", "hi", "hr", "hu", "hy", "id", "is", "it", "ja", "jv",
                            "ka", "kk", "kn", "ko", "la", "lb", "lt", "lv", "mk", "ml", "mn", "mr", "ms",
                            "my", "nds", "ne", "nl", "nn", "no", "oc", "pl", "pt", "ro", "ru", "scn", "sco",
                            "sh", "sk", "sl", "sq", "sr", "sv", "sw", "ta", "te", "th", "tl", "tr", "tt", 
                            "uk", "ur", "uz", "vi", "war", "zh", "zh-classical"])
    # download wikipedia pages
    import wikipediaapi
    wikipediaapi_langs = {}
    for l in pretrain_langs:
        wikipediaapi_langs[l] = wikipediaapi.Wikipedia(l)
    # get file list
    entity_path = os.path.join(args.data_dir, "entity_"+args.file_idx)
    entity_path_des = os.path.join(args.data_dir, "des", "entity_des_"+args.file_idx)
    # get description
    clean_key = set(["id", "labels", "descriptions"])
    # get description set
    with open(entity_path, "r") as r_file:
        with open(entity_path_des, "w") as wd_file:
            for line in r_file:
                curr_data = json.loads(line)
                # no description: continue
                if "labels" not in curr_data: continue
                if len(curr_data["labels"]) <= 1: continue
                # with description: write des
                tmp_data = {}
                for k, v in curr_data.items():
                    if k == "id":
                        tmp_data[k] = v
                    if k == "labels":
                        tmp_data_label = {}
                        for kk, kv in curr_data[k].items():
                            tmp_data_label[kk] = kv["value"]
                        tmp_data[k] = tmp_data_label
                tmp_data_des = {}
                for k, v in tmp_data["labels"].items():
                    wiki = wikipediaapi_langs[k]
                    page = wiki.page(v)
                    tmp_sent = wiki.extracts(page, exsentences=2)  # slow
                    if v in tmp_sent:
                        tmp_data_des[k] = tmp_sent
                tmp_data["descriptions"] = tmp_data_des
                wd_file.write(json.dumps(tmp_data))
                wd_file.write("\n")
    '''
    entity_path = os.path.join(args.data_dir, "entity_des.json")
    entity_path_half = os.path.join(args.data_dir, "entity_des_half.json")
    des_path = os.path.join(args.data_dir, "description.json")
    entity_set = set()
    with open(des_path, "w") as w_file:
        # entity_des.json
        with open(entity_path, "r") as r_file:
            for line in r_file:
                curr_data = json.loads(line)
                if len(curr_data["descriptions"])==0: continue
                w_file.write(line)
                entity_set.add(curr_data["id"])
        # entity_des_half.json
        with open(entity_path_half, "r") as r_file:
            for line in r_file:
                curr_data = json.loads(line)
                if len(curr_data["descriptions"])==0: continue
                w_file.write(line)
                entity_set.add(curr_data["id"])
    # des triple
    triple_path = os.path.join(args.data_dir, "triple.txt")
    triple_des_path = os.path.join(args.data_dir, "triple_des.txt")
    with open(triple_path, "r") as r_file:
        with open(triple_des_path, "w") as w_file:
            for line in tqdm(r_file):
                subj, pred, obj = line[:-1].split("\t")
                if subj in entity_set and obj in entity_set:
                    w_file.write(line)


def preprocess_clean(args, dataset="all"):
    # get file list
    input_path = os.path.join(args.data_dir, "latest-all.json.bz2")
    entity_path = os.path.join(args.data_dir, "entity.json")
    subentity_path = os.path.join(args.data_dir, "latest_all_clean.json")
    triple_path = os.path.join(args.data_dir, "triple.txt")
    clean_key = set(["id", "type", "datatype", "labels", "aliases", "descriptions"])
    # languages for mBERT and XLM-R
    pretrain_langs = set(["af", "an", "ar", "ast", "az", "bar", "be", "bg", "bn", "br", "bs", "ca", "ceb",
                            "cs", "cy", "da", "de", "el", "en", "es", "et", "eu", "fa", "fi", "fr", "fy",
                            "ga", "gl", "gu", "he", "hi", "hr", "hu", "hy", "id", "is", "it", "ja", "jv",
                            "ka", "kk", "kn", "ko", "la", "lb", "lt", "lv", "mk", "ml", "mn", "mr", "ms",
                            "my", "nds", "ne", "nl", "nn", "no", "oc", "pl", "pt", "ro", "ru", "scn", "sco",
                            "sh", "sk", "sl", "sq", "sr", "sv", "sw", "ta", "te", "th", "tl", "tr", "tt", 
                            "uk", "ur", "uz", "vi", "war", "zh", "zh-classical"])
    subset_langs = set(["af", "ar", "bg", "bn", "de", "el", "en", "es", "et", "eu", "fa", "fi", "fr", "he", 
                        "hi", "hu", "id", "it", "ja", "jv", "ka", "kk", "ko", "ml", "mr", "ms", "my", "nl", 
                        "pt", "ru", "sw", "ta", "te", "th", "tl", "tr", "ur", "vi", "yo", "zh", "ceb", "war"])
    if dataset == "all":
        # start to write
        entity_set = set()
        with bz2.BZ2File(input_path, "rb") as r_file:  # read file
            with open(entity_path, "w") as we_file:  # write entity
                with open(triple_path, "w") as wt_file:
                    for line in tqdm(r_file):
                        # read data
                        line = line.decode().strip()
                        if line in {"[", "]"}: continue
                        if line.endswith(","): line = line[:-1]
                        tmp_entity = json.loads(line)
                        # clean entity data
                        if tmp_entity["id"][0] != "Q": continue  # only for Q entity
                        if len(tmp_entity["labels"]) < 1: continue  # no label: skip
                        new_entity = {}
                        for k in clean_key:
                            if k not in tmp_entity: continue
                            if len(tmp_entity[k]) == 0: continue
                            if k=="labels" or k=="aliases" or k=="descriptions":
                                tmp_k = {}
                                for kk, kv in tmp_entity[k].items():
                                    if kk in pretrain_langs:
                                        tmp_k[kk] = kv
                                new_entity[k] = tmp_k
                            else:
                                new_entity[k] = tmp_entity[k]
                        # no label: skip
                        if len(new_entity["labels"]) < 1: continue
                        # check repeat entity QID
                        if new_entity["id"] in entity_set:
                            print("Repeat entity QID: ", new_entity["id"])
                        else:
                            entity_set.add(new_entity["id"])
                        # write entity data
                        we_file.write(json.dumps(new_entity))
                        we_file.write("\n")
                        # clean triples
                        if "claims" not in tmp_entity: continue
                        subj = tmp_entity["id"]
                        for k, v in tmp_entity["claims"].items():
                            pred = k
                            for o in v:
                                if "mainsnak" not in o: continue
                                if "datavalue" not in o["mainsnak"]: continue
                                if "value" not in o["mainsnak"]["datavalue"]: continue
                                if "id" not in o["mainsnak"]["datavalue"]["value"]: continue
                                if isinstance(o["mainsnak"]["datavalue"]["value"], str): continue
                                obj = o["mainsnak"]["datavalue"]["value"]["id"]
                                # check format
                                if subj[0]!="Q" or pred[0]!="P" or obj[0]!="Q":
                                    print("Error triple: ", subj, pred, obj)
                                else:
                                    wt_file.write(subj+"\t"+pred+"\t"+obj+"\n")
            # remove triples with empty entities & isolated entities
            # clean triple
            clean_triple_path = os.path.join(args.data_dir, "triple_clean.txt")
            relation_set = set()
            connect_entity_set = set()
            with open(triple_path, "r") as r_file:
                with open(clean_triple_path, "w") as w_file:
                    for line in tqdm(r_file):
                        subj, pred, obj = line[:-1].split("\t")
                        if subj in entity_set and obj in entity_set:
                            w_file.write(line)
                            relation_set.add(pred)
                            connect_entity_set.add(subj)
                            connect_entity_set.add(obj)
            print("Relation number: ", len(relation_set))
            lang_set = set()
            entity_count_wa = 0
            aliases_count = 0
            # clean entity
            clean_entity_path = os.path.join(args.data_dir, "entity_clean.json")
            with open(entity_path, "r") as r_file:
                with open(clean_entity_path, "w") as w_file:
                    for line in tqdm(r_file):
                        tmp_entity = json.loads(line)
                        if tmp_entity["id"] not in connect_entity_set: continue
                        w_file.write(line)
                        for k, _ in tmp_entity["labels"].items():
                            lang_set.add(k)
                        if "aliases" in tmp_entity:
                            entity_count_wa += 1
                            aliases_count += len(tmp_entity["aliases"])
            print(lang_set)
            print("Entity number : ", len(entity_set), entity_count_wa)
            print("Aliases number: ", aliases_count)
            print("Language number: ", len(lang_set))
    else: # get a subset  (80 languages)
        clean_key = set(["id", "labels"])
        subentity_set = set()
        with open(subentity_path, "r") as r_file:
            for line in tqdm(r_file):
                tmp_entity = json.loads(line)
                label_count = 0
                for l, _ in tmp_entity["labels"].items():
                    if l in pretrain_langs: 
                        label_count += 1
                if label_count >= 20: subentity_set.add(tmp_entity["id"])
        # clean triple
        print(len(subentity_set))
        sub_triple_path = os.path.join(args.data_dir, "triple_subset.txt")
        relation_set = set()
        connect_entity_set = set()
        with open(triple_path, "r") as r_file:
            with open(sub_triple_path, "w") as w_file:
                for line in tqdm(r_file):
                    subj, pred, obj = line[:-1].split("\t")
                    if subj in subentity_set and obj in subentity_set:
                        w_file.write(line)
                        relation_set.add(pred)
                        connect_entity_set.add(subj)
                        connect_entity_set.add(obj)
        print("Relation number: ", len(relation_set))
        lang_set = set()
        entity_count_wa = 0
        aliases_count = 0
        # clean entity
        sub_entity_path = os.path.join(args.data_dir, "entity_subset.json")
        with open(entity_path, "r") as r_file:
            with open(sub_entity_path, "w") as w_file:
                for line in tqdm(r_file):
                    tmp_entity = json.loads(line)
                    if tmp_entity["id"] not in connect_entity_set: continue
                    # only keep 42 languages
                    new_entity = {}
                    for k in clean_key:
                        if k not in tmp_entity: continue
                        if len(tmp_entity[k]) == 0: continue
                        if k ==  "aliases": continue
                        if k =="labels":
                            tmp_k = {}
                            for kk, kv in tmp_entity[k].items():
                                if kk in pretrain_langs:
                                    tmp_k[kk] = kv
                            if len(tmp_k) == 0: print("Error entity: ", tmp_entity["id"])
                            new_entity[k] = tmp_k
                        else:
                            new_entity[k] = tmp_entity[k]
                    w_file.write(json.dumps(new_entity))
                    w_file.write("\n")
                    for k, _ in tmp_entity["labels"].items():
                        lang_set.add(k)
                    if "aliases" in tmp_entity:
                        entity_count_wa += 1
                        aliases_count += len(tmp_entity["aliases"])
        print(lang_set)
        print("Entity number : ", len(subentity_set), entity_count_wa)
        print("Aliases number: ", aliases_count)
        print("Language number: ", len(lang_set))
    return


def preprocess_rlabel(args):
    from qwikidata.entity import WikidataItem, WikidataLexeme, WikidataProperty
    from qwikidata.linked_data_interface import get_entity_dict_from_api
    pretrain_langs = set(["af", "an", "ar", "ast", "az", "bar", "be", "bg", "bn", "br", "bs", "ca", "ceb",
                            "cs", "cy", "da", "de", "el", "en", "es", "et", "eu", "fa", "fi", "fr", "fy",
                            "ga", "gl", "gu", "he", "hi", "hr", "hu", "hy", "id", "is", "it", "ja", "jv",
                            "ka", "kk", "kn", "ko", "la", "lb", "lt", "lv", "mk", "ml", "mn", "mr", "ms",
                            "my", "nds", "ne", "nl", "nn", "no", "oc", "pl", "pt", "ro", "ru", "scn", "sco",
                            "sh", "sk", "sl", "sq", "sr", "sv", "sw", "ta", "te", "th", "tl", "tr", "tt", 
                            "uk", "ur", "uz", "vi", "war", "zh", "zh-classical"])
    bad_relations = ["P5326", "P5105", "P5330", "P107"]
    # get relation set
    relation_set = {}
    triple_path = os.path.join(args.data_dir, "relation_statistics.txt")
    relation_path = os.path.join(args.data_dir, "relation.json")
    '''
    with open(triple_path, "r") as r_file:
        for line in tqdm(r_file):
            subj, pred, obj = line[:-1].split("\t")
            if pred not in relation_set:
                relation_set[pred] = 1
            else:
                relation_set[pred] += 1
    print(relation_set)
    '''
    with open(triple_path, "r") as r_file:
        relation_set = json.loads(r_file.read())
    with open(relation_path, "w") as w_file:
        for rlabel in relation_set:
            tmp_r = {}
            tmp_r["id"] = rlabel
            if rlabel in bad_relations: 
                tmp_r["labels"] = {}
            else:
                rdata_dict = get_entity_dict_from_api(rlabel)
                tmp_label = {}
                for k, v in rdata_dict["labels"].items():
                    if k in pretrain_langs:
                        tmp_label[k] = v
                tmp_r["labels"] = tmp_label
            w_file.write(json.dumps(tmp_r))
            w_file.write("\n")


def preprocess_pre(args):
    langs = ["en", "de", "fr", "ar", "zh"]
    clean_key = ["id", "type", "datatype", "labels", "descriptions", "aliases"]
    input_path = os.path.join(args.data_dir, "latest-all.json.bz2")
    output_path = os.path.join(args.data_dir, "latest_pre.json")

    with bz2.BZ2File(input_path) as r_file:
        with open(output_path, "w") as w_file:
            for line in tqdm(r_file):
                line = line.decode().strip()
                if line in {"[", "]"}: continue
                if line.endswith(","): line = line[:-1]
                tmp_entity = json.loads(line)
                complete_count = 0
                for l in langs:
                    if l in tmp_entity["labels"]:
                        complete_count += 1
                    if l in tmp_entity["descriptions"]:
                        complete_count += 1
                if complete_count == 2*len(langs):
                    missing_feature = 0
                    new_entity = {}
                    for k in clean_key:
                        if k in tmp_entity:
                            if len(tmp_entity[k]) == 0:
                                missing_feature = 1
                            new_entity[k] = tmp_entity[k]
                    if len(new_entity) < 4: continue
                    if missing_feature == 1: continue
                    # write data
                    w_file.write(json.dumps(new_entity))
                    w_file.write("\n")

    return


def pre_cooccurrence(args):
    '''
    langs_sim = ["af", "da", "nl", "de", "en", "is", "lb", "no", "sv", "fy", "yi",  # group 1
            "ast", "ca", "fr", "gl", "it", "oc", "pt", "ro", "es",  # group 2
            "be", "bs", "bg", "hr", "cs", "mk", "pl", "ru", "sr", "sk", "sl", "uk"  # group 3
            "et", "fi", "hu", "lv", "lt",  # group 4
            "sq", "hy", "ka", "el",  # group 5
            "br", "ga", "gd", "cy",  # group 6
            "az", "ba", "kk", "tr", "uz",  # group 7
            "ja", "ko", "vi", "zh",  # group 8
            "bn", "gu", "hi", "kn", "mr", "ne", "or", "pa", "sd", "si", "ur", "ta",  # group 9
            "ceb", "ilo", "id", "jv", "mg", "ms", "ml", "su", "tl",  # group 10
            "my", "km", "lo", "th", "mn",  # group 11
            "ar", "he", "ps", "fa",  # group 12
            "am", "ff", "ha", "ig", "ln", "lg", "nso", "so", "sw", "ss", "tn", "wo", "xh", "yo", "zu",  # group 13
            "ht"  # group 14
            ]
    langs_xlm = ["af", "am", "ar", "as", "az", "be", "bg", "bn", "br", "bs", "ca", "cs", "cy", "da", "de", 
                 "el", "en", "eo", "es", "et", "eu", "fa", "fi", "fr", "fy", "ga", "gd", "gl", "gu", "ha",
                 "he", "hi", "hr", "hu", "hy", "id", "is", "it", "ja", "jv", "ka", "kk", "km", "kn", "ko",
                 "ku", "ky", "la", "lo", "lt", "lv", "mg", "mk", "ml", "mn", "mr", "ms", "my", "ne", "nl",
                 "no", "om", "or", "pa", "pl", "ps", "pt", "ro", "ru", "sa", "sd", "si", "sk", "sl", "so",
                 "sq", "sr", "su", "sv", "sw", "ta", "te", "th", "tl", "tr", "ug", "uk", "ur", "uz", "vi",
                 "xh", "yi", "zh"]
    langs_mbert = []
    '''
    input_path = os.path.join(args.data_dir, "latest_all_clean.json")
    output_stats_path = os.path.join(args.data_dir, "statistics", "count.json")
    output_idx_path = os.path.join(args.data_dir, "statistics", "idx.json")
    output_fig_path = os.path.join(args.data_dir, "statistics", "tsne.pdf")
    output_emb_path = os.path.join(args.data_dir, "statistics", "emb.npy")
    count_dict = {}
    cooccur_dict = {}
    sents = []
    max_length = 0
    min_count_num = 1000
    with open(input_path, "r") as f:
        for line in tqdm(f):
            tmp_data = json.loads(line)
            tmp_langs = []
            for k in tmp_data["labels"]:
                # count_dict
                if k not in count_dict:
                    count_dict[k] = 1
                else:
                    count_dict[k] += 1
                tmp_langs.append(k)
            # cooccur_dict
            for i in range(len(tmp_langs)):
                for j in range(i+1, len(tmp_langs)):
                    l1 = tmp_langs[i]
                    l2 = tmp_langs[j]
                    if l1+"_"+l2 not in cooccur_dict:
                        cooccur_dict[l1+"_"+l2] = 1
                    else:
                        cooccur_dict[l1+"_"+l2] += 1
            # sentence
            sents.append(tmp_langs)
            max_length = max(max_length, len(tmp_langs))
    print(len(cooccur_dict), len(count_dict), len(sents))
    word_model = Word2Vec(sents, 
                        vector_size=128, 
                        window=max_length, 
                        min_count=min_count_num, 
                        sg=1, 
                        hs=1, 
                        workers=20)
    # save numpy array
    save_dict = {}
    idx_dict = {}
    tmp_count = 0
    for k, v in count_dict.items():
        if v >= min_count_num:
            save_dict[k] = v
            idx_dict[k] = tmp_count
            tmp_count += 1
    with open(output_stats_path, "w") as f:
        f.write(json.dumps(save_dict))
    with open(output_idx_path, "w") as f:
        f.write(json.dumps(save_dict))
    save_feat = np.zeros((tmp_count, 128))
    for k, v in count_dict.items():
        if v >= min_count_num:
            save_feat[idx_dict[k]] = word_model.wv[k]
    # check embed
    check_embed = np.sum(save_feat, axis=1)
    if 0 in check_embed:
        print("Warning, check embedding for language")
    # save embed
    np.save(output_emb_path, save_feat)
    # plot
    data = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(save_feat)
    label = ["UNK" for i in range(tmp_count)]
    for k, v in idx_dict.items():
        label[v] = k
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    fig = plt.figure()
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 # color=plt.cm.Set1(label[i]),
                 fontdict={'size': 2})
    plt.xticks([])
    plt.yticks([])
    plt.savefig(output_fig_path, format='pdf', bbox_inches="tight")

    return
