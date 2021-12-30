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


def multithread_write(idx, n, file_list, input_folder, output_folder):
    clean_key = ["id", "type", "datatype", "labels", "descriptions", "aliases"]
    for input_path in file_list[idx::n]:
        if not os.path.exists(output_folder): os.makedirs(output_folder)
        target_path = input_path.replace(input_folder, output_folder)
        # start to write
        with bz2.BZ2File(input_path) as r_file:
            with open(target_path, "w") as w_file:
                for line in tqdm(r_file):
                    # read data
                    line = line.decode().strip()
                    if line in {"[", "]"}: continue
                    if line.endswith(","): line = line[:-1]
                    tmp_entity = json.loads(line)
                    # clean data
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


def preprocess_clean(args):
    # get file list
    input_folder = os.path.join(args.data_dir, "latest-all.json.bz2")
    output_folder = os.path.join(args.data_dir, "latest_all_clean.json")
    file_list = []
    for path, _, filenames in os.walk(input_folder):
        for filename in filenames:
            file_list.append(os.path.join(path, filename))
    # print(file_list)
    # define multi-thread
    n = 1
    p = Pool(n)
    for i in range(n):
        p.apply_async(multithread_write, args=(i, n, file_list, input_folder, output_folder))
    p.close()
    p.join()


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


def pre_cooccurrence(args):
    input_path = os.path.join(args.data_dir, "latest_all_clean.json")
    output_stats_path = os.path.join(args.data_dir, "statistics", "count.json")
    output_idx_path = os.path.join(args.data_dir, "statistics", "idx.json")
    output_fig_path = os.path.join(args.data_dir, "statistics", "tsne.pdf")
    output_emb_path = os.path.join(args.data_dir, "statistics", "emb.npy")
    count_dict = {}
    cooccur_dict = {}
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
    print(cooccur_dict, count_dict)
    sents = []
    for k, v in tqdm(cooccur_dict.items()):
        for i in range(v):
            sents.append(k.split("_"))
    word_model = local_model = Word2Vec(sents, 
                                        vector_size=128, 
                                        window=2, 
                                        min_count=10, 
                                        sg=1, 
                                        hs=1, 
                                        workers=20)
    # save numpy array
    save_dict = {}
    idx_dict = {}
    tmp_count = 0
    for k, v in count_dict.items():
        if v >= 10:
            save_dict[k] = v
            idx_dict[k] = tmp_count
            tmp_count += 1
    with open(output_stats_path, "w") as f:
        f.write(json.dumps(save_dict))
    with open(output_idx_path, "w") as f:
        f.write(json.dumps(save_dict))
    save_feat = np.zeros((tmp_count, 128))
    for k, v in count_dict.items():
        if v >= 10:
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
