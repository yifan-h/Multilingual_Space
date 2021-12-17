import os
import sys
import bz2
import json
from tqdm import tqdm
from multiprocessing import Pool


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
