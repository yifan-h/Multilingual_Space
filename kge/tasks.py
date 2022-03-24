import os
import torch
import time
from tqdm import tqdm
from transformers import AutoTokenizer
import random
import numpy as np
from info_nce import InfoNCE

from utils import load_data, grad_parameters, grad_aggregator
from models import MLKGLM, KGLM


def test_dbp5l(args):
    # load data
    entities, relation, kgs, aligns = load_data(args, "dbp5l")
    entity_pool = set()
    for k, v in entities.items():
        for vk, vv in v.items():
            entity_pool.add(vv)
    '''
    print(entities, relation, kgs, aligns)
    {'de': {0: '(500)_Days_of_Summer'}, 'el': {0: '(500)_Μέρες_με_τη_Σάμερ'}, 'en': {0: '500_Days_of_Summer'}, 'es': {0: '(500)_Days_of_Summer'}, 'fr': {0: '(500)_jours_ensemble'}, 'ja': {0: '(500)日のサマー'}} 
    {0: 'nfcchampion'} 
    {'el': {'train': ['4742\t419\t766'], 'val': ['1352\t358\t4073'], 'test': ['4045\t683\t4044']}, 'en': {'train': ['1938\t448\t1702'], 'val': ['12397\t150\t6235'], 'test': ['12461\t708\t7662']}, 'es': {'train': ['5789\t358\t10613'], 'val': ['749\t394\t7700'], 'test': ['632\t339\t2797']}, 'fr': {'train': ['11149\t289\t890'], 'val': ['8449\t352\t2422'], 'test': ['3815\t351\t1830']}, 'ja': {'train': ['4897\t284\t2664'], 'val': ['2218\t351\t3391'], 'test': ['3189\t225\t425']}} 
    {'el-en': ['3931\t3931'], 'el-es': ['1236\t1224'], 'el-fr': ['2844\t2825'], 'el-ja': ['3902\t3827'], 'en-fr': ['3436\t3414'], 'es-en': ['1653\t1669'], 'es-fr': ['6212\t6368'], 'ja-en': ['3072\t3119'], 'ja-es': ['1002\t1011'], 'ja-fr': ['2300\t2319']}
    '''
    # set loss function
    # lossfcn = torch.nn.CosineEmbeddingLoss()
    lossfcn = InfoNCE(negative_mode='unpaired')
    # lossfcn = InfoNCE()
    cos_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    # set tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    # get parameter set
    model = KGLM(args)
    param_list = []
    for name, param in model.named_parameters():
        if "new_all_aggregator" in name:
            param_list.append(name)
    '''
    # training alignment
    for k, v in aligns.items():
        # preprocess data
        l_src, l_dst = k.split("-")
        for i in range(0, len(v), args.batch_num):
            # get entity id
            a_src = [int(a.split("\t")[0]) for a in v[i: i+args.batch_num]]
            a_dst = [int(a.split("\t")[1]) for a in v[i: i+args.batch_num]]
            # get entity label
            e_src = [entities[l_src][a] for a in a_src]
            e_dst = [entities[l_dst][a] for a in a_dst]
            e_neg = random.sample(entity_pool, len(e_dst)*args.neg_num)
            # get tokens
            input_src = tokenizer(e_src, padding=True, truncation=True, max_length=500, return_tensors="pt").to(args.device)
            input_dst = tokenizer(e_dst, padding=True, truncation=True, max_length=500, return_tensors="pt").to(args.device)
            input_neg = tokenizer(e_neg, padding=True, truncation=True, max_length=500, return_tensors="pt").to(args.device)
            # get outputs
            output_src = model(**input_src)
            output_dst = model(**input_dst)
            output_neg = model(**input_neg)
            # get loss
            # target = torch.Tensor([1 for _ in range(output_dst.shape[0])]+[-1 for _ in range(output_neg.shape[0])]).to(args.device)
            # loss = lossfcn(output_src.repeat((1+args.neg_num), 1), torch.cat((output_dst, output_neg), dim=0), target)
            loss = lossfcn(output_src, output_dst, output_neg)
            # backward
            loss.backward()
            optimizer.step()
            print("Alignment: ", k, ": ", round(float(loss.data), 4))
    '''
    # training and testing KG for all languages
    for k, v in kgs.items():
        del model
        model = KGLM(args).to(args.device)
        # grad_parameters(model, True)
        # set model and optimizer
        aggregator_params = list(filter(lambda kv: kv[0] in param_list, model.named_parameters()))
        base_params = list(filter(lambda kv: kv[0] not in param_list, model.named_parameters()))
        aggregator_params = [i[1]for i in aggregator_params]
        base_params = [i[1]for i in base_params]
        optimizer = torch.optim.AdamW([{'params': base_params}, {'params': aggregator_params, 'lr': args.lr}], lr=1e-6, weight_decay=args.weight_decay)
        # optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        # dataset
        train_list, val_list, test_list = v["train"], v["val"], v["test"]
        # get object pool
        obj_pool = set()
        for t in train_list+test_list:
            obj_pool.add(entities[k][int(t.split("\t")[-1])])
        # training, validation, testing
        grad_parameters(model, False)
        max_val_loss = [1e10 for i in range(args.patience)]
        for e in range(args.epoch):
            # training
            random.shuffle(train_list)
            grad_parameters(model, True)
            loss_list = []
            for i in range(0, len(train_list), args.batch_num):
                # get text
                e_src = [entities[k][int(a.split("\t")[0])] + " " + relation[int(a.split("\t")[1])] for a in train_list[i: i+args.batch_num]]
                e_dst = [entities[k][int(a.split("\t")[2])] for a in train_list[i: i+args.batch_num]]
                e_neg = random.sample(obj_pool, int(len(e_dst)*args.neg_num))
                # e_neg = random.sample(obj_pool, args.neg_num)
                # get tokens
                input_src = tokenizer(e_src, padding=True, truncation=True, max_length=500, return_tensors="pt").to(args.device)
                input_dst = tokenizer(e_dst, padding=True, truncation=True, max_length=500, return_tensors="pt").to(args.device)
                input_neg = tokenizer(e_neg, padding=True, truncation=True, max_length=500, return_tensors="pt").to(args.device)
                # get outputs
                output_src = model(**input_src)
                output_dst = model(**input_dst)
                output_neg = model(**input_neg)
                # get loss
                loss = lossfcn(output_src, output_dst, output_neg)
                loss_list.append(float(loss.data))
                # backward
                loss.backward()
                optimizer.step()
            # validation
            grad_parameters(model, False)
            val_loss_list = []
            # grad_parameters(model, False)
            for i in range(0, len(val_list), args.batch_num):
                # get text
                e_src = [entities[k][int(a.split("\t")[0])] + " " + relation[int(a.split("\t")[1])] for a in val_list[i: i+args.batch_num]]
                e_dst = [entities[k][int(a.split("\t")[2])] for a in val_list[i: i+args.batch_num]]
                e_neg = random.sample(obj_pool, int(len(e_dst)*args.neg_num))
                # get tokens
                input_src = tokenizer(e_src, padding=True, truncation=True, max_length=500, return_tensors="pt").to(args.device)
                input_dst = tokenizer(e_dst, padding=True, truncation=True, max_length=500, return_tensors="pt").to(args.device)
                input_neg = tokenizer(e_neg, padding=True, truncation=True, max_length=500, return_tensors="pt").to(args.device)
                # get outputs
                output_src = model(**input_src)
                output_dst = model(**input_dst)
                output_neg = model(**input_neg)
                # get loss
                loss = lossfcn(output_src, output_dst, output_neg)
                val_loss_list.append(float(loss.data))
            # early stop
            print("KGC: [", k, "] | training loss: ", round(sum(loss_list)/len(loss_list), 4), \
                                "| val loss: ", round(sum(val_loss_list)/len(val_loss_list), 4))
            if sum(val_loss_list)/len(val_loss_list) < max(max_val_loss):
                max_val_loss.remove(max(max_val_loss))
                max_val_loss.append(sum(val_loss_list)/len(val_loss_list))
            else:
                break
        # testing
        rank_list = []
        obj_pool_test = set()
        for t in test_list:
            obj_pool_test.add(entities[k][int(t.split("\t")[-1])])
        obj_list = list(obj_pool_test)
        # print("The number of objects [", k, "] is: ", len(obj_list))
        inputs = tokenizer(obj_list, padding=True, truncation=True, max_length=500, return_tensors="pt").to(args.device)
        obj_emb = model(**inputs).cpu()
        for t in test_list:
            e_src = entities[k][int(t.split("\t")[0])] + " " + relation[int(t.split("\t")[1])]
            e_dst = entities[k][int(t.split("\t")[-1])]
            input_src = tokenizer(e_src, padding=True, truncation=True, max_length=500, return_tensors="pt").to(args.device)
            output_src = model(**input_src).cpu()
            # score = torch.squeeze(cos_sim(output_src, obj_emb)).numpy()  # for FT setting
            score = torch.squeeze(torch.mm(output_src, torch.t(obj_emb))).numpy()  # for ZS setting
            ranks = np.argsort(np.argsort(-score))
            rank = ranks[obj_list.index(e_dst)]
            rank_list.append(rank)
        count_1, count_10 = 0, 0 
        for r in rank_list:
            if r < 1: count_1 += 1
            if r < 10: count_10 += 1
        # grad_parameters(model, True)
        print("The performance (hit@1, hit@10) of language [", k, "] is: ", round(100*count_1/len(rank_list), 4), " and " ,
                                                                            round(100*count_10/len(rank_list), 4))

    return
