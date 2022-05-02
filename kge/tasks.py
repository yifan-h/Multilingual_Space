import os
import torch
import time
from tqdm import tqdm
from transformers import AutoTokenizer
import random
import numpy as np

from utils import load_data, grad_parameters, grad_aggregator, normalize
from models import MLKGLM, KGLM, lossfcn

seed = 123
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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
    cos_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    # set tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    # get parameter set
    model = KGLM(args)
    model = KGLM(args).to(args.device)
    # optimizer = torch.optim.AdamW([{'params': base_params}, {'params': aggregator_params, 'lr': args.lr}], lr=1e-6, weight_decay=args.weight_decay)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # training and testing KG for all languages
    # dataset (id to text)
    train_list_text = []
    obj_list_train, obj_list_val = [], []
    for k, v in kgs.items():
        if "train" in v:
            for t in v["train"]:
                s, r, o = t.split("\t")
                s, r, o = int(s), int(r), int(o)
                #if k == "en":
                train_list_text.append(entities[k][s]+"\t"+relation[r]+"\t"+entities[k][o])
                obj_list_train.append(entities[k][o])
    obj_list_train = list(set(obj_list_train))
    results = []
    # epoch loop
    for e in range(args.epoch):
        # training
        random.shuffle(train_list_text)
        grad_parameters(model, True)
        loss_list = []
        for i in range(0, len(train_list_text), args.batch_num):
            # get text
            e_src = [" ".join(a.split("\t")[:2]) for a in train_list_text[i:i+args.batch_num]]
            e_dst = [a.split("\t")[-1] for a in train_list_text[i:i+args.batch_num]]
            # e_neg = random.sample(obj_list_train, int(len(e_dst)*args.neg_num))
            # get tokens
            input_src = tokenizer(e_src, padding=True, truncation=True, max_length=32, return_tensors="pt").to(args.device)
            input_dst = tokenizer(e_dst, padding=True, truncation=True, max_length=32, return_tensors="pt").to(args.device)
            # input_neg = tokenizer(e_neg, padding=True, truncation=True, max_length=32, return_tensors="pt").to(args.device)
            # get outputs
            output_src = model(**input_src)
            output_dst = model(**input_dst)
            # output_neg = model(**input_neg)
            # get loss
            loss = lossfcn(output_src, output_dst)
            loss_list.append(float(loss.data))
            # backward
            loss.backward()
            optimizer.step()
        # testing
        grad_parameters(model, False)
        print("...testing...")
        for k, v in kgs.items():
            test_list = v["test"]
            obj_pool_test = set()
            for t in test_list:
                obj_pool_test.add(t.split("\t")[-1])
            # print("The number of objects [", k, "] is: ", len(obj_pool_test))
            obj_list_test = list(obj_pool_test)
            rank_list = []
            obj_emb = torch.Tensor()
            for i in range(0, len(obj_list_test), args.batch_num*10):
                inputs = tokenizer(obj_list_test[i:i+args.batch_num*10], padding=True, truncation=True, max_length=32, return_tensors="pt").to(args.device)
                obj_emb = torch.cat((obj_emb, model(**inputs).cpu()), dim=0)
            obj_emb = normalize(obj_emb)
            for i in range(0, len(test_list), args.batch_num*10):
                e_src = [t.split("\t")[0] + " " + t.split("\t")[1] for t in test_list[i:i+args.batch_num*10]]
                e_dst = [t.split("\t")[-1] for t in test_list[i:i+args.batch_num*10]]
                input_src = tokenizer(e_src, padding=True, truncation=True, max_length=32, return_tensors="pt").to(args.device)
                output_src = model(**input_src).cpu()
                for j in range(output_src.shape[0]):
                    tmp_output = torch.unsqueeze(output_src[j], 0)
                    tmp_output = normalize(tmp_output)
                    score = torch.squeeze(torch.mm(tmp_output, torch.t(obj_emb))).numpy()  # normalize + dotproduct = cosine
                    ranks = np.argsort(np.argsort(-score))
                    rank = round(ranks[obj_list_test.index(e_dst[j])], 5)
                    rank_list.append(rank)
            count_1, count_10, mrr = 0, 0, 0 
            for r in rank_list:
                if r <= 1: count_1 += 1
                if r <= 10: count_10 += 1
                mrr += 1/(r+1)
            print("KGC: [", k, "] | test hit@1: ", round(count_1/len(test_list), 5), "hit@10: ", round(count_10/len(test_list), 5), "MRR: ", round(mrr/len(test_list), 5))
    # print("The performance (hit@1, hit@10) of language [", k, "] is: ", max(results))
    # print("The performance (hit@1, hit@10) of language [", k, "] is: ", round(count_1/len(rank_list), 4), round(count_10/len(rank_list), 4))
    return


def test_wk3l60(args):
    # load data
    aligns = load_data(args, "wk3l60")
    # print(aligns)
    '''
    print(aligns)
    {'en_de': {'test': ['Þórður guðjónsson@@@Þórður guðjónsson'], 'train': ['lewiston, idaho@@@lewiston (idaho)']}, 
    'en_fr': {'test': ['emi@@@emi group'], 'train': ['africa@@@afrique']}}
    '''
    # set loss function
    # lossfcn = torch.nn.CosineEmbeddingLoss()
    # lossfcn = InfoNCE()
    cos_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    # set tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    # get parameter set
    model = KGLM(args).to(args.device)
    # training and testing KG for all languages
    grad_parameters(model, True)
    for k, v in aligns.items():
        if "train" not in v: continue
        # set model and optimizer
        # optimizer = torch.optim.AdamW([{'params': base_params}, {'params': aggregator_params, 'lr': args.lr}], lr=args.lm_lr, weight_decay=args.weight_decay)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        # dataset
        train_list, test_list = v["train"], v["test"]
        # get object pool
        entity_pool = set()
        for e in train_list:
            entity_pool.add(e.split("@@@")[0])
            entity_pool.add(e.split("@@@")[1])
        # training, validation, testing
        results = []
        # training： 1 epoch
        random.shuffle(train_list)
        loss_list = []
        for i in range(0, len(train_list), args.batch_num):
            # get text
            e_src = [e.split("@@@")[0] for e in train_list[i: i+args.batch_num]]
            e_dst = [e.split("@@@")[1] for e in train_list[i: i+args.batch_num]]
            e_neg = random.sample(entity_pool, int(len(e_dst)*args.neg_num))
            # e_neg = random.sample(obj_pool, args.neg_num)
            # get tokens
            input_src = tokenizer(e_src, padding=True, truncation=True, max_length=32, return_tensors="pt").to(args.device)
            input_dst = tokenizer(e_dst, padding=True, truncation=True, max_length=32, return_tensors="pt").to(args.device)
            input_neg = tokenizer(e_neg, padding=True, truncation=True, max_length=32, return_tensors="pt").to(args.device)
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
    # testing
    grad_parameters(model, False)
    for k, v in aligns.items():
        if "test" not in v: continue
        test_list = v["test"]
        rank_list = []
        entity_pool_test = set()
        for e in test_list:
            entity_pool_test.add(e.split("@@@")[-1])
        entity_list = list(entity_pool_test)
        # print("The number of objects [", k, "] is: ", len(entity_list))
        test_emb = torch.Tensor()
        for i in range(0, len(entity_list), args.batch_num*10):
            inputs = tokenizer(entity_list[i: i+args.batch_num*10], padding=True, truncation=True, max_length=32, return_tensors="pt").to(args.device)
            outputs_emb = model(**inputs).cpu()
            test_emb = torch.cat((test_emb, outputs_emb), dim=0)
        test_emb = normalize(test_emb)
        for i in range(0, len(test_list), args.batch_num*10):
            e_src = [e.split("@@@")[0] for e in test_list[i:i+args.batch_num*10]]
            e_dst = [e.split("@@@")[1] for e in test_list[i:i+args.batch_num*10]]
            input_src = tokenizer(e_src, padding=True, truncation=True, max_length=32, return_tensors="pt").to(args.device)
            output_src = model(**input_src).cpu()
            # for each entity
            for j in range(output_src.shape[0]):
                tmp_output = torch.unsqueeze(output_src[j], 0)
                tmp_output = normalize(tmp_output)
                # score = torch.squeeze(cos_sim(tmp_output, test_emb)).numpy()
                score = torch.squeeze(torch.mm(tmp_output, torch.t(test_emb))).numpy()
                ranks = np.argsort(np.argsort(-score))
                rank = round(ranks[entity_list.index(e_dst[j])], 4)
                rank_list.append(rank)
        count_1, count_5, mrr = 0, 0, 0
        for r in rank_list:
            if r <= 1: count_1 += 1
            if r <= 5: count_5 += 1
            mrr += 1/(r+1)
        result = [round(count_1/len(rank_list), 5), round(count_5/len(rank_list), 5), round(mrr/len(rank_list), 5)]
        results.append(result)
        print("KGC: [", k, "] | hit@1: ", result[0], "hit@5: ", result[1], "mrr: ", result[2])
        # grad_parameters(model, True)
        # print("The performance (hit@1, hit@10) of language [", k, "] is: ", min(results))

    return