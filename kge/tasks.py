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
    model = KGLM(args).to(args.device)
    # grad_parameters(model, True)
    # set model and optimizer
    aggregator_params = list(filter(lambda kv: kv[0] in param_list, model.named_parameters()))
    base_params = list(filter(lambda kv: kv[0] not in param_list, model.named_parameters()))
    aggregator_params = [i[1]for i in aggregator_params]
    base_params = [i[1]for i in base_params]
    optimizer = torch.optim.AdamW([{'params': base_params}, {'params': aggregator_params, 'lr': args.lr}], lr=1e-6, weight_decay=args.weight_decay)
    '''
    # dataset (id to text)
    train_list_text, val_list_text, test_list_text = [], [], []
    obj_list_train, obj_list_val = [], []
    for k, v in kgs.items():
        for t in v["train"]:
            s, r, o = t.split("\t")
            s, r, o = int(s), int(r), int(o)
            #if k == "en":
            train_list_text.append(entities[k][s]+"\t"+relation[r]+"\t"+entities[k][o])
            obj_list_train.append(entities[k][o])
        for t in v["val"]:
            s, r, o = t.split("\t")
            s, r, o = int(s), int(r), int(o)
            val_list_text.append(entities[k][s]+"\t"+relation[r]+"\t"+entities[k][o])
            obj_list_val.append(entities[k][o])
        for t in v["test"]:
            s, r, o = t.split("\t")
            s, r, o = int(s), int(r), int(o)
            test_list_text.append(entities[k][s]+"\t"+relation[r]+"\t"+entities[k][o])
    # prepare val and test data
    obj_list_train = list(set(obj_list_train))
    obj_list_val = list(set(obj_list_val))
    # training, validation, testing
    results = []
    # epoch loop
    for e in range(args.epoch):
        # training
        random.shuffle(train_list_text)
        grad_parameters(model, True)
        loss_list = []
        for i in range(0, len(train_list_text), args.batch_num):
            # get text
            e_src = [" ".join(a.split("\t")[:2]) for a in train_list_text[i: i+args.batch_num]]
            e_dst = [a.split("\t")[-1] for a in train_list_text[i: i+args.batch_num]]
            e_neg = random.sample(obj_list_train, int(len(e_dst)*args.neg_num))
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
        # print("validation...", len(obj_list_val))
        # validation
        grad_parameters(model, False)
        rank_list = []
        obj_emb = torch.Tensor()
        for i in range(0, len(obj_list_val), args.batch_num*10):
            inputs = tokenizer(obj_list_val[i: i+args.batch_num*10], padding=True, truncation=True, max_length=500, return_tensors="pt").to(args.device)
            outputs_emb = model(**inputs).cpu()
            obj_emb = torch.cat((obj_emb, outputs_emb), dim=0)
        # grad_parameters(model, False)
        val_list_text_sample = random.sample(val_list_text, int(len(val_list_text)/10))
        for t in val_list_text_sample:
            e_src = " ".join(t.split("\t")[:2])
            e_dst = t.split("\t")[-1]
            input_src = tokenizer(e_src, padding=True, truncation=True, max_length=500, return_tensors="pt").to(args.device)
            output_src = model(**input_src).cpu()
            score = torch.squeeze(cos_sim(output_src, obj_emb)).numpy()  # for FT setting
            # score = torch.squeeze(torch.mm(output_src, torch.t(obj_emb))).numpy()  # for ZS setting
            ranks = np.argsort(np.argsort(-score))
            rank = ranks[obj_list_val.index(e_dst)]
            rank_list.append(rank)
        count_1_val, count_10_val = 0, 0 
        for r in rank_list:
            if r < 1: count_1_val += 1
            if r < 10: count_10_val += 1
        print("----KGC: loss: ", round(sum(loss_list)/len(loss_list), 4), \
                "| val hit@1: ", round(count_1_val/len(val_list_text_sample), 4), "hit@10: ", round(count_10_val/len(val_list_text_sample), 4))
        # testing
        # print("testing...")
        for k, v in kgs.items():
            test_list = v["test"]
            obj_pool_test = set()
            for t in test_list:
                obj_pool_test.add(entities[k][int(t.split("\t")[-1])])
            # print("The number of objects [", k, "] is: ", len(obj_pool_test))
            obj_list_test = list(obj_pool_test)
            rank_list = []
            inputs = tokenizer(obj_list_test, padding=True, truncation=True, max_length=500, return_tensors="pt").to(args.device)
            obj_emb = model(**inputs).cpu()
            for t in test_list:
                e_src = entities[k][int(t.split("\t")[0])] + " " + relation[int(t.split("\t")[1])]
                e_dst = entities[k][int(t.split("\t")[-1])]
                input_src = tokenizer(e_src, padding=True, truncation=True, max_length=500, return_tensors="pt").to(args.device)
                output_src = model(**input_src).cpu()
                score = torch.squeeze(cos_sim(output_src, obj_emb)).numpy()  # for FT setting
                # score = torch.squeeze(torch.mm(output_src, torch.t(obj_emb))).numpy()  # for ZS setting
                ranks = np.argsort(np.argsort(-score))
                rank = ranks[obj_list_test.index(e_dst)]
                rank_list.append(rank)
            count_1, count_10 = 0, 0 
            for r in rank_list:
                if r < 1: count_1 += 1
                if r < 10: count_10 += 1
            result = [(count_1_val+count_10_val)/len(val_list_text_sample), round(count_1/len(rank_list), 4), round(count_10/len(rank_list), 4)]
            results.append(result)
            print("KGC: [", k, "] | test hit@1: ", round(count_1/len(test_list), 4), "hit@10: ", round(count_10/len(test_list), 4))
    # grad_parameters(model, True)
    print("The performance (hit@1, hit@10) of language [", k, "] is: ", max(results))
    # print("The performance (hit@1, hit@10) of language [", k, "] is: ", round(count_1/len(rank_list), 4), round(count_10/len(rank_list), 4))
    return
    '''
    # training and testing KG for all languages
    for k, v in kgs.items():
        del model
        model = KGLM(args).to(args.device)
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
        max_val_loss = [1e10 for i in range(args.patience)]
        results = []
        # prepare val and test data
        obj_pool_val = set()
        for t in val_list:
            obj_pool_val.add(entities[k][int(t.split("\t")[-1])])
        obj_list_val = list(obj_pool_val)
        obj_pool_test = set()
        for t in test_list:
            obj_pool_test.add(entities[k][int(t.split("\t")[-1])])
        obj_list_test = list(obj_pool_test)
        print("The number of objects [", k, "] is: ", len(obj_pool_test))
        # epoch loop
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
                # labels = torch.Tensor([1 for _ in range(output_src.shape[0])]+[-1 for _ in range(output_src.shape[0])])
                # labels = labels.to(args.device)
                # loss = lossfcn(torch.cat((output_src, output_src), dim=0), torch.cat((output_dst, output_neg), dim=0), labels)
                # loss = lossfcn(output_src, output_dst)
                loss_list.append(float(loss.data))
                # backward
                loss.backward()
                optimizer.step()
            # validation
            grad_parameters(model, False)
            rank_list = []
            inputs = tokenizer(obj_list_val, padding=True, truncation=True, max_length=500, return_tensors="pt").to(args.device)
            obj_emb = model(**inputs).cpu()
            # grad_parameters(model, False)
            for t in val_list:
                e_src = entities[k][int(t.split("\t")[0])] + " " + relation[int(t.split("\t")[1])]
                e_dst = entities[k][int(t.split("\t")[-1])]
                input_src = tokenizer(e_src, padding=True, truncation=True, max_length=500, return_tensors="pt").to(args.device)
                output_src = model(**input_src).cpu()
                score = torch.squeeze(cos_sim(output_src, obj_emb)).numpy()  # for FT setting
                # score = torch.squeeze(torch.mm(output_src, torch.t(obj_emb))).numpy()  # for ZS setting
                ranks = np.argsort(np.argsort(-score))
                rank = ranks[obj_list_val.index(e_dst)]
                rank_list.append(rank)
            count_1_val, count_10_val = 0, 0 
            for r in rank_list:
                if r < 1: count_1_val += 1
                if r < 10: count_10_val += 1
            # early stop
            '''
            print("KGC: [", k, "] | training loss: ", round(sum(loss_list)/len(loss_list), 4), \
                                "| val loss: ", round(sum(val_loss_list)/len(val_loss_list), 4))
            if sum(val_loss_list)/len(val_loss_list) < max(max_val_loss):
                max_val_loss.remove(max(max_val_loss))
                max_val_loss.append(sum(val_loss_list)/len(val_loss_list))
            else:
                break
            '''
            # testing
            rank_list = []
            inputs = tokenizer(obj_list_test, padding=True, truncation=True, max_length=500, return_tensors="pt").to(args.device)
            obj_emb = model(**inputs).cpu()
            for t in test_list:
                e_src = entities[k][int(t.split("\t")[0])] + " " + relation[int(t.split("\t")[1])]
                e_dst = entities[k][int(t.split("\t")[-1])]
                input_src = tokenizer(e_src, padding=True, truncation=True, max_length=500, return_tensors="pt").to(args.device)
                output_src = model(**input_src).cpu()
                score = torch.squeeze(cos_sim(output_src, obj_emb)).numpy()  # for FT setting
                # score = torch.squeeze(torch.mm(output_src, torch.t(obj_emb))).numpy()  # for ZS setting
                ranks = np.argsort(np.argsort(-score))
                rank = ranks[obj_list_test.index(e_dst)]
                rank_list.append(rank)
            count_1, count_10 = 0, 0 
            for r in rank_list:
                if r < 1: count_1 += 1
                if r < 10: count_10 += 1
            # result = [round(sum(val_loss_list)/len(val_loss_list), 4), round(count_1/len(rank_list), 4), round(count_10/len(rank_list), 4)]
            # results.append(result)
            print("KGC: [", k, "] | training loss: ", round(sum(loss_list)/len(loss_list), 4), \
                        "| val hit@1: ", round(count_1_val/len(val_list), 4), "hit@10: ", round(count_10_val/len(val_list), 4), \
                        "| test hit@1: ", round(count_1/len(test_list), 4), "hit@10: ", round(count_10/len(test_list), 4))
        # grad_parameters(model, True)
        # print("The performance (hit@1, hit@10) of language [", k, "] is: ", min(results))
        # print("The performance (hit@1, hit@10) of language [", k, "] is: ", round(count_1/len(rank_list), 4), round(count_10/len(rank_list), 4))

    return


def test_wk3l60(args):
    # load data
    aligns = load_data(args, "wk3l60")
    '''
    print(aligns)
    {'en_de': {'test': ['Þórður guðjónsson@@@Þórður guðjónsson'], 'train': ['lewiston, idaho@@@lewiston (idaho)']}, 
    'en_fr': {'test': ['emi@@@emi group'], 'train': ['africa@@@afrique']}}
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
    # training and testing KG for all languages
    for k, v in aligns.items():
        del model
        model = KGLM(args).to(args.device)
        # set model and optimizer
        aggregator_params = list(filter(lambda kv: kv[0] in param_list, model.named_parameters()))
        base_params = list(filter(lambda kv: kv[0] not in param_list, model.named_parameters()))
        aggregator_params = [i[1]for i in aggregator_params]
        base_params = [i[1]for i in base_params]
        optimizer = torch.optim.AdamW([{'params': base_params}, {'params': aggregator_params, 'lr': args.lr}], lr=args.lm_lr, weight_decay=args.weight_decay)
        # optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        # dataset
        train_list, test_list = v["train"], v["test"]
        # get object pool
        entity_pool = set()
        for e in train_list+test_list:
            entity_pool.add(e.split("@@@")[0])
            entity_pool.add(e.split("@@@")[1])
        # training, validation, testing
        results = []
        # training： 1 epoch
        random.shuffle(train_list)
        grad_parameters(model, True)
        loss_list = []
        for i in range(0, len(train_list), args.batch_num):
            # get text
            e_src = [e.split("@@@")[0] for e in train_list[i: i+args.batch_num]]
            e_dst = [e.split("@@@")[1] for e in train_list[i: i+args.batch_num]]
            e_neg = random.sample(entity_pool, int(len(e_dst)*args.neg_num))
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
        # testing
        grad_parameters(model, False)
        rank_list = []
        entity_pool_test = set()
        for e in test_list:
            entity_pool_test.add(e.split("@@@")[-1])
        entity_list = list(entity_pool_test)
        print("The number of objects [", k, "] is: ", len(entity_list))
        test_emb = torch.Tensor()
        for i in range(0, len(entity_list), args.batch_num*5):
            inputs = tokenizer(entity_list[i: i+args.batch_num*5], padding=True, truncation=True, max_length=500, return_tensors="pt").to(args.device)
            outputs_emb = model(**inputs).cpu()
            test_emb = torch.cat((test_emb, outputs_emb), dim=0)
        for e in test_list:
            e_src = e.split("@@@")[0]
            e_dst = e.split("@@@")[1]
            input_src = tokenizer(e_src, padding=True, truncation=True, max_length=500, return_tensors="pt").to(args.device)
            output_src = model(**input_src).cpu()
            score = torch.squeeze(cos_sim(output_src, test_emb)).numpy()
            # score = torch.squeeze(torch.mm(output_src, torch.t(test_emb))).numpy()
            ranks = np.argsort(np.argsort(-score))
            rank = ranks[entity_list.index(e_dst)]
            rank_list.append(rank)
        count_1, count_5, mrr = 0, 0, 0
        for r in rank_list:
            if r < 1: count_1 += 1
            if r < 5: count_5 += 1
            mrr += 1/(r+1)
        result = [round(count_1/len(rank_list), 4), round(count_5/len(rank_list), 4), round(mrr/len(rank_list), 4)]
        results.append(result)
        print("KGC: [", k, "] | hit@1: ", result[0], "hit@5: ", result[1], "mrr: ", result[2])
        # grad_parameters(model, True)
        print("The performance (hit@1, hit@10) of language [", k, "] is: ", min(results))

    return