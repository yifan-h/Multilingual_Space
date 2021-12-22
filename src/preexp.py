import os
import json
import torch
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import euclidean, cosine
from transformers import AutoTokenizer, AutoModelForMaskedLM

from models import preexp_retrieval


def open_data(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def aggregate_tokens(embed, metric="mean"):
    return np.squeeze(np.mean(embed, axis=1))


def mrr_calculator(preds, idx_test):
    preds = preds.cpu().detach().numpy()
    order = preds.argsort()
    ranks = order.argsort()
    tmp_rank = ranks[-1]
    return 1 / (len(preds) - tmp_rank)


def sim_metric(args, feat_1, feat_2, metric="cos", train_shuffle=False):
    tmp_dist = 0
    if metric == "euc":  # static distance
        for i in range(feat_1.shape[0]):
            tmp_dist += euclidean(feat_1[i], feat_2[i])
        return tmp_dist/feat_1.shape[0]
    elif metric == "cos":  # static distance
        for i in range(feat_1.shape[0]):
            tmp_dist += cosine(feat_1[i], feat_2[i])
        return tmp_dist/feat_1.shape[0]
    elif metric == "retrieval":  # mapping: feat_1=query, feat_2=key
        idx_list = np.array([i for i in range(feat_1.shape[0])])
        np.random.shuffle(idx_list)
        # data split
        idx_train = idx_list[:int(feat_1.shape[0]*0.7)]
        idx_test = idx_list[int(feat_1.shape[0]*0.7):]
        # negative sampling
        neg_sample_num = 20
        # define model
        model = preexp_retrieval(feat_1.shape[1]).to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        loss_fcn = torch.nn.BCELoss()
        # training
        model.train()
        for tmp_idx in tqdm(idx_train):
            # get query
            feat_q = feat_1[[tmp_idx]]
            feat_q = torch.FloatTensor(feat_q).to(args.device)
            # get key
            tmp_neg_idx = np.random.choice(idx_train, neg_sample_num)
            feat_k = feat_2[np.append(tmp_neg_idx, tmp_idx)]
            feat_k = torch.FloatTensor(feat_k).to(args.device)
            # get prediction
            optimizer.zero_grad()
            preds = model(feat_q, feat_k)
            labels = np.append(np.zeros(neg_sample_num), 1)
            if train_shuffle: np.random.shuffle(labels)
            labels = torch.FloatTensor(labels).to(args.device)
            loss = loss_fcn(preds, labels)
            loss.backward()
            optimizer.step()
        # test
        model.eval()
        mrr_score = []
        for tmp_idx in idx_test:
            # get query
            feat_q = feat_1[[tmp_idx]]
            feat_q = torch.FloatTensor(feat_q).to(args.device)
            # get key
            tmp_neg_idx = np.random.choice(idx_test, neg_sample_num)
            feat_k = feat_2[np.append(tmp_neg_idx, tmp_idx)]
            feat_k = torch.FloatTensor(feat_k).to(args.device)
            # get prediction
            preds = model(feat_q, feat_k)
            mrr_score.append(mrr_calculator(preds, np.append(tmp_neg_idx, tmp_idx)))
        return sum(mrr_score)/len(mrr_score)
    else:
        return


def get_embedding(args, langs, entities, entity_key="labels"):
    # load model
    if len(args.model_dir): 
        model_path = os.path.join(args.model_dir, args.simulate_model)
    else:
        model_path = args.simulate_model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForMaskedLM.from_pretrained(model_path,
                                                return_dict=True,
                                                output_hidden_states=True)
    model = model.to(args.device)
    # create empty tensor
    test_output =  model(**tokenizer("test", return_tensors='pt').to(args.device))
    feat_dim = test_output.hidden_states[0].shape[2]  # 768 or 1024
    embeddings = []
    for l in langs:
        embeddings.append(np.zeros((len(entities), feat_dim)))
    # calculate embedding
    for e_idx in tqdm(range(len(entities))):
        for l_idx in range(len(langs)):
            tmp_text = entities[e_idx][entity_key][langs[l_idx]]["value"]
            input_embed = tokenizer(tmp_text, return_tensors='pt', max_length=512)
            input_embed = input_embed.to(args.device)
            tmp_embed = model(**input_embed).hidden_states[-1]
            tmp_embed = tmp_embed.cpu().detach().numpy()
            tmp_embed = aggregate_tokens(tmp_embed)
            embeddings[l_idx][e_idx] = tmp_embed
    return embeddings


def pre_static_dist(args):
    # set languages
    langs = ["en", "de", "fr", "ar", "zh"]
    # load data
    entities = open_data(os.path.join(args.data_dir, "latest_pre.json"))

    # get embedding
    print("start to calculate embedding...")
    output_path = os.path.join("./embed_entity", args.simulate_model+".npy")
    if not os.path.exists("./embed_entity"): os.makedirs("./embed_entity")
    if not os.path.exists(output_path):
        embeddings = get_embedding(args, langs, entities)
        # check missing features
        for l_idx in range(len(langs)):
            embed = np.sum(embeddings[l_idx], axis=1)
            if 0 in embed:
                print("Warning, check embedding for language: ", langs[l_idx])
        # save embedding
        for l_idx in range(len(langs)):
            np.save(output_path, embeddings)
    else:
        embeddings = []
        tmp_embed = np.load(output_path)
        for l_idx in range(len(langs)):
            embeddings.append(tmp_embed[l_idx])

    # calculate similarity
    print("start to calculate distance...")
    dist_matrix = np.zeros((len(langs), len(langs)))
    random_baselines = np.zeros((len(langs), len(langs)))
    print(langs)
    for i in range(len(langs)):
        for j in range(len(langs)):
            dist_matrix[i,j] = sim_metric(args, embeddings[i], embeddings[j])
    print(dist_matrix)
    # calculate random shuffle distance
    for i in range(len(langs)):
        for j in range(len(langs)):
            shuffle_embedding = 0 + embeddings[j]  # hard copy
            np.random.shuffle(shuffle_embedding)
            random_baselines[i,j] = sim_metric(args, embeddings[i], shuffle_embedding)
    print(random_baselines)


def pre_mapping_dist(args):
    # set languages
    langs = ["en", "de", "fr", "ar", "zh"]
    # load data
    entities = open_data(os.path.join(args.data_dir, "latest_pre.json"))

    # get embedding
    print("start to calculate embedding...")
    output_path = os.path.join("./embed_description", args.simulate_model+".npy")
    if not os.path.exists("./embed_description"): os.makedirs("./embed_description")
    if not os.path.exists(output_path):
        embeddings = get_embedding(args, langs, entities, "descriptions")
        # check missing features
        for l_idx in range(len(langs)):
            embed = np.sum(embeddings[l_idx], axis=1)
            if 0 in embed:
                print("Warning, check embedding for language: ", langs[l_idx])
        # save embedding
        for l_idx in range(len(langs)):
            np.save(output_path, embeddings)
    else:
        embeddings = []
        tmp_embed = np.load(output_path)
        for l_idx in range(len(langs)):
            embeddings.append(tmp_embed[l_idx])

    # calculate mapping similarity
    print("start to calculate distance...")
    dist_matrix = np.zeros((len(langs), len(langs)))
    random_baselines = np.zeros((len(langs), len(langs)))
    print(langs)
    for i in range(len(langs)):
        for j in range(len(langs)):
            dist_matrix[i,j] = sim_metric(args, embeddings[i], embeddings[j], "retrieval")
    print(dist_matrix)
    # calculate random shuffle distance
    for i in range(len(langs)):
        for j in range(len(langs)):
            random_baselines[i,j] = sim_metric(args, embeddings[i], embeddings[j], "retrieval", True)
    print(random_baselines)
