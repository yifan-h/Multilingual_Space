import os
import json
import torch
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import euclidean, cosine
from transformers import AutoTokenizer, AutoModelForMaskedLM


def open_data(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def aggregate_tokens(embed, metric="mean"):
    return np.squeeze(np.mean(embed, axis=1))


def sim_metric(feat_1, feat_2, metric="cos"):
    tmp_dist = 0
    if metric == "euc":
        for i in range(feat_1.shape[0]):
            tmp_dist += euclidean(feat_1[i], feat_2[i])
    else:
        for i in range(feat_1.shape[0]):
            tmp_dist += cosine(feat_1[i], feat_2[i])
    return tmp_dist/feat_1.shape[0]


def get_embedding(args, langs, entities):
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
            tmp_text = entities[e_idx]["labels"][langs[l_idx]]["value"]
            input_embed = tokenizer(tmp_text, return_tensors='pt')
            input_embed = input_embed.to(args.device)
            tmp_embed = model(**input_embed).hidden_states[-1]
            tmp_embed = tmp_embed.cpu().detach().numpy()
            tmp_embed = aggregate_tokens(tmp_embed)
            embeddings[l_idx][e_idx] = tmp_embed
    return embeddings

def pre_sim(args):
    # set languages
    langs = ["en", "de", "fr", "ar", "zh"]
    # load data
    entities = open_data(os.path.join(args.data_dir, "latest_pre.json"))

    # get embedding
    print("start to calculate embedding...")
    output_path = os.path.join("./embed", args.simulate_model+".npy")
    if not os.path.exists("./embed"): 
        os.makedirs("./embed")
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
            dist_matrix[i,j] = sim_metric(embeddings[i], embeddings[j])
    print(dist_matrix)
    # calculate random shuffle distance
    for i in range(len(langs)):
        for j in range(len(langs)):
            shuffle_embedding = 0 + embeddings[j]  # hard copy
            np.random.shuffle(shuffle_embedding)
            random_baselines[i,j] = sim_metric(embeddings[i], shuffle_embedding)
    print(random_baselines)
