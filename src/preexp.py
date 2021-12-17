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
    for i in range(feat_1.shape[0]):
        tmp_dist += cosine(feat_1[i], feat_2[i])
    return tmp_dist/feat_1.shape[0]


def pre_sim(args):
    # set languages
    langs = ["en", "de", "fr", "ar", "zh"]
    # load data
    entities = open_data(os.path.join(args.data_dir, "latest_pre.json"))
    # load model
    if len(args.model_dir): 
        model_path = os.path.join(args.model_dir, args.simulate_model)
    else:
        model_path = args.simulate_model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForMaskedLM.from_pretrained(model_path,
                                                return_dict=True,
                                                output_hidden_states=True)
    # create output folder
    if not os.path.exists("./embed"): os.makedirs("./embed")
    output_path = os.path.join("./embed", args.simulate_model)
    if not os.path.exists(output_path): os.makedirs(output_path)

    # create empty tensor
    test_output =  model(**tokenizer("test", return_tensors='pt'))
    feat_dim = test_output.hiddenstates[0].shape[2]  # 768 or 1024
    embeddings = []
    for l in langs:
        embeddings.append(np.zeros((len(entities), feat_dim)))
    # calculate embedding
    for e_idx in tqdm(range(len(entities))):
        for l_idx in range(len(langs)):
            tmp_text = entities[eidx]["labels"][langs[l_idx]]["value"]
            tmp_embed = model(**tokenizer(tmp_text, return_tensors='pt')).hidden_states[-1]
            tmp_embed = tmp_embed.cpu().detach().numpy()
            tmp_embed = aggregate_tokens(tmp_embed)
            embeddings[l_idx][e_idx] = tmp_embed
    # check missing features
    for l_idx in range(len(langs)):
        embed = np.sum(embeddings[l_idx], axis=1)
        if 0 in embed:
            print("Warning, check embedding for language: ", langs[l_idx])
    # save embedding
    for l_idx in range(len(langs)):
        np.save(os.path.join(output_path, langs[l_idx]+".npy"), embeddings)
    # calculate similarity
    dist_matrix = np.zeros((len(langs), len(langs)))
    print(langs)
    for i in range(len(langs)):
        for j in range(len(langs)):
            dist_matrix[i,j] = sim_metric(embeddings[i], embeddings[j])
    print(dist_matrix)
