import json
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM


def open_data(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def preexp_label(args):
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
                                                output_hidden_states = True)
    # create output folder
    output_path = os.path.join("./embed", args.simulate_model)
    if not os.path.exists(output_path): os.makedirs(output_path)

    # create empty tensor
    test_output =  model(**tokenizer("test", return_tensors='pt'))
    feat_dim = test_output.hiddenstates[0].shape[2]
    embeddings = []
    for l in langs:
        embeddings.append(torch.zeros(len(entities), feat_dim))
    # calculate embedding
    for e_idx in range(len(entities)):
        for l_idx in range(len(langs)):
            pass
