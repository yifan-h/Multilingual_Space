from tqdm import tqdm
import torch.utils.data as Data
from transformers import AutoTokenizer

from utils import EntityLoader, TripleLoader
from models import MLKGLM


def ki_mlkg(args):
    # set data loader
    entity_dataset = EntityLoader(args.data_dir)
    entity_data = Data.DataLoader(dataset=entity_dataset, batch_size=1, num_workers=1)
    triple_dataset = TripleLoader(args.data_dir)
    triple_data = Data.DataLoader(dataset=triple_dataset, batch_size=10, num_workers=1)
    # define model
    KGLM = MLKGLM(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    # training adapter
    for entity_list in tqdm(entity_data, desc="Entity integration: "):
        encoded_inputs = tokenizer([e[0] for e in entity_list], padding=True, return_tensors="pt")
        print(encoded_inputs)
        outputs = MLKGLM(**encoded_inputs)
        print(outputs)
        break
    for data_list in tqdm(triple_data, desc="Triple integration: "):
        print(data_list)
        print(triple_dataset.negative_sampler(10))
        break
    # objective1: universal space
    # objective2: KI