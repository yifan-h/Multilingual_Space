from tqdm import tqdm
import torch.utils.data as Data

from utils import EntityLoader


def ki_mlkg(args):
    # set data loader
    ki_dataset = EntityLoader(args.data_dir)
    ki_data = Data.DataLoader(dataset=ki_dataset, batch_size=1, num_workers=1)
    # define model
    # training adapter
    for data_list in tqdm(ki_data, desc="MLKG integration: "):
        # generate postive entity (single) & triple list (pair)
        label_spos, triple_ppos = ki_dataset.cleaning(data_list[0])
        # get negative samples
        neg_pool = ki_dataset.negative_sampler(len(label_spos)*len(label_spos)+len(triple_ppos))
        print(label_spos)
        print("\n")
        print(triple_ppos)
        print("\n")
        print(neg_pool)
        print("\n")
        print(len(label_spos), len(triple_ppos), len(neg_pool))
        for k, v in ki_dataset.entity_dict.items():
            print(k, v)
            break
        for k, v in ki_dataset.relation_dict.items():
            print(k, v)
            break
        break
        # objective1: universal space
        # objective2: KI