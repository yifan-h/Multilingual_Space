import torch.utils.data as Data

from utils import EntityLoader


def ki_mlkg(args):
    # set data loader
    ki_dataset = EntityLoader(filepath = args.data_dir)
    ki_data = Data.DataLoader(dataset=ki_dataset, batch_size=1, num_workers=1)
    # define model
    # training adapter
    for b in ki_data:
        print(b)