from tqdm import tqdm
import torch.utils.data as Data
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from accelerate import Accelerator

from utils import EntityLoader, TripleLoader
from models import MLKGLM


def train_adapter(args, model_mlkg, tokenizer, entity_dataset, entity_data):
    # set optimizer
    optimizer = AdamW(model_mlkg.parameters(), lr=args.lr, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps=args.warmup_steps,
                                                num_training_steps=len(entity_data))
    model_mlkg = model_mlkg.to(args.device)
    model_mlkg, optimizer, entity_data = accelerate.prepare(model_mlkg, optimizer, entity_data)
    # set parameters: autograd
    model_mlkg.grad_parameters(model_mlkg, False)
    model_mlkg.grad_adapters(model_mlkg, True)
    model_mlkg.grad_triple_encoder(model_mlkg, False)
    # disable connection with triple encoder
    model_mlkg.obj = 1
    # training
    for entity_list in tqdm(entity_data, desc="Entity integration: "):
        optimizer.zero_grade()
        # positive set input
        inputs_pos = [e[0] for e in entity_list]
        inputs_neg = entity_dataset.negative_sampler(len(inputs_pos))
        encoded_inputs = tokenizer(inputs_pos+inputs_neg, padding=True, return_tensors="pt").to(args.device)
        outputs, _ = model_mlkg(**encoded_inputs)
        # backpropogation
        loss = model_mlkg.loss_universal(outputs)
        loss.backward()
        accelerate.backward(loss)
        optimizer.step()
        scheduler.step()
    return

def train_triple_encoder(args, model_mlkg, tokenizer, triple_dataset, triple_data):
    return

def train_both_noise(args, model_mlkg, tokenizer, entity_dataset, triple_dataset, entity_data, triple_data):
    return



def ki_mlkg(args):
    # set data loader
    entity_dataset = EntityLoader(args.data_dir)
    entity_data = Data.DataLoader(dataset=entity_dataset, batch_size=1, num_workers=1)
    triple_dataset = TripleLoader(args.data_dir)
    triple_data = Data.DataLoader(dataset=triple_dataset, batch_size=10, num_workers=1)
    # define model
    model_mlkg = MLKGLM(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    accelerator = Accelerator()
    args.device = accelerator.device

    # train adapter
    model_mlkg = train_adapter(args, model_mlkg, tokenizer, entity_dataset, entity_data)
    # train triple_encoder
    model_mlkg = train_triple_encoder(args, model_mlkg, tokenizer, triple_dataset, triple_data)
    # train both with noise
    model_mlkg = train_both_noise(args, model_mlkg, tokenizer, entity_dataset, triple_dataset, entity_data, triple_data)

    for data_list in tqdm(triple_data, desc="Triple integration: "):
        print(data_list)
        print(triple_dataset.negative_sampler(10))
        break
    # objective1: universal space
    # objective2: KI