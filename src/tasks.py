import os
import torch
import time
import random
from tqdm import tqdm
import torch.utils.data as Data
from transformers import AdamW, get_cosine_with_hard_restarts_schedule_with_warmup
from accelerate import Accelerator, DistributedDataParallelKwargs
from info_nce import InfoNCE

from utils import EntityLoader, TripleLoader, MixLoader, WOCLoader, WCLoader, grad_parameters, grad_kgencoder,\
                     save_model, load_model
from models import MLKGLM, loss_universal, loss_triple, loss_wocontext, fusion_adapter, simple_adapter

seed = 123
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def train_adapter_phrase(args, model_mlkg, simple=False):
    # load data, set model and optimizer
    wocontext_dataset = WOCLoader(args)
    wocontext_data = Data.DataLoader(dataset=wocontext_dataset, batch_size=1, num_workers=1)
    args.lm_mask_token_id = wocontext_dataset.lm_mask_token_id
    accelerator = Accelerator(kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
    optimizer = AdamW(model_mlkg.parameters(), lr=args.lr, eps=args.adam_epsilon, weight_decay=1e-4)
    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_cycles=5,
                                                num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.triple_epoch*len(wocontext_data)*2)
    model_mlkg, optimizer, wocontext_data = accelerator.prepare(model_mlkg, optimizer, wocontext_data)
    # training: adapter, non-context
    count_save = 0
    time_start = time.time()
    loss_list1, loss_list2 = [], []
    if simple:
        model_mlkg.module.MLLM.train_adapter("baseline")
    for e in range(args.triple_epoch):
        for encoded_inputs in wocontext_data:
            input_e1, input_e2, input_t1, input_t2 = encoded_inputs
            input_e1 = {k:torch.squeeze(v) for k, v in input_e1.items()}
            input_e2 = {k:torch.squeeze(v) for k, v in input_e2.items()}
            input_t1 = {k:torch.squeeze(v) for k, v in input_t1.items()}
            input_t2 = {k:torch.squeeze(v) for k, v in input_t2.items()}
            optimizer.zero_grad()
            #### triple
            if not simple:
                model_mlkg.module.stage = "tp"
                grad_parameters(model_mlkg, stage=model_mlkg.module.stage, fuse=False)
            outputs1 = model_mlkg(**input_t1)
            outputs2 = model_mlkg(**input_t2)
            loss = loss_wocontext(args, outputs1, outputs2)
            loss_list2.append(float(loss.data))
            accelerator.backward(loss)
            # zero grad
            optimizer.step()
            scheduler.step()
            #### entity
            if not simple:
                model_mlkg.module.stage = "ep"
                grad_parameters(model_mlkg, stage=model_mlkg.module.stage, fuse=False)
            outputs1 = model_mlkg(**input_e1)
            outputs2 = model_mlkg(**input_e2)
            loss = loss_wocontext(args, outputs1, outputs2)
            loss_list1.append(float(loss.data))
            accelerator.backward(loss)
            # zero grad
            optimizer.step()
            scheduler.step()
            # save model
            count_save += 1
            if count_save % 1e3 == 0 and accelerator.state.local_process_index == 0:
                # time
                time_length = round(time.time() - time_start, 4)
                time_start = time.time()
                # loss
                loss_avg1 = round(sum(loss_list1) / len(loss_list1), 4)
                loss_avg2 = round(sum(loss_list2) / len(loss_list2), 4)
                loss_list1, loss_list2 = [], []
                print("progress (w/o context) -- adapter: ", count_save, "/", len(wocontext_data)*args.triple_epoch, " |time: ", time_length, "s |loss (u, t): ",loss_avg1, " ", loss_avg2)
        # load data
        wocontext_dataset = WOCLoader(args)
        wocontext_data = Data.DataLoader(dataset=wocontext_dataset, batch_size=1, num_workers=1)
        wocontext_data = accelerator.prepare(wocontext_data)
    # save
    save_model(model_mlkg, accelerator, args.tmp_dir)
    del model_mlkg
    return

def train_adapter_sentence(args, model_mlkg,):
    # load data, set model and optimizer
    wcontext_dataset = WCLoader(args)
    wcontext_data = Data.DataLoader(dataset=wcontext_dataset, batch_size=1, num_workers=1)
    args.lm_mask_token_id = wcontext_dataset.lm_mask_token_id
    accelerator = Accelerator(kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
    optimizer = AdamW(model_mlkg.parameters(), lr=args.lr, eps=args.adam_epsilon, weight_decay=1e-4)
    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_cycles=5,
                                                num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.triple_epoch*len(wcontext_data)*2)
    model_mlkg, optimizer, wcontext_data = accelerator.prepare(model_mlkg, optimizer, wcontext_data)
    # training: adapter: context
    count_save = 0
    time_start = time.time()
    loss_list1, loss_list2 = [], []
    for e in range(args.triple_epoch):
        for encoded_inputs in wcontext_data:
            input_e1, input_e2, input_t1, input_t2 = encoded_inputs
            input_e1 = {k:torch.squeeze(v) for k, v in input_e1.items()}
            input_e2 = {k:torch.squeeze(v) for k, v in input_e2.items()}
            input_t1 = {k:torch.squeeze(v) for k, v in input_t1.items()}
            input_t2 = {k:torch.squeeze(v) for k, v in input_t2.items()}
            optimizer.zero_grad()
            #### triple
            if not simple:
                model_mlkg.module.stage = "ts"
                grad_parameters(model_mlkg, stage=model_mlkg.module.stage, fuse=False)
            outputs1 = model_mlkg(**input_t1)
            outputs2 = model_mlkg(**input_t2)
            loss = loss_wocontext(args, outputs1, outputs2)
            loss_list2.append(float(loss.data))
            accelerator.backward(loss)
            # zero grad
            optimizer.step()
            scheduler.step()
            #### entity
            if not simple:
                model_mlkg.module.stage = "es"
                grad_parameters(model_mlkg, stage=model_mlkg.module.stage, fuse=False)
            outputs1 = model_mlkg(**input_e1)
            outputs2 = model_mlkg(**input_e2)
            loss = loss_wocontext(args, outputs1, outputs2)
            loss_list1.append(float(loss.data))
            accelerator.backward(loss)
            # zero grad
            optimizer.step()
            scheduler.step()
            # save model
            count_save += 1
            if count_save % 1e3 == 0 and accelerator.state.local_process_index == 0:
                # time
                time_length = round(time.time() - time_start, 4)
                time_start = time.time()
                # loss
                loss_avg1 = round(sum(loss_list1) / len(loss_list1), 4)
                loss_avg2 = round(sum(loss_list2) / len(loss_list2), 4)
                loss_list1, loss_list2 = [], []
                print("progress (w context) -- adapter: ", count_save, "/", len(wcontext_data)*args.triple_epoch, " |time: ", time_length, "s |loss (u, t): ",loss_avg1, " ", loss_avg2)
        # load data
        wcontext_dataset = WCLoader(args)
        wcontext_data = Data.DataLoader(dataset=wcontext_dataset, batch_size=1, num_workers=1)
        wcontext_data = accelerator.prepare(wcontext_data)
    save_model(model_mlkg, accelerator, args.tmp_dir)
    del model_mlkg
    return


def train_fuse_phrase(args, model_mlkg):
    model_mlkg.fuse = True
    # load data, set model and optimizer
    wocontext_dataset = WOCLoader(args)
    wocontext_data = Data.DataLoader(dataset=wocontext_dataset, batch_size=1, num_workers=1)
    args.lm_mask_token_id = wocontext_dataset.lm_mask_token_id
    accelerator = Accelerator(kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
    optimizer = AdamW(model_mlkg.parameters(), lr=args.lr, eps=args.adam_epsilon, weight_decay=1e-4)
    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_cycles=5,
                                                num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.triple_epoch*len(wocontext_data)*2)
    model_mlkg, optimizer, wocontext_data = accelerator.prepare(model_mlkg, optimizer, wocontext_data)
    # training: adapter, non-context
    count_save = 0
    time_start = time.time()
    loss_list1, loss_list2 = [0.1], [0.1]
    for e in range(args.triple_epoch):
        for encoded_inputs in wocontext_data:
            if random.random() > 0.1: continue
            input_e1, input_e2, input_t1, input_t2 = encoded_inputs
            input_e1 = {k:torch.squeeze(v) for k, v in input_e1.items()}
            input_e2 = {k:torch.squeeze(v) for k, v in input_e2.items()}
            input_t1 = {k:torch.squeeze(v) for k, v in input_t1.items()}
            input_t2 = {k:torch.squeeze(v) for k, v in input_t2.items()}
            optimizer.zero_grad()
            #### triple
            model_mlkg.module.stage = "tp"
            grad_parameters(model_mlkg, stage=model_mlkg.module.stage, fuse=True)
            outputs1 = model_mlkg(**input_t1)
            outputs2 = model_mlkg(**input_t2)
            loss = loss_wocontext(args, outputs1, outputs2)
            loss_list2.append(float(loss.data))
            accelerator.backward(loss)
            # zero grad
            optimizer.step()
            scheduler.step()
            #### entity
            model_mlkg.module.stage = "ep"
            grad_parameters(model_mlkg, stage=model_mlkg.module.stage, fuse=True)
            outputs1 = model_mlkg(**input_e1)
            outputs2 = model_mlkg(**input_e2)
            loss = loss_wocontext(args, outputs1, outputs2)
            loss_list1.append(float(loss.data))
            accelerator.backward(loss)
            # zero grad
            optimizer.step()
            scheduler.step()
            # save model
            count_save += 1
            if count_save % 1e3 == 0 and accelerator.state.local_process_index == 0:
                # time
                time_length = round(time.time() - time_start, 4)
                time_start = time.time()
                # loss
                loss_avg1 = round(sum(loss_list1) / len(loss_list1), 4)
                loss_avg2 = round(sum(loss_list2) / len(loss_list2), 4)
                loss_list1, loss_list2 = [0.1], [0.1]
                print("progress (w/o context) -- fusion: ", count_save, "/", len(wocontext_data)*args.triple_epoch, " |time: ", time_length, "s |loss (u, t): ",loss_avg1, " ", loss_avg2)
        # load data
        wocontext_dataset = WOCLoader(args)
        wocontext_data = Data.DataLoader(dataset=wocontext_dataset, batch_size=1, num_workers=1)
        wocontext_data = accelerator.prepare(wocontext_data)
    # save
    save_model(model_mlkg, accelerator, args.tmp_dir, True)
    del model_mlkg
    return


def train_fuse_sentence(args, model_mlkg):
    model_mlkg.fuse = True
    # load data, set model and optimizer
    wcontext_dataset = WCLoader(args)
    wcontext_data = Data.DataLoader(dataset=wcontext_dataset, batch_size=1, num_workers=1)
    args.lm_mask_token_id = wcontext_dataset.lm_mask_token_id
    accelerator = Accelerator(kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
    optimizer = AdamW(model_mlkg.parameters(), lr=args.lr, eps=args.adam_epsilon, weight_decay=1e-4)
    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_cycles=5,
                                                num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.triple_epoch*len(wcontext_data)*2)
    model_mlkg, optimizer, wcontext_data = accelerator.prepare(model_mlkg, optimizer, wcontext_data)
    # training: adapter: context
    count_save = 0
    time_start = time.time()
    loss_list1, loss_list2 = [0.1], [0.1]
    for e in range(args.triple_epoch):
        for encoded_inputs in wcontext_data:
            if random.random() > 0.2: continue
            input_e1, input_e2, input_t1, input_t2 = encoded_inputs
            input_e1 = {k:torch.squeeze(v) for k, v in input_e1.items()}
            input_e2 = {k:torch.squeeze(v) for k, v in input_e2.items()}
            input_t1 = {k:torch.squeeze(v) for k, v in input_t1.items()}
            input_t2 = {k:torch.squeeze(v) for k, v in input_t2.items()}
            optimizer.zero_grad()
            #### triple
            model_mlkg.module.stage = "ts"
            grad_parameters(model_mlkg, stage=model_mlkg.module.stage, fuse=True)
            outputs1 = model_mlkg(**input_t1)
            outputs2 = model_mlkg(**input_t2)
            loss = loss_wocontext(args, outputs1, outputs2)
            loss_list2.append(float(loss.data))
            accelerator.backward(loss)
            # zero grad
            optimizer.step()
            scheduler.step()
            #### entity
            model_mlkg.module.stage = "es"
            grad_parameters(model_mlkg, stage=model_mlkg.module.stage, fuse=True)
            outputs1 = model_mlkg(**input_e1)
            outputs2 = model_mlkg(**input_e2)
            loss = loss_wocontext(args, outputs1, outputs2)
            loss_list1.append(float(loss.data))
            accelerator.backward(loss)
            # zero grad
            optimizer.step()
            scheduler.step()
            # save model
            count_save += 1
            if count_save % 1e3 == 0 and accelerator.state.local_process_index == 0:
                # time
                time_length = round(time.time() - time_start, 4)
                time_start = time.time()
                # loss
                loss_avg1 = round(sum(loss_list1) / len(loss_list1), 4)
                loss_avg2 = round(sum(loss_list2) / len(loss_list2), 4)
                loss_list1, loss_list2 = [0.1], [0.1]
                print("progress (w context) -- fusion: ", count_save, "/", len(wcontext_data)*args.triple_epoch, " |time: ", time_length, "s |loss (u, t): ",loss_avg1, " ", loss_avg2)
        # load data
        wcontext_dataset = WCLoader(args)
        wcontext_data = Data.DataLoader(dataset=wcontext_dataset, batch_size=1, num_workers=1)
        wcontext_data = accelerator.prepare(wcontext_data)
    save_model(model_mlkg, accelerator, args.tmp_dir, True)
    del model_mlkg
    return


'''
def print_params(model):
    for name, param in model.module.named_parameters():
        if param.requires_grad:
             print("----", name)
    print("==================================================\n")
'''


def ki_mlkg(args):
    # train adapter
    print("====> Adapter: phrase <====")
    model_mlkg = fusion_adapter(args)
    train_adapter_phrase(args, model_mlkg)
    print("====> Adapter: sentence <====")
    model_mlkg = fusion_adapter(args)
    load_model(model_mlkg, args.tmp_dir)
    train_adapter_sentence(args, model_mlkg)
    # train fusion
    args.batch_num = int(args.batch_num/2)
    args.triple_epoch = 1
    for i in range(16):
        print("====> Fusion: phrase <====")
        model_mlkg = fusion_adapter(args)
        load_model(model_mlkg, args.tmp_dir, True)
        train_fuse_phrase(args, model_mlkg)
        print("====> Fusion: sentence <====")
        model_mlkg = fusion_adapter(args)
        load_model(model_mlkg, args.tmp_dir, True)
        train_fuse_sentence(args, model_mlkg)


def ki_mlkg_baseline(args):
    # train adapter
    print("====> Adapter: phrase <====")
    model_mlkg = simple_adapter(args)
    train_adapter_phrase(args, model_mlkg, simple=True)
    print("====> Adapter: sentence <====")
    model_mlkg = simple_adapter(args)
    load_model(model_mlkg, args.tmp_dir)
    train_adapter_sentence(args, model_mlkg, simple=True)
