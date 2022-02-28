import os
import torch
import time
from tqdm import tqdm
import torch.utils.data as Data
from transformers import AdamW, get_linear_schedule_with_warmup
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from info_nce import InfoNCE

from utils import EntityLoader, TripleLoader, grad_parameters, grad_adapters, grad_triple_encoder, save_model, load_model
from models import MLKGLM, loss_universal, loss_triple


def train_adapter(args, model_mlkg):
    # load data
    entity_dataset = EntityLoader(args)
    entity_data = Data.DataLoader(dataset=entity_dataset, batch_size=1, num_workers=1)
    # set parameters: autograd
    grad_parameters(model_mlkg, False)
    grad_adapters(model_mlkg, True)
    grad_triple_encoder(model_mlkg, False)
    # set model and optimizer
    accelerator = Accelerator(kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
    optimizer = AdamW(model_mlkg.parameters(), lr=args.lr, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps=args.warmup_steps,
                                                num_training_steps=len(entity_data))
    # model_mlkg = model_mlkg.to(args.device)
    model_mlkg, optimizer, entity_data = accelerator.prepare(model_mlkg, optimizer, entity_data)
    # set loss function
    lossfcn_universal = InfoNCE(negative_mode='unpaired')
    # disable connection with triple encoder
    model_mlkg.obj = 1
    # training
    count_save = 0
    time_start = time.time()
    loss_list = []
    for encoded_inputs in entity_data:
        encoded_inputs = {k:torch.squeeze(v) for k, v in encoded_inputs.items()}
        model_mlkg.zero_grad()
        # positive set input
        outputs, _ = model_mlkg(**encoded_inputs)
        # backpropogation
        loss = loss_universal(args, outputs, lossfcn_universal)
        loss_list.append(float(loss.data))
        # loss.backward()
        accelerator.backward(loss)
        optimizer.step()
        scheduler.step()
        # save model
        count_save += 1
        if count_save % 1e5 == 0:
            save_model(model_mlkg, accelerator, os.path.join(args.tmp_dir, "adapters_"+str(int(count_save/1e5))+".pt"))
        if count_save % 1e2 == 0:
            # time
            time_length = round(time.time() - time_start, 4)
            time_start = time.time()
            # loss
            loss_avg = round(sum(loss_list) / len(loss_list), 4)
            loss_list = []
            # print
            print("progress (adapter): ", count_save, " / ", len(entity_data), " | time: ", time_length, "s | loss: ",loss_avg)
    # save model
    save_model(model_mlkg, accelerator, os.path.join(args.tmp_dir, "final_v1.pt"))
    del model_mlkg
    return


def train_triple_encoder(args, model_mlkg):
    # load data
    entity_dataset = EntityLoader(args)
    entity_data = Data.DataLoader(dataset=entity_dataset, batch_size=1, num_workers=1)
    triple_dataset = TripleLoader(args, entity_dataset.entity_dict)
    triple_data = Data.DataLoader(dataset=triple_dataset, batch_size=1, num_workers=1)
    # set parameters: autograd
    grad_parameters(model_mlkg, False)
    grad_adapters(model_mlkg, False)
    grad_triple_encoder(model_mlkg, True)
    # set model and optimizer
    accelerator = Accelerator(kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
    optimizer = AdamW(model_mlkg.parameters(), lr=args.lr, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.epoch*len(triple_data))
    # model_mlkg = model_mlkg.to(args.device)
    model_mlkg, optimizer, triple_data = accelerator.prepare(model_mlkg, optimizer, triple_data)
    # set loss function
    lossfcn_triple = InfoNCE()
    # enable connection with triple encoder
    model_mlkg.obj = 2
    # training
    count_save = 0
    time_start = time.time()
    loss_list = []
    for e in range(args.epoch):
        for encoded_inputs in triple_data:
            encoded_inputs = {k:torch.squeeze(v) for k, v in encoded_inputs.items()}
            optimizer.zero_grad()
            # positive set input
            _, outputs = model_mlkg(**encoded_inputs)
            # backpropogation
            loss = loss_triple(outputs, lossfcn_triple)
            loss_list.append(float(loss.data))
            # loss.backward()
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            # save model
            count_save += 1
            if count_save % 1e5 == 0:
                save_model(model_mlkg, accelerator, os.path.join(args.tmp_dir, "triple_encoder_"+str(int(count_save/1e5))+".pt"))
            if count_save % 1e2 == 0:
                # time
                time_length = round(time.time() - time_start, 4)
                time_start = time.time()
                # loss
                loss_avg = round(sum(loss_list) / len(loss_list), 4)
                loss_list = []
                # print
                print("progress (encoder): ", count_save, " / ", len(triple_data)*args.epoch, " | time: ", time_length, "s | loss: ",loss_avg)
        # load data
        entity_dataset = EntityLoader(args)
        entity_data = Data.DataLoader(dataset=entity_dataset, batch_size=1, num_workers=1)
        triple_dataset = TripleLoader(args, entity_dataset.entity_dict)
        triple_data = Data.DataLoader(dataset=triple_dataset, batch_size=1, num_workers=1)
        # model_mlkg = model_mlkg.to(args.device)
        model_mlkg, optimizer, triple_data = accelerator.prepare(model_mlkg, optimizer, triple_data)
    # save model
    save_model(model_mlkg, accelerator, os.path.join(args.tmp_dir, "final_v2.pt"))
    del model_mlkg
    return


def train_both_noise(args, model_mlkg):
    # load data
    entity_dataset = EntityLoader(args)
    entity_data = Data.DataLoader(dataset=entity_dataset, batch_size=1, num_workers=1)
    triple_dataset = TripleLoader(args, entity_dataset.entity_dict)
    triple_data = Data.DataLoader(dataset=triple_dataset, batch_size=1, num_workers=1)
    # set parameters: autograd
    grad_parameters(model_mlkg, False)
    grad_adapters(model_mlkg, True)
    grad_triple_encoder(model_mlkg, False)
    # set model and optimizer
    accelerator = Accelerator(kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
    optimizer = AdamW(model_mlkg.parameters(), lr=args.lr, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.epoch*len(triple_data)+len(entity_data))
    # model_mlkg = model_mlkg.to(args.device)
    model_mlkg, optimizer, entity_data, triple_data = accelerator.prepare(model_mlkg, optimizer, entity_data, triple_data)
    # set loss function
    lossfcn_universal = InfoNCE(negative_mode='unpaired')
    lossfcn_triple = InfoNCE()
    # enable connection with triple encoder
    model_mlkg.obj = 2
    # training entity
    # training
    count_save = 0
    time_start = time.time()
    loss_list = []
    for encoded_inputs in entity_data:
        encoded_inputs = {k:torch.squeeze(v) for k, v in encoded_inputs.items()}
        model_mlkg.zero_grad()
        # positive set input
        outputs, _ = model_mlkg(**encoded_inputs)
        # backpropogation
        loss = loss_universal(args, outputs, lossfcn_universal)
        loss_list.append(float(loss.data))
        # loss.backward()
        accelerator.backward(loss)
        optimizer.step()
        scheduler.step()
        # save model
        count_save += 1
        if count_save % 1e5 == 0:
            save_model(model_mlkg, accelerator, os.path.join(args.tmp_dir, "both_"+str(int(count_save/1e5))+".pt"))
        if count_save % 1e2 == 0:
            # time
            time_length = round(time.time() - time_start, 4)
            time_start = time.time()
            # loss
            loss_avg = round(sum(loss_list) / len(loss_list), 4)
            loss_list = []
            # print
            print("progress (both): ", count_save, " / ", len(entity_data), " | time: ", time_length, "s | loss: ",loss_avg)
    # set parameters: autograd
    grad_parameters(model_mlkg, False)
    grad_adapters(model_mlkg, True)
    grad_triple_encoder(model_mlkg, True)
    # training triple
    time_start = time.time()
    loss_list = []
    for e in range(args.epoch):
        # load data
        entity_dataset = EntityLoader(args)
        entity_data = Data.DataLoader(dataset=entity_dataset, batch_size=1, num_workers=1)
        triple_dataset = TripleLoader(args, entity_dataset.entity_dict)
        triple_data = Data.DataLoader(dataset=triple_dataset, batch_size=1, num_workers=1)
        # model_mlkg = model_mlkg.to(args.device)
        model_mlkg, optimizer, entity_data, triple_data = accelerator.prepare(model_mlkg, optimizer, entity_data, triple_data)
        for encoded_inputs in triple_data:
            encoded_inputs = {k:torch.squeeze(v) for k, v in encoded_inputs.items()}
            optimizer.zero_grad()
            # positive set input
            _, outputs = model_mlkg(**encoded_inputs)
            # backpropogation
            loss = loss_triple(outputs, lossfcn_triple)
            loss_list.append(float(loss.data))
            # loss.backward()
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            # save model
            count_save += 1
            if count_save % 1e5 == 0:
                save_model(model_mlkg, accelerator, os.path.join(args.tmp_dir, "both_"+str(int(count_save/1e5))+".pt"))
            if count_save % 1e2 == 0:
                # time
                time_length = round(time.time() - time_start, 4)
                time_start = time.time()
                # loss
                loss_avg = round(sum(loss_list) / len(loss_list), 4)
                loss_list = []
                # print
                print("progress (both): ", count_save, " / ", len(triple_data)*args.epoch, " | time: ", time_length, "s | loss: ",loss_avg)
    # save model
    save_model(model_mlkg, accelerator, os.path.join(args.tmp_dir, "final_v3.pt"))
    del model_mlkg
    return



def ki_mlkg(args):
    # define model
    model_mlkg = MLKGLM(args)
    # train adapter
    if not os.path.exists(os.path.join(args.tmp_dir, "final_v1.pt")):
        train_adapter(args, model_mlkg)
    model_mlkg = MLKGLM(args)
    load_model(model_mlkg, os.path.join(args.tmp_dir, "final_v1.pt"))
    # train triple_encoder
    if not os.path.exists(os.path.join(args.tmp_dir, "final_v2.pt")):
        train_triple_encoder(args, model_mlkg)
    model_mlkg = MLKGLM(args)
    load_model(model_mlkg, os.path.join(args.tmp_dir, "final_v2.pt"))
    # train both with noise
    if not os.path.exists(os.path.join(args.tmp_dir, "final_v3.pt")):
        train_both_noise(args, model_mlkg)
    model_mlkg = MLKGLM(args)
    load_model(model_mlkg, os.path.join(args.tmp_dir, "final_v3.pt"))