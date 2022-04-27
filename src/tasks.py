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
from models import MLKGLM, loss_universal, loss_triple, loss_wocontext, fusion_adapter

seed = 123
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

'''
def train_entity_universal(args, model_mlkg):
    #### (h, h')
    # load data
    entity_dataset = EntityLoader(args)
    entity_data = Data.DataLoader(dataset=entity_dataset, batch_size=1, num_workers=1)
    # set masking tokenizer
    args.lm_mask_token_id = entity_dataset.lm_mask_token_id
    model_mlkg.lm_mask_token_id = args.lm_mask_token_id
    # set parameters: autograd
    grad_parameters(model_mlkg, False)
    grad_kgencoder(model_mlkg, True)
    # set model and optimizer
    accelerator = Accelerator(kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
    optimizer = AdamW(model_mlkg.parameters(), lr=args.lr, eps=args.adam_epsilon)
    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_cycles=10,
                                                num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.entity_epoch*len(entity_data))
    # model_mlkg = model_mlkg.to(args.device)
    model_mlkg, optimizer, entity_data = accelerator.prepare(model_mlkg, optimizer, entity_data)
    # set loss function
    lossfcn_universal = InfoNCE(negative_mode='unpaired')
    # training
    count_save = 0
    time_start = time.time()
    loss_list = []
    for e in range(args.entity_epoch):
        for encoded_inputs in entity_data:
            encoded_inputs = {k:torch.squeeze(v) for k, v in encoded_inputs.items()}
            model_mlkg.zero_grad()
            # positive set input
            outputs = model_mlkg(**encoded_inputs)
            # backpropogation
            loss = loss_universal(args, outputs, lossfcn_universal)
            loss_list.append(float(loss.data))
            # loss.backward()
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            # save model
            count_save += 1
            if count_save % 1e3 == 0:
                # time
                time_length = round(time.time() - time_start, 4)
                time_start = time.time()
                # loss
                loss_avg = round(sum(loss_list) / len(loss_list), 4)
                loss_list = []
                # print
                print("progress (entity): ", count_save, " / ", len(entity_data)*args.entity_epoch, " | time: ", time_length, "s | loss: ",loss_avg)
        # load data
        entity_dataset = EntityLoader(args)
        entity_data = Data.DataLoader(dataset=entity_dataset, batch_size=1, num_workers=1)
        # model_mlkg = model_mlkg.to(args.device)
        entity_data = accelerator.prepare(entity_data)
    #### (h, t)
    # load data
    entity_dataset = EntityLoader(args)
    triple_dataset = TripleLoader(args, entity_dataset.entity_dict)
    triple_data = Data.DataLoader(dataset=triple_dataset, batch_size=1, num_workers=1)
    # set model and optimizer
    accelerator = Accelerator(kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
    optimizer = AdamW(model_mlkg.parameters(), lr=args.lr, eps=args.adam_epsilon)
    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_cycles=10,
                                                num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.triple_epoch*len(triple_data))
    # model_mlkg = model_mlkg.to(args.device)
    model_mlkg, optimizer, triple_data = accelerator.prepare(model_mlkg, optimizer, triple_data)
    # set loss function
    lossfcn_triple = InfoNCE(negative_mode='unpaired')
    # training
    count_save = 0
    time_start = time.time()
    loss_list = []
    for e in range(args.triple_epoch):
        for encoded_inputs in triple_data:
            encoded_inputs = {k:torch.squeeze(v) for k, v in encoded_inputs.items()}
            optimizer.zero_grad()
            # positive set input
            outputs = model_mlkg(**encoded_inputs)
            # backpropogation
            loss = loss_triple(args, outputs, lossfcn_triple)
            loss_list.append(float(loss.data))
            # loss.backward()
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            # save model
            count_save += 1
            if count_save % 1e5 == 0:
                save_model(model_mlkg, accelerator, os.path.join(args.tmp_dir, "triple_encoder_"+str(int(count_save/1e5))+".pt"))
            if count_save % 1e3 == 0:
                # time
                time_length = round(time.time() - time_start, 4)
                time_start = time.time()
                # loss
                loss_avg = round(sum(loss_list) / len(loss_list), 4)
                loss_list = []
                # print
                print("progress (triple): ", count_save, " / ", len(triple_data)*args.triple_epoch, " | time: ", time_length, "s | loss: ",loss_avg)
        # load data
        triple_dataset = TripleLoader(args, entity_dataset.entity_dict)
        triple_data = Data.DataLoader(dataset=triple_dataset, batch_size=1, num_workers=1)
        # model_mlkg = model_mlkg.to(args.device)
        triple_data = accelerator.prepare(triple_data)
    # save model
    save_model(model_mlkg, accelerator, os.path.join(args.tmp_dir, "final_v1.pt"))
    del model_mlkg
    return


def train_triple_encoder(args, model_mlkg):
    # load data
    entity_dataset = EntityLoader(args)
    mix_dataset = MixLoader(args, entity_dataset.entity_dict, triple_context=True)
    mix_data = Data.DataLoader(dataset=mix_dataset, batch_size=1, num_workers=1)
    # set masking tokenizer
    args.lm_mask_token_id = mix_dataset.lm_mask_token_id
    model_mlkg.lm_mask_token_id = args.lm_mask_token_id
    # set parameters: autograd
    grad_parameters(model_mlkg, False)
    grad_kgencoder(model_mlkg, True)
    # set model and optimizer
    accelerator = Accelerator(kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
    optimizer = AdamW(model_mlkg.parameters(), lr=args.lr, eps=args.adam_epsilon)
    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_cycles=10,
                                                num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.triple_epoch*2*len(mix_data))
    # model_mlkg = model_mlkg.to(args.device)
    model_mlkg, optimizer, mix_data = accelerator.prepare(model_mlkg, optimizer, mix_data)
    # set loss function
    lossfcn_universal = InfoNCE(negative_mode='unpaired')
    lossfcn_triple = InfoNCE(negative_mode='unpaired')
    # training
    count_save = 0
    time_start = time.time()
    loss_list1, loss_list2 = [], []
    for e in range(args.triple_epoch):
        for encoded_inputs in mix_data:
            a=time.time()
            encoded_inputs_e, encoded_inputs_t = encoded_inputs
            encoded_inputs_e = {k:torch.squeeze(v) for k, v in encoded_inputs_e.items()}
            encoded_inputs_t = {k:torch.squeeze(v) for k, v in encoded_inputs_t.items()}
            #### ((h,t), t)
            optimizer.zero_grad()
            # positive set input
            outputs = model_mlkg(**encoded_inputs_t)
            # backpropogation: triple
            loss = loss_triple(args, outputs, lossfcn_triple, encoded_inputs_t["input_ids"])
            loss_list2.append(float(loss.data))
            accelerator.backward(loss)
            # zero grad
            optimizer.step()
            scheduler.step()
            #### ((h,t), h')
            optimizer.zero_grad()
            outputs = model_mlkg(**encoded_inputs_e)
            # backpropogation: entity
            loss = loss_universal(args, outputs, lossfcn_universal, encoded_inputs_e["input_ids"])
            loss_list1.append(float(loss.data))
            accelerator.backward(loss)
            # zero grad
            optimizer.step()
            scheduler.step()
            # save model
            count_save += 1
            if count_save % 1e3 == 0:
                # time
                time_length = round(time.time() - time_start, 4)
                time_start = time.time()
                # loss
                loss_avg1 = round(sum(loss_list1) / len(loss_list1), 4)
                loss_avg2 = round(sum(loss_list2) / len(loss_list2), 4)
                loss_list1, loss_list2 = [], []
                # print
                print("progress (context triple): ", count_save, "/", len(mix_data)*args.triple_epoch, " |time: ", time_length, "s |loss (u, t): ",loss_avg1, " ", loss_avg2)
        # load data
        mix_dataset = MixLoader(args, entity_dataset.entity_dict, triple_context=True)
        mix_data = Data.DataLoader(dataset=mix_dataset, batch_size=1, num_workers=1)
        # model_mlkg = model_mlkg.to(args.device)
        mix_data = accelerator.prepare(mix_data)
    # save model
    save_model(model_mlkg, accelerator, os.path.join(args.tmp_dir, "final_v2.pt"))
    del model_mlkg
    return


def train_sentence_all(args, model_mlkg):
    args.batch_num = int(args.batch_num/4)
    args.triple_epoch = max(int(args.triple_epoch/10), 1)
    # load data
    entity_dataset = EntityLoader(args)
    mix_dataset = MixLoader(args, entity_dataset.entity_dict, triple_context=False)
    mix_data = Data.DataLoader(dataset=mix_dataset, batch_size=1, num_workers=1)
    # set masking tokenizer
    args.lm_mask_token_id = mix_dataset.lm_mask_token_id
    # set parameters: autograd
    grad_parameters(model_mlkg, False)
    grad_kgencoder(model_mlkg, True)
    # grad_universal(model_mlkg, True)
    # grad_triple_encoder(model_mlkg, True)
    # set model and optimizer
    accelerator = Accelerator(kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
    optimizer = AdamW(model_mlkg.parameters(), lr=args.lr, eps=args.adam_epsilon)
    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_cycles=10,
                                                num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.triple_epoch*2*len(mix_data))
    # model_mlkg = model_mlkg.to(args.device)
    model_mlkg, optimizer, mix_data = accelerator.prepare(model_mlkg, optimizer, mix_data)
    # set loss function
    lossfcn_universal = InfoNCE(negative_mode='unpaired')
    lossfcn_triple = InfoNCE(negative_mode='unpaired')
    # training
    count_save = 0
    time_start = time.time()
    loss_list1, loss_list2 = [], []
    for e in range(min(1, args.triple_epoch)):
        for encoded_inputs in mix_data:
            encoded_inputs_e, encoded_inputs_s = encoded_inputs
            encoded_inputs_e = {k:torch.squeeze(v) for k, v in encoded_inputs_e.items()}
            encoded_inputs_s = {k:torch.squeeze(v) for k, v in encoded_inputs_s.items()}
            #### ((h,t), t)
            optimizer.zero_grad()
            # positive set input
            outputs = model_mlkg(**encoded_inputs_s)
            # backpropogation: triple
            loss = loss_triple(args, outputs, lossfcn_triple, encoded_inputs_s["input_ids"])
            loss_list2.append(float(loss.data))
            accelerator.backward(loss)
            # zero grad
            optimizer.step()
            scheduler.step()
            #### ((h,t), h')
            outputs = model_mlkg(**encoded_inputs_e)
            # backpropogation: entity
            loss = loss_universal(args, outputs, lossfcn_universal, encoded_inputs_e["input_ids"])
            loss_list1.append(float(loss.data))
            accelerator.backward(loss)
            # zero grad
            optimizer.step()
            scheduler.step()
            # save model
            count_save += 1
            if count_save % 1e3 == 0:
                # time
                time_length = round(time.time() - time_start, 4)
                time_start = time.time()
                # loss
                loss_avg1 = round(sum(loss_list1) / len(loss_list1), 4)
                loss_avg2 = round(sum(loss_list2) / len(loss_list2), 4)
                loss_list1, loss_list2 = [], []
                print("progress (context sent.): ", count_save, "/", len(mix_data)*args.triple_epoch, " |time: ", time_length, "s |loss (u, t): ",loss_avg1, " ", loss_avg2)
        # load data
        mix_dataset = MixLoader(args, entity_dataset.entity_dict, triple_context=False)
        mix_data = Data.DataLoader(dataset=mix_dataset, batch_size=1, num_workers=1)
        # model_mlkg = model_mlkg.to(args.device)
        mix_data = accelerator.prepare(mix_data)
    # save model
    save_model(model_mlkg, accelerator, os.path.join(args.tmp_dir, "final_v3.pt"))
    del model_mlkg
    return



def train_wocontext(args, model_mlkg):
    # load data
    wocontext_dataset = WOCLoader(args)
    wocontext_data = Data.DataLoader(dataset=wocontext_dataset, batch_size=1, num_workers=1)
    args.lm_mask_token_id = wocontext_dataset.lm_mask_token_id
    # set parameters: autograd
    grad_parameters(model_mlkg, False)
    grad_kgencoder(model_mlkg, True)
    # set model and optimizer
    accelerator = Accelerator(kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
    optimizer = AdamW(model_mlkg.parameters(), lr=args.lr, eps=args.adam_epsilon, weight_decay=1e-4)
    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_cycles=5,
                                                num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.triple_epoch*len(wocontext_data))
    # model_mlkg = model_mlkg.to(args.device)
    model_mlkg, optimizer, wocontext_data = accelerator.prepare(model_mlkg, optimizer, wocontext_data)
    # training
    count_save = 0
    time_start = time.time()
    loss_list1, loss_list2 = [], []
    for e in range(args.triple_epoch):
        for encoded_inputs in wocontext_data:
            encoded_inputs_e, encoded_inputs_t = encoded_inputs
            encoded_inputs_e = {k:torch.squeeze(v) for k, v in encoded_inputs_e.items()}
            encoded_inputs_t = {k:torch.squeeze(v) for k, v in encoded_inputs_t.items()}
            #### triple
            optimizer.zero_grad()
            outputs, _ = model_mlkg(**encoded_inputs_t)
            # backpropogation: triple
            loss = loss_wocontext(args, outputs)
            loss_list2.append(float(loss.data))
            accelerator.backward(loss)
            # zero grad
            optimizer.step()
            scheduler.step()
            #### entity
            outputs, _ = model_mlkg(**encoded_inputs_e)
            # backpropogation: entity
            loss = loss_wocontext(args, outputs)
            loss_list1.append(float(loss.data))
            accelerator.backward(loss)
            # zero grad
            optimizer.step()
            scheduler.step()
            # save model
            count_save += 1
            if count_save % 1e3 == 0:
                # time
                time_length = round(time.time() - time_start, 4)
                time_start = time.time()
                # loss
                loss_avg1 = round(sum(loss_list1) / len(loss_list1), 4)
                loss_avg2 = round(sum(loss_list2) / len(loss_list2), 4)
                loss_list1, loss_list2 = [], []
                print("progress (w/o context): ", count_save, "/", len(wocontext_data)*args.triple_epoch, " |time: ", time_length, "s |loss (u, t): ",loss_avg1, " ", loss_avg2)
        # load data
        wocontext_dataset = WOCLoader(args)
        wocontext_data = Data.DataLoader(dataset=wocontext_dataset, batch_size=1, num_workers=1)
        wocontext_data = accelerator.prepare(wocontext_data)
    # save model
    save_model(model_mlkg, accelerator, os.path.join(args.tmp_dir, "pytorch_model_wocontext.bin"))
    del model_mlkg
    return


def train_wcontext(args, model_mlkg):
    args.lr = args.lr / 10
    args.batch_num = int(args.batch_num / 2)
    # load data
    wcontext_dataset = WCLoader(args)
    wcontext_data = Data.DataLoader(dataset=wcontext_dataset, batch_size=1, num_workers=1)
    args.lm_mask_token_id = wcontext_dataset.lm_mask_token_id
    # set parameters: autograd
    grad_parameters(model_mlkg, False)
    grad_kgencoder(model_mlkg, True)
    # set model and optimizer
    accelerator = Accelerator(kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
    optimizer = AdamW(model_mlkg.parameters(), lr=args.lr, eps=args.adam_epsilon, weight_decay=1e-4)
    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_cycles=10,
                                                num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.triple_epoch*len(wcontext_data))
    # model_mlkg = model_mlkg.to(args.device)
    model_mlkg, optimizer, wcontext_data = accelerator.prepare(model_mlkg, optimizer, wcontext_data)
    # training
    count_save = 0
    time_start = time.time()
    loss_list1, loss_list2 = [], []
    for e in range(1):
        for encoded_inputs in wcontext_data:
            encoded_inputs_e, encoded_inputs_s = encoded_inputs
            encoded_inputs_e = {k:torch.squeeze(v) for k, v in encoded_inputs_e.items()}
            encoded_inputs_s = {k:torch.squeeze(v) for k, v in encoded_inputs_s.items()}
            #### triple
            optimizer.zero_grad()
            outputs, lm_emb = model_mlkg(**encoded_inputs_s)
            # backpropogation: triple
            loss = loss_wocontext(args, outputs, encoded_inputs_s["input_ids"], lm_emb)
            loss_list2.append(float(loss.data))
            accelerator.backward(loss)
            # zero grad
            optimizer.step()
            scheduler.step()
            #### entity
            outputs, lm_emb = model_mlkg(**encoded_inputs_e)
            # backpropogation: entity
            loss = loss_wocontext(args, outputs, encoded_inputs_e["input_ids"], lm_emb)
            loss_list1.append(float(loss.data))
            accelerator.backward(loss)
            # zero grad
            optimizer.step()
            scheduler.step()
            # save model
            count_save += 1
            if count_save % 1e3 == 0:
                # time
                time_length = round(time.time() - time_start, 4)
                time_start = time.time()
                # loss
                loss_avg1 = round(sum(loss_list1) / len(loss_list1), 4)
                loss_avg2 = round(sum(loss_list2) / len(loss_list2), 4)
                loss_list1, loss_list2 = [], []
                print("progress (w context): ", count_save, "/", len(wcontext_data)*args.triple_epoch, " |time: ", time_length, "s |loss (u, t): ",loss_avg1, " ", loss_avg2)
        # load data
        wcontext_dataset = WOCLoader(args)
        wcontext_data = Data.DataLoader(dataset=wcontext_dataset, batch_size=1, num_workers=1)
        wcontext_data = accelerator.prepare(wcontext_data)
    # save model
    save_model(model_mlkg, accelerator, os.path.join(args.tmp_dir, "pytorch_model.bin"))
    del model_mlkg
    return
'''

def train_adapter_phrase(args, model_mlkg):
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
    for e in range(args.triple_epoch):
        for encoded_inputs in wocontext_data:
            input_e1, input_e2, input_t1, input_t2 = encoded_inputs
            input_e1 = {k:torch.squeeze(v) for k, v in input_e1.items()}
            input_e2 = {k:torch.squeeze(v) for k, v in input_e2.items()}
            input_t1 = {k:torch.squeeze(v) for k, v in input_t1.items()}
            input_t2 = {k:torch.squeeze(v) for k, v in input_t2.items()}
            optimizer.zero_grad()
            #### triple
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

def train_adapter_sentence(args, model_mlkg):
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
    loss_list1, loss_list2 = [], []
    for e in range(args.triple_epoch):
        for encoded_inputs in wocontext_data:
            if random.random() > 0.4: continue
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
                loss_list1, loss_list2 = [], []
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
    loss_list1, loss_list2 = [], []
    for e in range(args.triple_epoch):
        for encoded_inputs in wcontext_data:
            if random.random() > 0.4: continue
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
                loss_list1, loss_list2 = [], []
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
    # define model
    '''
    model_mlkg = MLKGLM(args)
    # train uncontextualized
    if not os.path.exists(os.path.join(args.tmp_dir, "final_v1.pt")):
        train_entity_universal(args, model_mlkg)
    model_mlkg = MLKGLM(args)
    load_model(model_mlkg, os.path.join(args.tmp_dir, "final_v1.pt"))
    # train triple context
    if not os.path.exists(os.path.join(args.tmp_dir, "final_v2.pt")):
        train_triple_encoder(args, model_mlkg)
    model_mlkg = MLKGLM(args)
    load_model(model_mlkg, os.path.join(args.tmp_dir, "final_v2.pt"))
    # train sentence context
    if not os.path.exists(os.path.join(args.tmp_dir, "final_v3.pt")):
        train_sentence_all(args, model_mlkg)
    model_mlkg = MLKGLM(args)
    load_model(model_mlkg, os.path.join(args.tmp_dir, "final_v3.pt"))
    '''
    '''
    # train uncontextualized
    if not os.path.exists(os.path.join(args.tmp_dir, "pytorch_model_wocontext.bin")):
        train_wocontext(args, model_mlkg)
    model_mlkg = MLKGLM(args)
    load_model(model_mlkg, os.path.join(args.tmp_dir, "pytorch_model_wocontext.bin"))
    # train sentence context
    if not os.path.exists(os.path.join(args.tmp_dir, "pytorch_model.bin")):
        train_wcontext(args, model_mlkg)
    model_mlkg = MLKGLM(args)
    load_model(model_mlkg, os.path.join(args.tmp_dir, "pytorch_model.bin"))
    '''
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
    for i in range(10):
        print("====> Fusion: phrase <====")
        model_mlkg = fusion_adapter(args)
        load_model(model_mlkg, args.tmp_dir)
        train_fuse_phrase(args, model_mlkg)
        print("====> Fusion: sentence <====")
        model_mlkg = fusion_adapter(args)
        load_model(model_mlkg, args.tmp_dir, True)
        train_fuse_sentence(args, model_mlkg)
