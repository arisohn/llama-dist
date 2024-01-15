"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import time
from collections import defaultdict

import torch
from torch.utils.data.dataloader import DataLoader
from mingpt.utils import CfgNode as CN

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler

import sys

from tqdm import tqdm
import math
import os
import numpy as np

class Trainer:

    @staticmethod
    def get_default_config():
        C = CN()
        # device to train on
        C.device = 'auto'
        # dataloder parameters
        C.num_workers = 4
        # optimizer parameters
        C.batch_size = 64
        C.learning_rate = 3e-4
        C.betas = (0.9, 0.95)
        C.weight_decay = 0.1 # only applied on matmul weights
        C.grad_norm_clip = 1.0
        C.eta_hat = 1.0
        C.max_epochs = None
        return C

    def __init__(self, config, model, train_dataset, eval_dataset, local_rank, rank, world_size, work_dir):
        self.dev = torch.device('cuda', local_rank) 

        self.config = config
        self.model = model.to(self.dev)
        self.optimizer = None
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.local_rank = local_rank
        self.rank = rank
        self.world_size = world_size
        self.work_dir = work_dir

        self.ddp_model = DDP(self.model, device_ids=[local_rank]) 

        print(local_rank, rank, world_size, self.dev) # 0 cuda cuda:0

    # https://github.com/karpathy/minGPT/tree/031ad36f292df6d3bcbec677f10a09ce54cb7f50
    # https://tutorials.pytorch.kr/advanced/ddp_pipeline.html
    def run(self):
        model, config, ddp_model = self.model, self.config, self.ddp_model

        self.optimizer = model.configure_optimizers(config)

        def run_epoch(loader, is_train):
            ddp_model.train(is_train)

            iter_time = time.time()
            losses = []

            for it, (x, y, x_m, y_m) in enumerate(loader):

                # place data on the correct device
                x = x.to(self.dev)
                y = y.to(self.dev)

                x_m = x_m.to(self.dev)
                y_m = y_m.to(self.dev)

                # forward the model
                with torch.set_grad_enabled(is_train):
                    logits, loss = ddp_model(x, y, x_m, y_m)
                    losses.append(loss)

                if is_train:

                    # backprop and update the parameters
                    ddp_model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), config.grad_norm_clip)
                    self.optimizer.step()

                    # https://github.com/huggingface/transformers/blob/v4.24.0/src/transformers/optimization.py#L104
                    if self.current_step < self.num_warmup_steps:
                        lr_mult = float(self.current_step) / float(max(1, self.num_warmup_steps))
                    else:
                        progress = float(self.current_step - self.num_warmup_steps) / float(max(1, self.num_training_steps - self.num_warmup_steps))
                        lr_mult = max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(0.5) * 2.0 * progress)))
                    lr = config.learning_rate * lr_mult
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = lr

                    # report progress
                    if it%100 == 0:
                        tnow = time.time()
                        sys.stdout.write(f"local rank {self.local_rank} epoch {epoch} iter {it:06d}/{len(loader)}" \
                                            + f" step {self.current_step:06d}/{self.num_training_steps} train loss {loss.item():8.5f}" \
                                            + f" lr {lr:e} lr_mult {lr_mult:e} duration {tnow-iter_time:8.5f}\n")
                        sys.stdout.flush()
                        iter_time = tnow

                    self.current_step += 1
            
            if not is_train:
                losses = [l.cpu().item() for l in losses]
                test_loss = np.mean(losses)
                return test_loss

        best_loss = float('inf')

        train_sampler = DistributedSampler(self.train_dataset, num_replicas=self.world_size, rank=self.rank, drop_last=True)
        train_loader = DataLoader(
            self.train_dataset,
            shuffle=False,
            pin_memory=True,
            batch_size=config.batch_size,
            num_workers=config.num_workers, 
            sampler=train_sampler)

        eval_sampler = DistributedSampler(self.eval_dataset, num_replicas=self.world_size, rank=self.rank, drop_last=True)
        eval_loader = DataLoader(
            self.eval_dataset,
            shuffle=False,
            pin_memory=True,
            batch_size=config.batch_size,
            num_workers=config.num_workers, 
            sampler=eval_sampler)

        self.current_step = 0
        self.num_training_steps = len(train_loader) * config.max_epochs
        self.num_warmup_steps = 2000

        for epoch in range(config.max_epochs):
            train_sampler.set_epoch(epoch)                      # shuffle every epoch
            run_epoch(train_loader, is_train=True)

            test_loss = run_epoch(eval_loader, is_train=False)

            if self.local_rank == 0 and test_loss < best_loss:
                print("saving model...")

                best_loss = test_loss
                ckpt_path = os.path.join(self.work_dir, "model_{}_{}.pt".format(epoch, test_loss))
                torch.save(self.model.state_dict(), ckpt_path)

