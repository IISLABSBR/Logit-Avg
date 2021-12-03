from torch.optim import lr_scheduler
import torch
import numpy as np
import logging
import os
from datetime import datetime
import sys


def setup_default_logging(dataset, seed, exp_dir, default_level=logging.INFO,
                          format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"):

    output_dir = os.path.join(exp_dir, f'{dataset}_{datetime.now().strftime("%Y%m%d_%H%M")}')

    os.makedirs(output_dir, exist_ok=True)

    logger = logging.getLogger('train')

    logging.basicConfig(  # unlike the root logger, a custom logger can’t be configured using basicConfig()
        filename=os.path.join(output_dir, f'{datetime.now()}.log'),
        format=format,
        datefmt="%m/%d/%Y %H:%M:%S",
        level=default_level)

    # print
    # file_handler = logging.FileHandler()
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(default_level)
    console_handler.setFormatter(logging.Formatter(format))
    logger.addHandler(console_handler)

    return logger, output_dir

def fix_weight_decay(model):
    # ignore weight decay for parameters in bias, batch norm and activation
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(map(lambda x: x in name, ['bias', 'batch_norm', 'activation'])):
            no_decay.append(param)
        else:
            decay.append(param)
    params = [{'params': decay}, {'params': no_decay, 'weight_decay': 0}]
    return params


def get_k_scores(topk, labels, num_items, Ks, results):

    for K in Ks:
        hit_ranks = torch.where(topk[:, :K] == labels)[1] + 1
        hit_ranks = hit_ranks.float().cpu()
        results[f'HR@{K}'] += hit_ranks.numel()
        results[f'MRR@{K}'] += hit_ranks.reciprocal().sum().item()
        results[f'NDCG@{K}'] += torch.log2(1 + hit_ranks).reciprocal().sum().item()
        results[f'Cov@{K}'] += topk[:, :K].reshape(-1).unique().size()[0]/num_items

    return results


def prepare_batch(batch, device):
    inputs, last_items, last_items_sidx, labels = batch
    inputs_gpu = [x.to(device) for x in inputs]
    last_items_gpu = last_items.to(device)
    labels_gpu = labels.to(device)

    return inputs_gpu, last_items_gpu, last_items_sidx, labels_gpu


def print_results(results, epochs=None):
    print("Metric\t" + '\t'.join(results.keys()))
    print('Value\t' + '\t'.join([f'{round(val * 100, 2):.2f}' for val in results.values()]))
    if epochs is not None:
        print('Epoch\t' + '\t'.join([str(epochs[metric]) for metric in results]))


class AverageMeter(object):
    """
    Computes and stores the average and current value
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum = self.sum + val
        self.count += n
        # self.avg = self.sum / (self.count + 1e-20)
        self.avg = self.sum / self.count


class WarmupCosineLrScheduler(lr_scheduler._LRScheduler):

    def __init__(
            self,
            optimizer,
            max_iter,
            warmup_iter,
            warmup_ratio=5e-4,
            warmup='exp',
            last_epoch=-1,
    ):
        self.max_iter = max_iter
        self.warmup_iter = warmup_iter
        self.warmup_ratio = warmup_ratio
        self.warmup = warmup
        super(WarmupCosineLrScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        ratio = self.get_lr_ratio()
        lrs = [ratio * lr for lr in self.base_lrs]
        return lrs

    def get_lr_ratio(self):
        if self.last_epoch < self.warmup_iter:
            ratio = self.get_warmup_ratio()
        else:
            real_iter = self.last_epoch - self.warmup_iter
            real_max_iter = self.max_iter - self.warmup_iter
            ratio = np.cos((7 * np.pi * real_iter) / (16 * real_max_iter))
            #ratio = 0.5 * (1. + np.cos(np.pi * real_iter / real_max_iter))
        return ratio

    def get_warmup_ratio(self):
        assert self.warmup in ('linear', 'exp')
        alpha = self.last_epoch / self.warmup_iter
        if self.warmup == 'linear':
            ratio = self.warmup_ratio + (1 - self.warmup_ratio) * alpha
        elif self.warmup == 'exp':
            ratio = self.warmup_ratio ** (1. - alpha)
        return ratio