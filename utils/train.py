# Logit Averaging Model - loss1 + loss2

import time
import numpy as np
import random
from collections import defaultdict
import tensorboard_logger

import torch
from torch import nn
import torch.nn.functional as F

from model import LESSR_part
from utils.utils import AverageMeter, WarmupCosineLrScheduler, \
    fix_weight_decay, get_k_scores, prepare_batch, setup_default_logging


def set_model(num_items, embedding_dim, num_layers, device, feat_drop):
    model = LESSR_part(num_items, embedding_dim, num_layers, device, feat_drop)
    model = model.to(device)
    model.train()
    return model

def train_one_epoch(epoch,
                    top_labels_logits,
                    device, logger, log_interval,
                    dataset, exp_dir,
                    train_loader,
                    model, optim, lr_schdlr,
                    lam_p
                    ):
    model.train()

    loss_o_meter = AverageMeter()
    loss_p_meter = AverageMeter()

    epoch_start = time.time()
    it = 0

    for batch in train_loader:
        inputs, last_items, top_labels_sidx, labels = prepare_batch(batch, device)

        # --------------------------------------------

        _, logits = model(*inputs)
        loss_o = nn.functional.cross_entropy(logits, labels)

        probs = logits.clone()

        with torch.no_grad():
            # logits = logits.detach()
            labels = labels.detach()

            if epoch == 0 and it == 0:
                for i, sidx in enumerate(top_labels_sidx):
                    if len(sidx) == 0:
                        top_labels_logits.append([])
                    else:
                        gathered_logits = torch.mean(logits[sidx], dim=0)
                        probs.index_copy_(0, torch.tensor(sidx).to(device),
                                          gathered_logits.view(1, -1).repeat(len(sidx), 1))
                        top_labels_logits.append(gathered_logits)

            else:
                for i, sidx in enumerate(top_labels_sidx):
                    if len(sidx) == 0:
                        pass
                    else:
                        if len(top_labels_logits[i]) == 0:
                            gathered_logits = torch.mean(logits[sidx], dim=0)
                            probs.index_copy_(0, torch.tensor(sidx).to(device),
                                              gathered_logits.view(1, -1).repeat(len(sidx), 1))
                        else:
                            gathered_logits = torch.mean(logits[sidx], dim=0)
                            gathered_logits = (gathered_logits + top_labels_logits[i]) / 2
                            probs.index_copy_(0, torch.tensor(sidx).to(device),
                                              gathered_logits.view(1, -1).repeat(len(sidx), 1))
                        top_labels_logits[i] = gathered_logits

        probs = probs.to(device)
        # probs = torch.softmax(probs, dim=1)

        loss_p = nn.functional.cross_entropy(probs, labels)
        loss = loss_o + lam_p * loss_p

        loss.backward()
        optim.step()
        lr_schdlr.step()
        optim.zero_grad()

        loss_o_meter.update(loss_o.item())
        loss_p_meter.update(loss_p.item())

        it += 1
        if (it + 1) % log_interval == 0:
            t = time.time() - epoch_start

            lr_log = [pg['lr'] for pg in optim.param_groups]
            lr_log = sum(lr_log) / len(lr_log)

            logger.info("{} {} | epoch:{}, iter: {}. loss_o: {:.3f}. loss_p: {:.3f}. LR: {:.3f}. Time: {:.2f}"
                        .format(dataset, exp_dir, epoch, it+1, loss_o_meter.avg, loss_p_meter.avg, lr_log, t))

            epoch_start = time.time()

    return top_labels_logits, loss_o_meter.avg, loss_p_meter.avg


def evaluate(model, data_loader, device, num_items, Ks=[20]):
    model.eval()
    num_samples = 0
    max_K = max(Ks)
    results = defaultdict(float)

    with torch.no_grad():
        for batch in data_loader:
            inputs, _, _, labels = prepare_batch(batch, device)
            _, logits = model(*inputs)
            batch_size = logits.size(0)
            num_samples += batch_size
            topk = torch.topk(logits, k=max_K, sorted=True)[1]
            labels = labels.unsqueeze(-1)
            results = get_k_scores(topk, labels, num_items, Ks, results)

    for metric in results:
        if 'Cov' in metric:
            results[metric] /= len(data_loader)
        else:
            results[metric] /= num_samples

    return results


def train(lessr_part_args, epochs, Ks, dataset_name,
          device, exp_dir, load_model, log_interval,
          train_loader, test_loader, train_len, num_items,
          batch_size, lam_p,
          lr=1e-3, weight_decay=1e-4,
          seed=None):

    logger, output_dir = setup_default_logging(dataset_name, seed, exp_dir)
    tb_logger = tensorboard_logger.Logger(logdir=output_dir, flush_secs=2)

    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

    n_iters_per_epoch = train_len // batch_size
    n_iters_all = n_iters_per_epoch * epochs

    logger.info("***** Running training *****")
    logger.info(f"  Task = {dataset_name}")

    if load_model is not None:
        model = set_model(**lessr_part_args)
        model.load_state_dict(torch.load(load_model))
    else:
        model = set_model(**lessr_part_args)

    logger.info("Total params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1e6))

    if weight_decay > 0:
        params = fix_weight_decay(model)
    else:
        params = model.parameters()

    optim = torch.optim.AdamW(params=params, lr=lr, weight_decay=weight_decay)
    lr_schdlr = WarmupCosineLrScheduler(optim, n_iters_all, warmup_iter=0)

    train_args = dict(model=model,
                      optim=optim,
                      lr_schdlr=lr_schdlr,
                      train_loader=train_loader,
                      device=device,
                      logger=logger,
                      log_interval=log_interval,
                      dataset=dataset_name,
                      exp_dir=exp_dir,
                      lam_p=lam_p,
                      )

    dlval = test_loader

    max_results = defaultdict(float)
    max_epochs = defaultdict(int)

    top_labels_logits = []

    logger.info('-----------start training--------------')

    for epoch in range(epochs):

        top_labels_logits, loss_o, loss_p = train_one_epoch(epoch, top_labels_logits,  **train_args)
        curr_result = evaluate(model, dlval, device, num_items, Ks)

        tb_logger.log_value('loss_o', loss_o, epoch)
        tb_logger.log_value('loss_p', loss_p, epoch)
        tb_logger.log_value('HR@10', curr_result["HR@10"], epoch)
        tb_logger.log_value('MRR@10', curr_result["MRR@10"], epoch)
        tb_logger.log_value("NDCG@10", curr_result["NDCG@10"], epoch)
        tb_logger.log_value("Cov@10", curr_result["Cov@10"], epoch)
        tb_logger.log_value("HR@20", curr_result["HR@20"], epoch)
        tb_logger.log_value("MRR@20", curr_result['MRR@20'], epoch)
        tb_logger.log_value("NDCG@20", curr_result["NDCG@20"], epoch)
        tb_logger.log_value("Cov@20", curr_result["Cov@20"], epoch)

        any_better_result = False
        for metric in curr_result:
            if curr_result[metric] > max_results[metric]:
                max_results[metric] = curr_result[metric]
                max_epochs[metric] = epoch
                any_better_result = True

            if any_better_result:
                torch.save(model.state_dict(), f'{output_dir}.pt')

        logger.info("Metric\t" + '\t'.join(curr_result.keys()))
        logger.info("Value\t" + '\t'.join([f'{round(val * 100, 2):.2f}' for val in curr_result.values()]))

        epoch += 1


    logger.info("Metric\t" + '\t'.join(max_results.keys()))
    logger.info("Value\t" + '\t'.join([f'{round(val * 100, 2):.2f}' for val in max_results.values()]))
    logger.info('Epoch\t' + '\t'.join([str(max_epochs[metric]) for metric in max_results]))

    return max_results
