# Logit Averaging Model
#


import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import pickle

from utils.dataset import read_dataset, Dataset
from utils.collate import seq_to_eop_multigraph, collate_fn_factory
from utils.train import train

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='yoochoose', type=str, help='dataset directory')
parser.add_argument('--embedding-dim', type=int, default=32, help='the embedding size')
parser.add_argument('--num-layers', type=int, default=3, help='the number of layers')
parser.add_argument('--feat-drop', type=float, default=0.2, help='the dropout ratio for features')
parser.add_argument('--lr', type=float, default=1e-3, help='the learning rate')
parser.add_argument('--batch-size', type=int, default=512, help='the batch size for training')
parser.add_argument('--epochs', type=int, default=30, help='the number of training epochs')
parser.add_argument('--weight-decay', type=float, default=1e-4, help='the parameter for L2 regularization', )
parser.add_argument('--Ks', default='10,20', help='the values of K in evaluation metrics, separated by commas', )
parser.add_argument('--patience', type=int, default=10,
                        help='the number of epochs that the performance does not improves after which the training stops', )
parser.add_argument('--num-workers', type=int, default=4, help='the number of processes to load the input graphs', )
parser.add_argument('--valid-split', type=float, default=None, help='the fraction for the validation set', )
parser.add_argument('--log-interval', type=int, default=100,
                        help='print the loss after this number of iterations', )
parser.add_argument('--log-savedir', default='my_exp_logitavg', help='the directory name for saving log information')
parser.add_argument('--seed', type=int, default=220,
                        help='seed for random behaviors, no seed if negtive')
parser.add_argument('--load-model', type=str, default=None)
parser.add_argument('--lam-p', type=float, default=1)
args = parser.parse_args()
print(args)

dataset_dir = Path(f'datasets/{args.dataset}')
dataset_name = args.dataset
args.Ks = [int(K) for K in args.Ks.split(',')]

print("reading Dataset")
train_sessions, test_sessions, num_items = read_dataset(dataset_dir)

with open(dataset_dir / f'{dataset_name}_train_top_75.pickle', 'rb') as f:
    top_labels = pickle.load(f)

train_set = Dataset(train_sessions)
test_set = Dataset(test_sessions)

print("loading Dataset")
collate_fn = collate_fn_factory(seq_to_eop_multigraph, top_labels)

train_loader = DataLoader(train_set,
                          batch_size=args.batch_size,
                          shuffle=False,
                          num_workers=args.num_workers,
                          collate_fn=collate_fn,
                          )

test_loader = DataLoader(test_set,
                         batch_size=args.batch_size,
                         shuffle=False,
                         num_workers=args.num_workers,
                         collate_fn=collate_fn)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

lessr_part_args = dict(num_items=num_items,
                       embedding_dim=args.embedding_dim,
                       num_layers=args.num_layers,
                       device=device,
                       feat_drop=args.feat_drop,
                       )

print("Start training")

train(lessr_part_args, args.epochs, args.Ks, dataset_name,
      device, args.log_savedir, args.load_model, args.log_interval,
      train_loader, test_loader, len(train_set), num_items,
      args.batch_size, args.lam_p,
      args.lr, args.weight_decay,
      seed=args.seed
      )