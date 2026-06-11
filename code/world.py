import random
import numpy as np
import torch
from parse import parse_args
args = parse_args()


config = {}
config['bpr_batch_size'] = args.bpr_batch
config['K'] = args.K

config['test_u_batch_size'] = args.testbatch

config['epochs'] = args.epochs

config['dataset'] = args.dataset

GPU = torch.cuda.is_available()

device = torch.device('cuda' if GPU else "cpu")

seed = args.seed

def set_seed(seed_value: int) -> None:
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(seed)

dataset = args.dataset

TRAIN_epochs = args.epochs

patience = args.patience

num_neg = args.num_neg

dropout_rate = args.dropout

decay = args.decay

tau = args.tau

init_weight = args.init_weight

lambda_ = args.lambda_
lr = args.lr
flag = 0
def cprint(words: str):
    print(f"\033[0;30;43m{words}\033[0m")

def bprint(words:str):
    print(f"\033[0;30;45m{words}\033[0m")