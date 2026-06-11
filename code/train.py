import os
import sys
import time

import torch
from torch.utils.data import DataLoader

code_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if code_root not in sys.path:
    sys.path.insert(0, code_root)

import utils
import world
from dataloader import Loader
from model import HMGCN
from procedure import test, get_topk_list
from utils import init_logger, print_log, write_final_log

if world.config['dataset'] == 'yelp2018':
    config = {
        'init':'normal',#NORMAL DISTRIBUTION
        'init_weight':world.init_weight,#INIT WEIGHT
        'dim':64,#EMBEDDING_SIZE
        'decay':world.decay,#L2_NORM
        'K':3,
        'beta':0.8,#BETA
        'lambda': world.lambda_,
        'lr':world.lr,#LEARNING_RATE
        'lam_hat': 0.9,
        'prop_step': 10,
        'gamma': 6.0,
        'quasi_newton': True,
        'eta': None,
    }

def Fast_Sampling(dataset:Loader):
    """
    With Uniformal Sampling on Graph
    """
    train_edge_index = dataset.train_edge_index.to(device)
    num_items = dataset.num_items
    batch_size = 2048
    mini_batch = []
    indexes = []
    train_loader = DataLoader(
            range(train_edge_index.size(1)),
            shuffle=True,
            batch_size=batch_size)
    for index in train_loader:
        pos_edge_label_index = train_edge_index[:,index]
        neg_edge_label_index = torch.randint(0, num_items,(index.numel(), ), device=device)
        edge_label_index = torch.stack([
            pos_edge_label_index[0],
            pos_edge_label_index[1],
            neg_edge_label_index,
        ])
        mini_batch.append(edge_label_index)
        indexes.append(index)
    return mini_batch,indexes

def train(dataset:Loader,
          model:HMGCN,
          opt:torch.optim.Optimizer):
    model = model
    model.train()
    edge_index,indexes = Fast_Sampling(dataset=dataset)
    aver_loss = 0.
    total_batch = len(edge_index)
    for edge_label_index,index in zip(edge_index,indexes):
        opt.zero_grad()
        loss = model.get_loss(edge_label_index)
        loss.backward()
        opt.step()   
        aver_loss += (loss)
    aver_loss /= total_batch
    return aver_loss

device = world.device
project_root = os.path.abspath(os.path.join(code_root, ".."))
dataset = Loader(path=os.path.join(project_root, "data") + os.sep)
log_dir = os.path.join(code_root, "noise", "log")
os.makedirs(log_dir, exist_ok=True)
log_path = init_logger(model_name='HMGCN', dataset_name=world.config['dataset'], log_dir=log_dir)


train_edge_index = dataset.train_edge_index.to(device)
test_edge_index = dataset.test_edge_index.to(device)
num_users = dataset.num_users
num_items = dataset.num_items
model = HMGCN(num_users=num_users,
                  num_items=num_items,
                  edge_index=train_edge_index,
                  config=config).to(device)
opt = torch.optim.Adam(params=model.parameters(),lr=config['lr'])
best = 0.
patience = 0.
max_score = 0.
best_recall = 0.
best_epoch = 0
best_ndcg = 0.
best_edge_diff = None
# print(model.generate_weight(train_edge_index))
for epoch in range(1, 2001):
    start_time = time.time()
    loss = train(dataset=dataset,model=model,opt=opt)
    end_time = time.time()
    recall,ndcg = test([10,20,50],model,train_edge_index,test_edge_index,num_users)
    flag,best,patience = utils.early_stopping(recall[20],ndcg[20],best,patience,model)
    if patience == 0:
        best_epoch = epoch
        best_recall = recall[20]
        best_ndcg = ndcg[20]
    if flag == 1:
        break
    print_log(f'Epoch: {epoch:03d}, aver_loss : {loss:.5f}, R@20: '
            f'{recall[20]:.4f}, N@20: {ndcg[20]:.4f}, '
            f'R@10: {recall[10]:.4f}, N@10: {ndcg[10]:.4f}, '
            f'R@50: {recall[50]:.4f}, N@50: {ndcg[50]:.4f}, '
            f'time:{end_time-start_time:.2f} seconds')
write_final_log(best_epoch=best_epoch, recall=best_recall, ndcg=best_ndcg, config=config)
print_log(f"Log saved to: {log_path}")
