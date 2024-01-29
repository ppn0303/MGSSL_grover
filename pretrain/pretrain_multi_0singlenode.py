#python pretrain_multi.py --emb_dim 300 --hidden_size 300 --epochs 100 --dropout_ratio 0.1 --dataset data/merge_0 --vocab data/merge_0/clique.txt --output_path saved_model/grover --batch_size 40 --order dfs --grover_dataset --multi
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torch.autograd import Variable

import math, random, sys
import numpy as np
from optparse import OptionParser
from gnn_model import GNN, GNN_grover

sys.path.append('./util/')

from mol_tree import *
from nnutils import *
from datautils import *
from motif_generation import *

import rdkit

# add for grover
import os, time
import wandb
from grover.topology.mol_tree import *
from grover.topology.grover_datasets import *
from sklearn.model_selection import train_test_split

#for torch ddp
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os

def ddp_setup(rank: int, world_size: int):
    """
    Args:
    rank: Unique identifier of each process
    world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "0.0.0.0"
    os.environ["MASTER_PORT"] = "7731"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)
# 치명적 오류가 발생되면 로그기록해라

def group_node_rep(node_rep, batch_index, batch_size):
    group = []
    count = 0
    for i in range(batch_size):
        num = sum(batch_index == i)
        group.append(node_rep[count:count + num])		# count += num번째 node의 표현을 그룹에 더해라
        count += num
    return group						# 최종 그룹을 출력

def train(args, model_list, loader, optimizer_list, rank):
    model, motif_model = model_list                             # 훈련간 사용 모델은 GNN모델과 motif모델이다.
    optimizer_model, optimizer_motif = optimizer_list        # 옵티마이저도 둘에 대해 각각 사용하라.

    model.train()					#모델, 모티프 모델 훈련!
    motif_model.train()
    word_acc, topo_acc = 0, 0			# 분자와 위상 정확도 변수 설정
    for step, batch in enumerate(loader):	# 데이터로더에서 순회 진행바 표시형태로 순회해서 step과 batch대로 반복하자

        batch_size = len(batch)

        graph_batch = moltree_to_graph_data(batch)		# 분자식을 파이토치 지오메트릭 패키지에서 요구되는 그래프 데이터 형태로 변경해서 배치단위로 저장   /datautils에 있음
        #store graph object data in the process stage	
        batch_index = graph_batch.batch.numpy()			# 배치내의 배치텐서를 넘파이로 인덱스에 넘겨라
        graph_batch = graph_batch.to(rank)			# 그래프배치는 GPU로 되게
        node_rep = model(graph_batch.x, graph_batch.edge_index, graph_batch.edge_attr)	# GNN모델에 그래프(x, 엣지인덱스, 엣지의 특성) 투입
        node_rep = group_node_rep(node_rep, batch_index, batch_size)			# rep는 representation의 줄임말로 노드 표현을 의미
        loss, word_loss, topo_loss, wacc, tacc = motif_model(batch, node_rep)		# motif모델에서 손실, motif정확도, 위상 정확도 출력

        optimizer_model.zero_grad()				#옵티마이저 0으로
        optimizer_motif.zero_grad()
        loss.backward()					#손실 역전파

        optimizer_model.step()				#옵티마이저 시행
        optimizer_motif.step()

        word_acc += wacc
        topo_acc += tacc					#위상 정확도
            
    return loss, word_loss, topo_loss, word_acc*100, topo_acc*100

def validation(args, model_list, loader, rank):
    model, motif_model = model_list                             # 훈련간 사용 모델은 GNN모델과 motif모델이다.

    model.eval()					#모델, 모티프 모델 훈련!
    motif_model.eval()
    word_acc, topo_acc = 0, 0			# 분자와 위상 정확도 변수 설정
    for step, batch in enumerate(loader):	# 데이터로더에서 순회 진행바 표시형태로 순회해서 step과 batch대로 반복하자

        batch_size = len(batch)

        graph_batch = moltree_to_graph_data(batch)		# 분자식을 파이토치 지오메트릭 패키지에서 요구되는 그래프 데이터 형태로 변경해서 배치단위로 저장   /datautils에 있음
        #store graph object data in the process stage	
        batch_index = graph_batch.batch.numpy()			# 배치내의 배치텐서를 넘파이로 인덱스에 넘겨라
        graph_batch = graph_batch.to(rank)			# 그래프배치는 GPU로 되게
        node_rep = model(graph_batch.x, graph_batch.edge_index, graph_batch.edge_attr)	# GNN모델에 그래프(x, 엣지인덱스, 엣지의 특성) 투입
        node_rep = group_node_rep(node_rep, batch_index, batch_size)			# rep는 representation의 줄임말로 노드 표현을 의미
        loss, word_loss, topo_loss, wacc, tacc = motif_model(batch, node_rep)		# motif모델에서 손실, motif정확도, 위상 정확도 출력

        word_acc += wacc
        topo_acc += tacc					#위상 정확도

    return loss, word_loss, topo_loss, word_acc*100, topo_acc*100


def main(rank: int, world_size: int):
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0.2,
                        help='dropout ratio (default: 0.2)')
    parser.add_argument('--graph_pooling', type=str, default="sum",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--dataset', type=str, default='./data/zinc/all.txt',
                        help='root directory of dataset. For now, only classification.')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--input_model_file', type=str, default="", help='filename to read the model (if there is any)')
    parser.add_argument('--output_path', type=str, default='./saved_model/grover',
                        help='filename to output the pre-trained model')
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers for dataset loading')   #원래는 8이었음 오류로 0으로 바꿈
    parser.add_argument("--hidden_size", type=int, default=300, help='hidden size')
    parser.add_argument("--vocab", type=str, default='./data/zinc/clique.txt', help='vocab path')
    parser.add_argument('--order', type=str, default="dfs",
                        help='motif tree generation order (bfs or dfs)')
    parser.add_argument('--seed', type=int, default=0,
                        help='setting seed number')
    #for wandb
    parser.add_argument('--wandb', action='store_true', default=False, help='add wandb log')
    parser.add_argument('--wandb_name', type=str, default = 'MGSSL_Grover', help='wandb name')
    #for grovermode
    parser.add_argument('--grover_dataset', action='store_true', default=False, help='grover dataset mode')
    parser.add_argument('--multi', action='store_true', default=False, help='use multiprocess mode')
    args = parser.parse_args()
    
    #for distributed
    #rank = int(os.environ["RANK"]) if args.multi else 0
    #local_rank = int(os.environ["LOCAL_RANK"]) if args.multi else 0
    #world_size = int(os.environ["WORLD_SIZE"]) if args.multi else 1
    ## binding training to GPUs.
    #master_worker = (rank == 0) if args.multi else True
    
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)
        
    logger = create_logger('pretrain', args.output_path)
    info = logger.info
    
    info(f'rank : {rank}')
    
    ddp_setup(rank, world_size)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    #device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        
    if args.grover_dataset:
        grover_data, sample_per_file = get_motif_data(data_path = args.dataset, logger=logger)
        train_dataset, val_dataset = split_data_grover(grover_data, sizes=(0.8,0.2,0), seed=args.seed, logger=logger)
        shared_dict = {}
        GMC = GroverMotifCollator(shared_dict=shared_dict, args=args)
        
        pre_load_data(dataset=train_dataset, rank=rank, num_replicas=world_size, sample_per_file=sample_per_file, logger=logger)
        pre_load_data(dataset=val_dataset, rank=rank, num_replicas=world_size, sample_per_file=sample_per_file, logger=logger)
        
        train_sampler = DistributedSampler(dataset=train_dataset, num_replicas=world_size, rank=rank, shuffle=True, sample_per_file=sample_per_file)
        val_sampler = DistributedSampler(dataset=val_dataset, num_replicas=world_size, rank=rank, shuffle=False, sample_per_file=sample_per_file)
        train_sampler.set_epoch(args.epochs)
        val_sampler.set_epoch(1)
        
        #train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=GMC)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=GMC, sampler=train_sampler)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=GMC, sampler=val_sampler)
        
    else : 
        dataset = MoleculeDataset_grover(args.dataset)
        train_dataset, val_dataset = train_test_split(dataset, test_size=0.1, random_state=args.seed)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=lambda x:x)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=lambda x:x)

    model = GNN(5, args.emb_dim, JK='last', drop_ratio=args.dropout_ratio, gnn_type='gin').to(rank)
    if os.path.exists(args.input_model_file):
        model.load_state_dict(torch.load(args.input_model_file))
        
    model = DDP(model, device_ids = [rank])

    vocab = [x.strip("\r\n ") for x in open(args.vocab)]
    vocab = Vocab(vocab)
    motif_model = Motif_Generation_Grover(vocab, args.hidden_size, rank, args.order).to(rank)
    motif_model = DDP(motif_model, device_ids = [rank])
    
    model_list = [model, motif_model]
    optimizer_model = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_motif = optim.Adam(motif_model.parameters(), lr=args.lr, weight_decay=args.decay)

    optimizer_list = [optimizer_model, optimizer_motif]
    
    if args.wandb :
        wandb.init(project=args.wandb_name)
        wandb.config = args
        #wandb.watch(model)

    #train start
    best_val_loss = 1e+10
    for epoch in range(1, args.epochs + 1):
        info("====epoch " + str(epoch))
        train_sampler.set_epoch(epoch)
        
        idxs = train_sampler.get_indices()
        for local_rank in idxs:
            train_dataset.load_data(local_rank)
    
        
        #training
        train_start = time.time()
        train_loss, train_node_loss, train_topo_loss, train_node_acc, train_topo_acc = train(args, model_list, train_loader, optimizer_list, rank)
        train_end = time.time() - train_start
        info(f'epoch : {epoch:04d} train_loss : {train_loss:.4f} train_node_loss : {train_node_loss:.4f} train_topo_loss : {train_topo_loss:.4f} train_node_acc : {train_node_acc:.2f} train_topo_acc : {train_topo_acc:.2f} train_time : {train_end:.2f}s')
        
        #validation
        val_start = time.time()
        val_loss, val_node_loss, val_topo_loss, val_node_acc, val_topo_acc = validation(args, model_list, val_loader, rank)
        val_end = time.time() - val_start
        info(f'epoch : {epoch:04d} val_loss : {val_loss:.4f} val_node_loss : {val_node_loss:.4f} val_topo_loss : {val_topo_loss:.4f} val_node_acc : {val_node_acc:.2f} val_topo_acc : {val_topo_acc:.2f} val_tim : {val_end:.2f}s')
        
        if args.wandb :         
            wandb.log({"train_loss" : train_loss, "train_node_loss" : train_node_loss, "train_topo_loss" : train_topo_loss, 
                       "val_loss" : val_loss, "val_node_loss" : val_node_loss, "val_topo_loss" : val_topo_loss})
            
        #torch.save(model.state_dict(), os.path.join(args.output_path, 'temp.pth'))
        if rank==0:
            torch.save(model.module.state_dict(), os.path.join(args.output_path, 'temp.pth'))
            if best_val_loss > val_loss:
                torch.save(model.state_dict(), os.path.join(args.output_path, f'best.pth'))
            if epoch % 5 == 0:
                torch.save(model.state_dict(), os.path.join(args.output_path, f'{epoch}.pth'))
    
    info('all train clear')
    destroy_process_group()


if __name__ == "__main__":
    world_size = 2 #torch.cuda.device_count()-2
    mp.spawn(main, args=(world_size, ), nprocs=world_size)