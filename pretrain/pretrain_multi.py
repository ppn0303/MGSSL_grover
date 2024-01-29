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

sys.path.append('./util')
sys.path.append('./grover')

from util.mol_tree import *
from util.nnutils import *
from util.datautils import *
from util.motif_generation import *

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

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)

def parse_args():
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
    parser.add_argument('--dropout_ratio', type=float, default=0.1,
                        help='dropout ratio (default: 0.1)')
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
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers for dataset loading')  
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
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--master_worker', action='store_true', default=True)
    

    args = parser.parse_args()
    return args

def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def group_node_rep(node_rep, batch_index, batch_size):
    group = []
    count = 0
    for i in range(batch_size):
        num = sum(batch_index == i)
        group.append(node_rep[count:count + num])
        count += num
    return group

def train(args, logger, model_list, loader, optimizer_list, epoch, best_val_loss, resume_batch=0):
    model, motif_model = model_list
    optimizer_model, optimizer_motif = optimizer_list

    model.train()
    motif_model.train()
    word_acc, topo_acc = 0, 0
    starting = False
    for step, batch in enumerate(loader):
        if step==resume_batch:
            starting=True
        if starting : 
            batch_size = len(batch)
            
            if args.grover_dataset:
                graph_batch = moltree_to_grover_data(batch)
            else:
                graph_batch = moltree_to_graph_data(batch)
            batch_index = graph_batch.batch.numpy()
            graph_batch = graph_batch.to(args.rank)
            node_rep = model(graph_batch.x, graph_batch.edge_index, graph_batch.edge_attr)
            node_rep = group_node_rep(node_rep, batch_index, batch_size)
            loss, word_loss, topo_loss, word_acc, topo_acc = motif_model(batch, node_rep)
            
            optimizer_model.zero_grad()
            optimizer_motif.zero_grad()
            
            loss.backward()
    
            optimizer_model.step()
            optimizer_motif.step()

            if step%50==0:
                if args.master_worker:
                    logger.info(f'epoch : {epoch:04d} step : {step:04d} train_loss : {loss:.4f} train_node_loss : {word_loss:.4f} train_topo_loss : {topo_loss:.4f} train_node_acc : {word_acc:.2f} train_topo_acc : {topo_acc:.2f}')
                    save_cp(model, motif_model, optimizer_model, optimizer_motif, os.path.join(args.output_path, 'temp.pth'), epoch, step+1, best_val_loss, args.multi)
    torch.cuda.empty_cache()

def validation(args, model_list, loader):
    model, motif_model = model_list

    model.eval()
    motif_model.eval()
    loss_sum, word_loss_sum, topo_loss_sum, word_acc, topo_acc = 0, 0, 0, 0, 0
    for step, batch in enumerate(loader):
        with torch.no_grad():
            batch_size = len(batch)

            if args.grover_dataset:
                graph_batch = moltree_to_grover_data(batch)
            else:
                graph_batch = moltree_to_graph_data(batch)	
            batch_index = graph_batch.batch.numpy()			
            graph_batch = graph_batch.to(args.rank)			
            node_rep = model(graph_batch.x, graph_batch.edge_index, graph_batch.edge_attr)	
            node_rep = group_node_rep(node_rep, batch_index, batch_size)			
            loss, word_loss, topo_loss, wacc, tacc = motif_model(batch, node_rep)		
    
            loss_sum += loss
            word_loss_sum += word_loss
            topo_loss_sum += topo_loss_sum
            word_acc += wacc
            topo_acc += tacc					
        torch.cuda.empty_cache()    

    step += 1
    loss_sum /= step
    word_loss_sum /= step
    topo_loss_sum /= step
    
    return loss_sum, word_loss_sum, topo_loss_sum, word_acc, topo_acc


def main():    
    args = parse_args()
    args.rank = int(os.environ["LOCAL_RANK"]) if args.multi else 0
    args.master_worker = (args.rank == 0) if args.multi else True
    if args.master_worker : 
        if not os.path.exists(args.output_path):
            os.mkdir(args.output_path)
    logger = create_logger('pretrain', args)
    debug = logger.debug
    info = logger.info

    #for distributed
    ddp_setup()
    world_size = int(os.environ["WORLD_SIZE"]) if args.multi else 1
    
    if args.master_worker : 
        info(f'emb_dim : {args.emb_dim}, lr : {args.lr}, dropout : {args.dropout_ratio}, batch_size : {args.batch_size}')
    info(f'rank : {args.rank}')
    
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    #device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        
    vocab = [x.strip("\r\n ") for x in open(args.vocab)]
    vocab = Vocab(vocab)

    if args.grover_dataset:
        grover_data, sample_per_file = get_motif_data(data_path = args.dataset, logger=logger)
        train_dataset, val_dataset = split_data_grover(grover_data, sizes=(0.9,0.1,0), seed=args.seed, logger=logger)
        shared_dict = {}
        GMC = GroverMotifCollator(shared_dict=shared_dict, args=args)
        
        train_sampler = DistributedSampler(dataset=train_dataset, num_replicas=world_size, rank=args.rank, shuffle=True, sample_per_file=sample_per_file)
        val_sampler = DistributedSampler(dataset=val_dataset, num_replicas=world_size, rank=args.rank, shuffle=False, sample_per_file=sample_per_file)
        train_sampler.set_epoch(args.epochs)
        val_sampler.set_epoch(1)
        idxs = val_sampler.get_indices()
        for local_rank in idxs:
            val_dataset.load_data(local_rank)

        pre_load_data(dataset=train_dataset, rank=args.rank, num_replicas=world_size, sample_per_file=sample_per_file, logger=logger)
        pre_load_data(dataset=val_dataset, rank=args.rank, num_replicas=world_size, logger=logger)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=GMC, sampler=train_sampler)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=GMC, sampler=val_sampler)
        
        model = GNN_grover(5, args.emb_dim, JK='last', drop_ratio=args.dropout_ratio, gnn_type='gin').to(args.rank)        
        motif_model = Motif_Generation_Grover(vocab, args.hidden_size, args.rank, args.order).to(args.rank)

    else : 
        dataset = MoleculeDataset_grover(args.dataset)
        train_dataset, val_dataset = train_test_split(dataset, test_size=0.1, random_state=args.seed)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=lambda x:x)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=lambda x:x)
        model = GNN(5, args.emb_dim, JK='last', drop_ratio=args.dropout_ratio, gnn_type='gin').to(args.rank)  
        motif_model = Motif_Generation(vocab, args.hidden_size, args.rank, args.order).to(args.rank)

  
    model = DDP(model, device_ids = [args.rank])
    motif_model = DDP(motif_model, device_ids = [args.rank])

    optimizer_model = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_motif = optim.Adam(motif_model.parameters(), lr=args.lr, weight_decay=args.decay)

    cp_path = f'{args.output_path}/temp.pth'
    if os.path.exists(cp_path):
        resume_epoch, resume_batch, best_val_loss = load_cp(model, motif_model, optimizer_model, optimizer_motif, cp_path, args.multi)
        if args.master_worker:
            info(f'load checkpoint : {resume_epoch}epoch, batch : {resume_batch}')
        else : 
            debug(f'rank : {args.rank} load checkpoint : {resume_epoch}epoch, batch : {resume_batch}')
    else : 
        resume_epoch = 0
        resume_batch = 0
        best_val_loss = 1e+10
        
    model_list = [model, motif_model]
    optimizer_list = [optimizer_model, optimizer_motif]
    
    if args.wandb :
        wandb.init(project=args.wandb_name)
        wandb.config = args
        #wandb.watch(model)

    #train start
    for epoch in range(resume_epoch, args.epochs):
        if args.master_worker:info("====epoch " + str(epoch))

        train_sampler.set_epoch(epoch)
        train_dataset.clean_cache()
        idxs = train_sampler.get_indices()
        for local_rank in idxs:
            train_dataset.load_data(local_rank)

        #training
        train_start = time.time()
        train_loss = train(args, logger, model_list, train_loader, optimizer_list, epoch, best_val_loss, resume_batch=resume_batch)
        train_end = time.time() - train_start
        if args.master_worker : 
            info(f'train_time : {train_end:.2f}s')
        
        #validation
        val_start = time.time()
        val_loss, val_node_loss, val_topo_loss, val_node_acc, val_topo_acc = validation(args, model_list, val_loader)
        val_end = time.time() - val_start
        if args.master_worker : 
            info(f'epoch : {epoch:04d} val_loss : {val_loss:.4f} val_node_loss : {val_node_loss:.4f} val_topo_loss : {val_topo_loss:.4f} val_node_acc : {val_node_acc:.2f} val_topo_acc : {val_topo_acc:.2f} val_time : {val_end:.2f}s')
        else : 
            debug(f'rank {args.rank} epoch : {epoch:04d} val_loss : {val_loss:.4f} val_node_loss : {val_node_loss:.4f} val_topo_loss : {val_topo_loss:.4f} val_node_acc : {val_node_acc:.2f} val_topo_acc : {val_topo_acc:.2f} val_time : {val_end:.2f}s')

        
        if args.wandb :         
            wandb.log({"val_loss" : val_loss, "val_node_loss" : val_node_loss, "val_topo_loss" : val_topo_loss})
            
        if args.master_worker:
            save_cp(model, motif_model, optimizer_model, optimizer_motif, os.path.join(args.output_path, 'temp.pth'), epoch+1, 0, best_val_loss, args.multi)
            if best_val_loss > val_loss:
                info('best model saved')
                save_cp(model, motif_model, optimizer_model, optimizer_motif, os.path.join(args.output_path, 'best.pth'), epoch+1, 0, best_val_loss, args.multi)
            if epoch % 5 == 0:
                info(f'{epoch}th model saved')
                save_cp(model, motif_model, optimizer_model, optimizer_motif, os.path.join(args.output_path, f'{epoch}.pth'), epoch+1, 0, best_val_loss, args.multi)
    
    destroy_process_group()
    info('all train clear')


if __name__ == "__main__":
    main()