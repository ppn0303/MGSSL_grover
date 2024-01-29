import argparse

from loader import MoleculeDataset, MoleculeDataset_other
from torch_geometric.data import DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np

from model import GNN, GNN_graphpred

from splitters import scaffold_split, random_split
import pandas as pd

import os
import shutil

from util import calcul_loss, save_cp, confusion_mat
from sklearn.metrics import roc_auc_score

criterion = nn.BCEWithLogitsLoss(reduction = "none")

def train(args, model, device, loader, optimizer):
    model.train()
    
    loss_sum = 0
    iter_count = 0
    
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        y = batch.y.view(pred.shape).to(torch.float64)

        #loss matrix after removing null target
        loss = calcul_loss(pred, y, criterion)
            
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        
        loss_sum += loss
        iter_count += 1
        
    torch.cuda.empty_cache()
    return loss_sum / iter_count



def valid(args, model, device, loader):
    model.eval()
    y_true = []
    y_scores = []
    cum_loss = 0

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            y = batch.y.view(pred.shape)
            loss = calcul_loss(pred, y, criterion)
        
        cum_loss += loss
        y_true.append(y)
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim = 0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim = 0).cpu().numpy()

    roc_list = []
    for i in range(y_true.shape[1]):
        #AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == -1) > 0:
            is_valid = y_true[:,i]**2 > 0
            roc_list.append(roc_auc_score((y_true[is_valid,i] + 1)/2, y_scores[is_valid,i]))

    if len(roc_list) < y_true.shape[1]:
        print("Some target is missing!")
        print("Missing ratio: %f" %(1 - float(len(roc_list))/y_true.shape[1]))
        
    torch.cuda.empty_cache()
    return cum_loss, sum(roc_list)/len(roc_list) #y_true.shape[1]

def test(args, model, device, loader):
    
    model.eval()
    y_true = []
    y_scores = []
    cum_loss = 0

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            y = batch.y.view(pred.shape)
            loss = calcul_loss(pred, y, criterion)

        y_true.append(y)
        y_scores.append(pred)
        cum_loss += loss

    y_true = torch.cat(y_true, dim = 0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim = 0).cpu().numpy()

    auc_list = []
    acc_list = []
    rec_list = []
    prec_list = []
    f1s_list = []
    BA_list = []
    tp_list = []
    fp_list = []
    tn_list = []
    fn_list = []
    for i in range(y_true.shape[1]):
        auc, acc, rec, prec, f1s, BA, tp, fp, tn, fn = confusion_mat(y_true[:,i], y_scores[:,i])
        auc_list.append(auc)
        acc_list.append(acc)
        rec_list.append(rec)
        prec_list.append(prec)
        f1s_list.append(f1s)
        BA_list.append(BA)
        tp_list.append(tp)
        fp_list.append(fp)
        tn_list.append(tn)
        fn_list.append(fn)

    torch.cuda.empty_cache()
    return cum_loss, auc_list, acc_list, rec_list, prec_list, f1s_list, BA_list, tp_list, fp_list, tn_list, fn_list


def cross_validate(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        
    #set up dataset
    dataset = MoleculeDataset_other("dataset/" + args.dataset, dataset=args.dataset)
    num_tasks = len(dataset[0]['y'])

    
    if args.split == "scaffold":
        smiles_list = pd.read_csv('dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1)
        print(f'scaffold_balanced_split')
    elif args.split == "random":
        train_dataset, valid_dataset, test_dataset = random_split(dataset, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed = args.seed)
        print("random")
    elif args.split == "random_scaffold":
        smiles_list = pd.read_csv('dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = random_scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed = args.seed)
        print("random scaffold")
    else:
        raise ValueError("Invalid split option.")
        
    print(f'total_size:{len(dataset)} train_size:{len(train_dataset)} val_size:{len(valid_dataset)} test_size:{len(test_dataset)}')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)

    #set up model
    model = GNN_graphpred(args.num_layer, args.emb_dim, num_tasks, JK = args.JK, drop_ratio = args.dropout_ratio, graph_pooling = args.graph_pooling, gnn_type = args.gnn_type)
    if not args.input_model_file == "":
        model.from_pretrained(args.input_model_file)
    
    model.to(device)

    #set up optimizer
    #different learning rate for different part of GNN
    model_param_group = []
    model_param_group.append({"params": model.gnn.parameters()})
    if args.graph_pooling == "attention":
        model_param_group.append({"params": model.pool.parameters(), "lr":args.lr*args.lr_scale})
    model_param_group.append({"params": model.graph_pred_linear.parameters(), "lr":args.lr*args.lr_scale})
    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
    print(optimizer)

    best_val_loss = 9999
    best_model_path = os.path.join(args.output_path, str(args.seed))
    for epoch in range(1, args.epochs+1):
        print("====epoch " + str(epoch))
        
        train_loss = train(args, model, device, train_loader, optimizer)

        print("====Evaluation")
        val_loss, val_auc = valid(args, model, device, val_loader)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_cp(args, model, path=best_model_path)
        print(f'train_loss:{train_loss:.4f} val_loss:{val_loss:.4f} val_auc:{val_auc:.4f}')
    
    best_state = torch.load(os.path.join(best_model_path,'model.pt'))
    model.load_state_dict(best_state['state_dict'])
    
    test_loss, auc, acc, rec, prec, f1s, BA, tp, fp, tn, fn = test(args, model, device, test_loader)
    avg_auc = sum(auc)/num_tasks
    avg_acc = sum(acc)/num_tasks
    avg_rec = sum(rec)/num_tasks
    avg_prec = sum(prec)/num_tasks
    avg_f1s = sum(f1s)/num_tasks
    avg_BA = sum(BA)/num_tasks
    avg_tp = sum(tp)/num_tasks
    avg_fp = sum(fp)/num_tasks
    avg_tn = sum(tn)/num_tasks
    avg_fn = sum(fn)/num_tasks
        
    print(f'seed:{args.seed} loss:{test_loss} auc:{avg_auc} acc:{avg_acc} rec:{avg_rec} prec:{avg_prec} f1:{avg_f1s} BA:{avg_BA}\ntp:{avg_tp} fp:{avg_fp} fn:{avg_fn} tn:{avg_tn}')
    #delete for memory
    del train_dataset, valid_dataset, test_dataset, train_loader, val_loader, test_loader

    return avg_auc, avg_acc, avg_rec, avg_prec, avg_f1s, avg_BA, avg_tp, avg_fp, avg_tn, avg_fn

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--lr_scale', type=float, default=1,
                        help='relative learning rate for the feature extraction layer (default: 1)')
    parser.add_argument('--decay', type=float, default=1e-7,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0.2,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--graph_pooling', type=str, default="mean",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--dataset', type=str, default = 'sider', help='root directory of dataset. For now, only classification.')
    parser.add_argument('--input_model_file', type=str, default = 'pretrained.pth', help='filename to read the model (if there is any)')
    parser.add_argument('--output_path', type=str, default = 'output', help='output filename')
    parser.add_argument('--seed', type=int, default=0, help = "Seed for splitting the dataset.")
    parser.add_argument('--runseed', type=int, default=0, help = "Seed for minibatch selection, random initialization.")
    parser.add_argument('--split', type = str, default="scaffold", help = "random or scaffold or random_scaffold")
    parser.add_argument('--eval_train', type=int, default = 1, help='evaluating training or not')
    parser.add_argument('--num_workers', type=int, default = 4, help='number of workers for dataset loading')
    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    auc_list = []
    acc_list = []
    rec_list = []
    prec_list = []
    f1s_list = []
    BA_list = []
    tp_list = []
    fp_list = []
    tn_list = []
    fn_list = []
    for k in range(3):
        auc, acc, rec, prec, f1s, BA, tp, fp, tn, fn = cross_validate(args)
        auc_list.append(auc)
        acc_list.append(acc)
        rec_list.append(rec)
        prec_list.append(prec)
        f1s_list.append(f1s)
        BA_list.append(BA)
        tp_list.append(tp)
        fp_list.append(fp)
        tn_list.append(tn)
        fn_list.append(fn)
        args.seed += 1
    print(f'all test end')
    print(f'overall test_auc : {np.nanmean(auc_list):.4f}\nstd={np.nanstd(auc_list):.4f}')
    print(f'overall test_accuracy : {np.nanmean(acc_list):.4f}\nstd={np.nanstd(acc_list):.4f}')
    print(f'overall test_recall : {np.nanmean(rec_list):.4f}\nstd={np.nanstd(rec_list):.4f}')
    print(f'overall test_precision : {np.nanmean(prec_list):.4f}\nstd={np.nanstd(prec_list):.4f}')
    print(f'overall test_f1score : {np.nanmean(f1s_list):.4f}\nstd={np.nanstd(f1s_list):.4f}')
    print(f'overall test_Balanced_Accuracy : {np.nanmean(BA_list):.4f}\nstd={np.nanstd(BA_list):.4f}')
    print(f'overall test_tp : {np.nanmean(tp_list):.2f}\nstd={np.nanstd(tp_list):.2f}')
    print(f'overall test_fp : {np.nanmean(fp_list):.2f}\nstd={np.nanstd(fp_list):.2f}')
    print(f'overall test_fn : {np.nanmean(fn_list):.2f}\nstd={np.nanstd(fn_list):.2f}')
    print(f'overall test_tn : {np.nanmean(tn_list):.2f}\nstd={np.nanstd(tn_list):.2f}')
    print(f'\n       (pred)pos    neg(pred)')
    print(f'pos(true)    {tp:.2f}  {fn:.2f}')
    print(f'neg(true)    {fp:.2f}  {tn:.2f}')

if __name__ == "__main__":
    main()
