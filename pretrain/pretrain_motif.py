import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torch.autograd import Variable

import math, random, sys
from tqdm import tqdm
import numpy as np
from optparse import OptionParser
from gnn_model import GNN, GNN_grover

sys.path.append('./util/')

from mol_tree import *
from nnutils import *
from datautils import *
from motif_generation import *

import rdkit

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

def train(args, model_list, loader, optimizer_list, device):
    model, motif_model = model_list                             # 훈련간 사용 모델은 GNN모델과 motif모델이다.
    optimizer_model, optimizer_motif = optimizer_list        # 옵티마이저도 둘에 대해 각각 사용하라.

    model.train()					#모델, 모티프 모델 훈련!
    motif_model.train()
    word_acc, topo_acc = 0, 0			# 분자와 위상 정확도 변수 설정
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):	# 데이터로더에서 순회 진행바 표시형태로 순회해서 step과 batch대로 반복하자

        batch_size = len(batch)

        graph_batch = moltree_to_graph_data(batch)		# 분자식을 파이토치 지오메트릭 패키지에서 요구되는 그래프 데이터 형태로 변경해서 배치단위로 저장   /datautils에 있음
        #store graph object data in the process stage	
        batch_index = graph_batch.batch.numpy()			# 배치내의 배치텐서를 넘파이로 인덱스에 넘겨라
        graph_batch = graph_batch.to(device)			# 그래프배치는 GPU로 되게
        node_rep = model(graph_batch.x, graph_batch.edge_index, graph_batch.edge_attr)	# GNN모델에 그래프(x, 엣지인덱스, 엣지의 특성) 투입
        node_rep = group_node_rep(node_rep, batch_index, batch_size)			# rep는 representation의 줄임말로 노드 표현을 의미
        loss, wacc, tacc = motif_model(batch, node_rep)		# motif모델에서 손실, motif정확도, 위상 정확도 출력

        optimizer_model.zero_grad()				#옵티마이저 0으로
        optimizer_motif.zero_grad()
	
        loss.backward()					#손실 역전파

        optimizer_model.step()				#옵티마이저 시행
        optimizer_motif.step()

        word_acc += wacc
        topo_acc += tacc					#위상 정확도

        if (step+1) % 20 == 0:
            word_acc = word_acc / 20 * 100
            topo_acc = topo_acc / 20 * 100
            print("Loss: %.1f, Word: %.2f, Topo: %.2f" % (loss, word_acc, topo_acc))
            word_acc, topo_acc = 0, 0

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
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
    parser.add_argument('--graph_pooling', type=str, default="mean",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--dataset', type=str, default='./data/zinc/all.txt',
                        help='root directory of dataset. For now, only classification.')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--input_model_file', type=str, default='./saved_model/init', help='filename to read the model (if there is any)')
    parser.add_argument('--output_model_file', type=str, default='./saved_model/motif_pretrain',
                        help='filename to output the pre-trained model')
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers for dataset loading')   #원래는 8이었음 오류로 0으로 바꿈
    parser.add_argument("--hidden_size", type=int, default=300, help='hidden size')
    parser.add_argument("--latent_size", type=int, default=56, help='latent size')
    parser.add_argument("--vocab", type=str, default='./data/zinc/clique.txt', help='vocab path')
    parser.add_argument('--order', type=str, default="bfs",
                        help='motif tree generation order (bfs or dfs)')
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    dataset = MoleculeDataset(args.dataset)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=lambda x:x, drop_last=True)

    model = GNN(5, args.hidden_size, JK='last', drop_ratio=0.2, gnn_type='gin').to(device)
    if not args.input_model_file == "":
        model.load_state_dict(torch.load(args.input_model_file + ".pth"))

    vocab = [x.strip("\r\n ") for x in open(args.vocab)]
    vocab = Vocab(vocab)
    motif_model = Motif_Generation(vocab, args.hidden_size, args.latent_size, 3, device, args.order).to(device)

    model_list = [model, motif_model]
    optimizer_model = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_motif = optim.Adam(motif_model.parameters(), lr=1e-3, weight_decay=args.decay)

    optimizer_list = [optimizer_model, optimizer_motif]

    for epoch in range(1, args.epochs + 1):
        print("====epoch " + str(epoch))

        train(args, model_list, loader, optimizer_list, device)

        if not args.output_model_file == "":
            torch.save(model.state_dict(), args.output_model_file + ".pth")


if __name__ == "__main__":
    main()
