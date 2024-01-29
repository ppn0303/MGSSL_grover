import argparse
import sys
import csv
import time

sys.path.append('./util/')
from mol_tree import *

parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
parser.add_argument('--datapath', type=str, default='./data/zinc/all.txt',
                        help='root directory of dataset. For now, only classification.')
parser.add_argument('--output_path', type=str, default='./clique.txt',
                        help='filename to output the pre-trained model')
args = parser.parse_args()


lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

cset = set()
counts = {}
num=0

print("start")
s_time = time.time()
with open(args.datapath, "r") as file:
    data_length = len(file.readlines())
    print(f'data size is {data_length}')
file.close()

with open(args.datapath, "r") as f:
    for line in f.readlines():
        if num%10000==0:print(f'process : {num} / {data_length}')
        line = line.strip('\n')
        if line=="smiles":continue
        
        print(line)
        mol = MolTree(line)
        for c in mol.nodes:
            cset.add(c.smiles)
            if c.smiles not in counts:
                counts[c.smiles] = 1
            else:
                counts[c.smiles] += 1

        try : 
            Mol = Chem.MolFromSmiles(mol.smiles)
            Mol.GetNumHeavyAtoms()
        except AttributeError :
            print(f'{line} has Error in GetNumHeavyAtoms')

        num+=1

print("Preprocessing Completed!")
t_time = time.time() - s_time
print(f'total time is {t_time:.4f}s')

clique_list = list(cset)

with open(args.output_path, 'w') as file:
    for c in clique_list:
        file.write(c)
        file.write('\n')