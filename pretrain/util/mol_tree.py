import rdkit
import rdkit.Chem as Chem
import numpy as np
import copy
from util.chemutils import get_clique_mol, tree_decomp, brics_decomp, get_mol, get_smiles, set_atommap, enum_assemble, decode_stereo

def get_slots(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return [(atom.GetSymbol(), atom.GetFormalCharge(), atom.GetTotalNumHs()) for atom in mol.GetAtoms()]

class Vocab(object):

    def __init__(self, smiles_list):
        self.vocab = smiles_list
        self.vmap = {x:i for i,x in enumerate(self.vocab)}
        self.slots = [get_slots(smiles) for smiles in self.vocab]
        
    def get_index(self, smiles):
        return self.vmap[smiles]

    def get_smiles(self, idx):
        return self.vocab[idx]

    def get_slots(self, idx):
        return copy.deepcopy(self.slots[idx])

    def size(self):
        return len(self.vocab)

class MolTreeNode(object):

    def __init__(self, smiles, clique=[]):   #스마일식을 받고, clique(무리, 군벌 등을 의미, 여기선 원자집단 말하는 듯)라는 벡터를 받아라
        self.smiles = smiles
        self.mol = get_mol(self.smiles)  #스마일식에서 분자를 얻어내고, 이중결합등이 명시된 결과를 반환하라.
        #self.mol = cmol

        self.clique = [x for x in clique] #copy
        self.neighbors = []
        
    def add_neighbor(self, nei_node):		#인접 노드들을 이웃 변수에 추가해라
        self.neighbors.append(nei_node)	

    def recover(self, original_mol):	
        clique = []
        clique.extend(self.clique)		# clique에 자신의 clique리스트들을 추가
        if not self.is_leaf:
            for cidx in self.clique:
                original_mol.GetAtomWithIdx(cidx).SetAtomMapNum(self.nid)	#Clique들에서 원자들의 id를 가져오고, 원자맵 숫자를 지정하라.

        for nei_node in self.neighbors:
            clique.extend(nei_node.clique)		# 위의 화학식 리스트들에 인접노드들을 추가해라
            if nei_node.is_leaf: #Leaf node, no need to mark(끄트머리꺼면 마킹할 필요 없어)
                continue
            for cidx in nei_node.clique:			# 인접노드의 분자식들에서 원자 id가져오고, 원자맵 숫자 지정(위에꺼)
                #allow singleton node override the atom mapping
                if cidx not in self.clique or len(nei_node.clique) == 1:
                    atom = original_mol.GetAtomWithIdx(cidx)
                    atom.SetAtomMapNum(nei_node.nid)

        clique = list(set(clique))		# n노드와 인접 노드들의 분자식
        label_mol = get_clique_mol(original_mol, clique)	# 그들의 레이블
        self.label = Chem.MolToSmiles(Chem.MolFromSmiles(get_smiles(label_mol)))	#아니 도대체 왜 스마일식을 얻고, 이걸 원자로 바꾸고, 다시 원자를 스마일 식으로 바꾸냐고...
        self.label_mol = get_mol(self.label)

        for cidx in clique:
            original_mol.GetAtomWithIdx(cidx).SetAtomMapNum(0)		#원본 분자의 원자 id를 얻고, 원자맵 숫자는 0으로

        return self.label		#레이블 스마일식을 레이블로 반환하라
    
    def assemble(self):
        neighbors = [nei for nei in self.neighbors if nei.mol.GetNumAtoms() > 1]
        neighbors = sorted(neighbors, key=lambda x:x.mol.GetNumAtoms(), reverse=True)
        singletons = [nei for nei in self.neighbors if nei.mol.GetNumAtoms() == 1]
        neighbors = singletons + neighbors

        cands = enum_assemble(self, neighbors)
        if len(cands) > 0:
            self.cands, self.cand_mols, _ = zip(*cands)
            self.cands = list(self.cands)
            self.cand_mols = list(self.cand_mols)
        else:
            self.cands = []
            self.cand_mols = []

class MolTree(object):

    def __init__(self, smiles):
        self.smiles = smiles
        self.mol = get_mol(smiles)

        '''
        #Stereo Generation
        mol = Chem.MolFromSmiles(smiles)
        self.smiles3D = Chem.MolToSmiles(mol, isomericSmiles=True)
        self.smiles2D = Chem.MolToSmiles(mol)
        self.stereo_cands = decode_stereo(self.smiles2D)
        '''

        cliques, edges = brics_decomp(self.mol)
        if len(edges) <= 1:
            cliques, edges = tree_decomp(self.mol)
        self.nodes = []
        root = 0
        for i,c in enumerate(cliques):
            cmol = get_clique_mol(self.mol, c)
            node = MolTreeNode(get_smiles(cmol), c)
            self.nodes.append(node)
            if min(c) == 0:
                root = i

        for x,y in edges:
            self.nodes[x].add_neighbor(self.nodes[y])
            self.nodes[y].add_neighbor(self.nodes[x])
        
        if root > 0:
            self.nodes[0],self.nodes[root] = self.nodes[root],self.nodes[0]

        for i,node in enumerate(self.nodes):
            node.nid = i + 1
            if len(node.neighbors) > 1: #Leaf node mol is not marked
                set_atommap(node.mol, node.nid)
            node.is_leaf = (len(node.neighbors) == 1)

    def size(self):
        return len(self.nodes)

    def recover(self):
        for node in self.nodes:
            node.recover(self.mol)

    def assemble(self):
        for node in self.nodes:
            node.assemble()

if __name__ == "__main__":
    import sys
    import csv
    lg = rdkit.RDLogger.logger() 
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    cset = set()
    counts = {}

    print("start")
    with open("../data/zinc/all.txt", "r") as f:
        for line in f.readlines():
            line = line.strip('\n')
            print(line)
            mol = MolTree(line)
            for c in mol.nodes:
                cset.add(c.smiles)
                if c.smiles not in counts:
                    counts[c.smiles] = 1
                else:
                    counts[c.smiles] += 1
    print("Preprocessing Completed!")
    clique_list = list(cset)
    with open('../data/zinc/clique.txt', 'w') as file:
        for c in clique_list:
            file.write(c)
            file.write('\n')


