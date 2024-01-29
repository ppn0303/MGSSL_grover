pytorch                   1.8.1     # 그냥 최신버전+cuda 맞게받았다. 파이토치 버전은 같은건데,,,          
torch-geometric           1.7.0  # https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html 여기서 
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.0+cu113.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-1.10.0+cu113.html
pip install torch-geometric

rdkit                     2020.09.1
tqdm                      4.31.1
tensorboardx              1.6

Lib - Multiprocessing - spawn에서 
def _main(fd):
    with os.fdopen(fd, 'rb', closefd=True) as from_parent:
        process.current_process()._inheriting = True
        #try:
            #preparation_data = reduction.pickle.load(from_parent)
            #prepare(preparation_data)
            #self = reduction.pickle.load(from_parent)
        #finally:
            #del process.current_process()._inheriting
    return #self._bootstrap()

이렇게 바꿈... 근데 이건 GPU다중처리 때문에 그런거 같으니깐 num_worker랑 GPU파트만지면 될듯

MGSSL - motif_based_pretrain - pretrain_motif에서
num_worker=8인걸 바꾸고, 

util폴더에서 nnutils에서 #"cuda:" + "1" 이걸 적절히 바꿔야하는듯...

#221211
def _load_sider_dataset(input_path):
(중략)
    return smiles_list, rdkit_mol_objs_list, labels.value
맨 뒤에 values여야함... 오타 ㅜㅜ
