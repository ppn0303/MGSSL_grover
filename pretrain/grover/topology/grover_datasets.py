import numpy as np
import math, time, os, pickle, csv
import logging

import torch
from torch.utils.data.sampler import Sampler
import torch.distributed as dist
from torch.utils.data import Dataset
from torch.utils.data.dataset import Dataset
from torch_geometric.data import Batch
from torch_geometric.data import Data

from mol_tree import MolTree

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem

from typing import Union, List
from argparse import Namespace
from grover.topology.mol_tree import *

class DistributedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, sample_per_file=None):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.sample_per_file = sample_per_file
        self.shuffle = shuffle

    def get_indices(self):

        indices = list(range(len(self.dataset)))

        if self.sample_per_file is not None:
            indices = self.sub_indices_of_rank(indices)
        else:
            # add extra samples to make it evenly divisible
            indices += indices[:(self.total_size - len(indices))]
            assert len(indices) == self.total_size
            # subsample
            s = self.rank * self.num_samples
            e = min((self.rank + 1) * self.num_samples, len(indices))

            # indices = indices[self.rank:self.total_size:self.num_replicas]
            indices = indices[s:e]

        if self.shuffle:
            g = torch.Generator()
            # the seed need to be considered.
            new_seed = (self.epoch + 1) * (self.rank + 1) * np.long(time.time())
            
            g.manual_seed(new_seed)
            idx = torch.randperm(len(indices), generator=g).tolist()
            indices = [indices[i] for i in idx]

        # disable this since sub_indices_of_rank.
        # assert len(indices) == self.num_samples

        return indices

    def sub_indices_of_rank(self, indices):

        # fix generator for each epoch
        g = torch.Generator()
        # All data should be loaded in each epoch.
        g.manual_seed((self.epoch + 1) * 2 + 3)

        # the fake file indices to cache
        f_indices = list(range(int(math.ceil(len(indices) * 1.0 / self.sample_per_file))))
        idx = torch.randperm(len(f_indices), generator=g).tolist()
        f_indices = [f_indices[i] for i in idx]

        file_per_rank = int(math.ceil(len(f_indices) * 1.0 / self.num_replicas))
        # add extra fake file to make it evenly divisible
        f_indices += f_indices[:(file_per_rank * self.num_replicas - len(f_indices))]

        # divide index by rank
        rank_s = self.rank * file_per_rank
        rank_e = min((self.rank + 1) * file_per_rank, len(f_indices))

        # get file index for this rank
        f_indices = f_indices[rank_s:rank_e]
        res_indices = []
        for fi in f_indices:
            # get real indices for this rank
            si = fi * self.sample_per_file
            ei = min((fi + 1) * self.sample_per_file, len(indices))
            cur_idx = [indices[i] for i in range(si, ei)]
            res_indices += cur_idx

        self.num_samples = len(res_indices)
        return res_indices

    def __iter__(self):
        return iter(self.get_indices())

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
        
class MoleculeDatapoint_motif:
    """A MoleculeDatapoint contains a single molecule and its associated features and targets."""

    def __init__(self,
                 line: List[str],
                 args: Namespace = None,
                 moltrees: object = None,
                 use_compound_names: bool = False):
        """
        Initializes a MoleculeDatapoint, which contains a single molecule.

        :param line: A list of strings generated by separating a line in a data CSV file by comma.
        :param args: Arguments.
        :param features: A numpy array containing additional features (ex. Morgan fingerprint).
        :param use_compound_names: Whether the data CSV includes the compound name on each line.
        """
        self.args = None
        if args is not None:
            self.args = args

        self.moltrees = moltrees

        if use_compound_names:
            self.compound_name = line[0]  # str
            line = line[1:]
        else:
            self.compound_name = None

        self.smiles = line[0]  # str

        # Create targets
        self.targets = [float(x) if x != '' else None for x in line[1:]]
        
    def set_moltrees(self, moltrees: list):
        """
        Sets the moltree of the molecule.

        :param moltree: moltree object
        """
        self.moltrees = moltrees
        
    def clean_moltree(self):
        """
        clean moltree for memory
        """
        self.moltrees = None

    def num_tasks(self) -> int:
        """
        Returns the number of prediction tasks.

        :return: The number of tasks.
        """
        return len(self.targets)

    def set_targets(self, targets: List[float]):
        """
        Sets the targets of a molecule.

        :param targets: A list of floats containing the targets.
        """
        self.targets = targets

class BatchDatapoint_motif:
    def __init__(self,
                 smiles_file,
                 moltree_file,
                 n_samples,
                 ):
        self.smiles_file = smiles_file
        self.moltree_file = moltree_file
        # deal with the last batch graph numbers.
        self.n_samples = n_samples
        self.datapoints = None

    def load_datapoints(self):
        moltrees = self.load_moltree()
        self.datapoints = []

        with open(self.smiles_file) as f:
            reader = csv.reader(f)
            next(reader)
            for i, line in enumerate(reader):
                # line = line[0]
                d = MoleculeDatapoint_motif(line=line,
                                      moltrees=moltrees[i])
                self.datapoints.append(d)
        f.close()

        assert len(self.datapoints) == self.n_samples
    
    def load_moltree(self):
        with open(self.moltree_file, 'rb') as f:
            moltrees = pickle.load(f)            
        return moltrees

    def shuffle(self):
        pass

    def clean_cache(self):
        del self.datapoints
        self.datapoints = None

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        assert self.datapoints is not None
        return self.datapoints[idx]

    def is_loaded(self):
        return self.datapoints is not None

class BatchMolDataset_motif(Dataset):
    def __init__(self, data: List[BatchDatapoint_motif],
                 graph_per_file=None):
        self.data = data

        self.len = 0
        for d in self.data:
            self.len += len(d)
        if graph_per_file is not None:
            self.sample_per_file = graph_per_file
        else:
            self.sample_per_file = len(self.data[0]) if len(self.data) != 0 else None

    def shuffle(self, seed: int = None):
        pass

    def clean_cache(self):
        for d in self.data:
            d.clean_cache()

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, idx) -> Union[MoleculeDatapoint_motif, List[MoleculeDatapoint_motif]]:
        # print(idx)
        dp_idx = int(idx / self.sample_per_file)
        real_idx = idx % self.sample_per_file
        return self.data[dp_idx][real_idx]

    def load_data(self, idx):
        dp_idx = int(idx / self.sample_per_file)
        if not self.data[dp_idx].is_loaded():
            self.data[dp_idx].load_datapoints()

    def count_loaded_datapoints(self):
        res = 0
        for d in self.data:
            if d.is_loaded():
                res += 1
        return res
    
def get_motif_data(data_path, logger=None):
    """
    Load data from the data_path.
    :param data_path: the data_path.
    :param logger: the logger.
    :return:
    """
    info = logger.info if logger is not None else print
    summary_path = os.path.join(data_path, "summary.txt")
    smiles_path = os.path.join(data_path, "graph")
    moltree_path = os.path.join(data_path, "moltrees")

    fin = open(summary_path)
    n_files = int(fin.readline().strip().split(":")[-1])
    n_samples = int(fin.readline().strip().split(":")[-1])
    sample_per_file = int(fin.readline().strip().split(":")[-1])
    info("Loading data:")
    info("Number of files: %d" % n_files)
    info("Number of samples: %d" % n_samples)
    info("Samples/file: %d" % sample_per_file)

    datapoints = []
    for i in range(n_files):
        smiles_path_i = os.path.join(smiles_path, str(i) + ".csv")
        moltree_path_i = os.path.join(moltree_path, str(i) + ".p")
        if i != (n_files-1):
            n_samples_i = sample_per_file
        elif n_samples % sample_per_file == 0:
            n_samples_i = sample_per_file
        else : 
            n_samples_i = n_samples % sample_per_file
        #n_samples_i = sample_per_file if i != (n_files - 1) else n_samples % sample_per_file
        datapoints.append(BatchDatapoint_motif(smiles_path_i, moltree_path_i, n_samples_i))
    return BatchMolDataset_motif(datapoints), sample_per_file

class GroverMotifCollator(object):
    def __init__(self, shared_dict, args):
        self.args = args
        self.shared_dict = shared_dict

    def __call__(self, batch):
        smiles_batch = [d.smiles for d in batch] # 여기서 말하는 batch는 batchmoldataset_motif다 그리고 d는 batchdatapoint_motif고
        #batchgraph = mol2graph(smiles_batch, self.shared_dict, self.args).get_components()

        #fgroup_label = torch.Tensor(np.array([d.features for d in batch])).float()
        moltree_batch = [d.moltrees for d in batch]
        
        # may be some mask here

        return moltree_batch
    
def split_data_grover(data,
               split_type='random',
               sizes=(0.8, 0.1, 0.1),
               seed=0,
               logger=None):
    """
    Split data with given train/validation/test ratio.
    :param data:
    :param split_type:
    :param sizes:
    :param seed:
    :param logger:
    :return:
    """
    assert len(sizes) == 3 and sum(sizes) == 1

    if split_type == "random":
        data.shuffle(seed=seed)
        data = data.data

        train_size = int(sizes[0] * len(data))
        train_val_size = int((sizes[0] + sizes[1]) * len(data))

        train = data[:train_size]
        val = data[train_size:]
        logger.info(f'train size : {len(train)}, val size : {len(val)}')

        return BatchMolDataset_motif(train), BatchMolDataset_motif(val)
    else:
        raise NotImplementedError("Do not support %s splits" % split_type)
    
def pre_load_data(dataset: BatchMolDataset_motif, rank: int, num_replicas: int, sample_per_file: int = None, epoch: int = 0, logger=None):
    mock_sampler = DistributedSampler(dataset=dataset, num_replicas=num_replicas, rank=rank, shuffle=False,
                                      sample_per_file=sample_per_file)
    mock_sampler.set_epoch(epoch)
    pre_indices = mock_sampler.get_indices()
    num = 0
    for i in pre_indices:
        dataset.load_data(i)
        num += 1
    
    logger.info(f'rank : {rank}, total {num} data pre-loading')
    
def create_logger(name, args, quiet = False):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # Set logger depending on desired verbosity
    ch = logging.StreamHandler()
    if quiet:
        ch.setLevel(logging.INFO)
    else:
        ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)

    if args.master_worker:
        if args.output_path is not None:
            fh_v = logging.FileHandler(os.path.join(args.output_path, 'verbose.log'))
            fh_v.setLevel(logging.DEBUG)
            fh_q = logging.FileHandler(os.path.join(args.output_path, 'quiet.log'))
            fh_q.setLevel(logging.INFO)

            logger.addHandler(fh_v)
            logger.addHandler(fh_q)
    else : 
        time.sleep(2)
        fh_v = logging.FileHandler(os.path.join(args.output_path, 'verbose.log'))
        fh_v.setLevel(logging.DEBUG)
        fh_q = logging.FileHandler(os.path.join(args.output_path, 'quiet.log'))
        fh_q.setLevel(logging.INFO)

        logger.addHandler(fh_v)
        logger.addHandler(fh_q)
    return logger