# Motif-based Graph Self-Supervised Learning for Molecular Property Prediction with GROVER
this model based on Official Pytorch implementation of NeurIPS'21 paper "Motif-based Graph Self-Supervised Learning for Molecular Property Prediction"
(https://arxiv.org/abs/2110.00987). 
## Requirements
```
pytorch                   1.12.0             
torch-geometric           2.3.1
rdkit                     2022.9.2
tqdm                      
sklearn
wandb
```
I recommand that RDKit version must same, because different version make different clique

* `pretrain/` contains codes for motif-based graph self-supervised pretraining.
* `finetune/` contains codes for finetuning on MoleculeNet benchmarks for evaluation.

## changed
1. I changed molecular feature of atom/bond to GROVER's diverse feature.
   * GROVER Model : https://github.com/tencent-ailab/grover  
2. Apply Torch DDP Multi-gpu in pretrain(but don't apply multi-node)
3. Add multi-class classfication and regression on finetune

## Training
You can pretrain the model here
```
cd motif_based_pretrain
python pretrain_grover.py
```
if you can use multi_gpu, then use below
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node 2 pretrain_multi.py --grover_dataset --multi --[arguments]

## Finetune
You can finetune here
```
cd finetune
python finetune_grover.py --grover --[arguments]
```

## Cite
If you find this repo to be useful, please cite below MGSSL paper. Thank you.
```
@article{zhang2021motif,
  title={Motif-based Graph Self-Supervised Learning for Molecular Property Prediction},
  author={Zhang, Zaixi and Liu, Qi and Wang, Hao and Lu, Chengqiang and Lee, Chee-Kong},
  journal={arXiv preprint arXiv:2110.00987},
  year={2021}
}
```
