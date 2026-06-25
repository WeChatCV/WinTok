#!/usr/bin

export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=bond1
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_HCA=mlx5_bond_1:1,mlx5_bond_2:1,mlx5_bond_3:1,mlx5_bond_4:1,mlx5_bond_5:1,mlx5_bond_6:1,mlx5_bond_7:1,mlx5_bond_8:1
export NCCL_IB_GID_INDEX=3
export NCCL_IB_SL=3
export NCCL_SOCKET_NTHREADS=4
export NCCL_NSOCKS_PERTHREAD=4
export TORCH_NCCL_BLOCKING_WAIT=1

export PYTHONPATH=$(pwd):$PYTHONPATH

# imagenet
torchrun --standalone --nproc_per_node=8 examples/eval_tokenizer/eval_metrics_ddp.py \
    --ckpt_path path_to_ckpt \
    --imagenet_val path_to_ImageNet-1k/val/

# mscoco
torchrun --standalone --nproc_per_node=8 examples/eval_tokenizer/eval_metrics_ddp.py \
    --ckpt_path path_to_ckpt \
    --mscoco path_to_MSCOCO2017/val2017