#!/bin/bash

p=0.6
k=1

first_channel=32
second_channel=64

num_epochs=200
optimizer="SGD"
lr=0.1
momentum=0.0
weight_decay=0.000
decay_schedule="cosine"

#our method parameters
ws_lr1=0.001
ws_lr2=0.0001


rho=0.0
lasso_lr=0.000

#beta lasso parameters
beta=0.0
lmbda=0.0
prox_layer="FC12"

# model_name="Net_2CNN"
model_name="Net_2FC"

checkpoint="True"
# checkpoint="False"

wandb_mode="online"
# wandb_mode="disabled"
# wandb_mode="offline"


#resource:
# cluster="debug"
# cluster="ComputeCanada"
cluster="Mila"

# partition="long"
partition="main"
#partition="unkillable"

# slurm_time="3-00:00:00"
slurm_time="2-00:00:00"
# slurm_time="1-00:00:00"
# slurm_time="0-10:00:00"
# slurm_time="0-01:00:00"

# slurm_time="0-03:00:00"
# partition="short-unkillable"

# cpus_per_task=24
cpus_per_task=4

entity="motahareh-s"
project="mnist-torus"


args="--config.data.p $p
--config.data.k $k
--config.train.epochs $num_epochs
--config.model.name $model_name
--config.model.kwargs.first_out_channel $first_channel
--config.model.kwargs.second_out_channel $second_channel
--config.optim.name $optimizer
--config.optim.lr $lr
--config.optim.kwargs.momentum $momentum
--config.optim.kwargs.weight_decay $weight_decay
--config.optim.decay_schedule $decay_schedule
--config.optim.regularization.ws_lr "${ws_lr1},${ws_lr2}"
--config.optim.regularization.lasso_lr $lasso_lr
--config.optim.regularization.rho $rho
--config.optim.regularization.prox_layer $prox_layer
--config.optim.regularization.beta $beta
--config.optim.regularization.lmbda $lmbda
--config.train.checkpoint $checkpoint
--config.verbose.verbose True
--config.verbose.log_freq 1
--config.wandb_entity $entity
--config.wandb_project $project
--config.wandb_mode $wandb_mode
--config.resource.cluster $cluster
--config.resource.partition $partition
--config.resource.time $slurm_time
--config.resource.cpus_per_task $cpus_per_task "


python ./src/main.py --config=./configs/mnist_14.py $args
