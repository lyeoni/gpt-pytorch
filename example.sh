# choose the network interface to use for distributed training
# export NCCL_SOCKET_IFNAME=eth0

# pre-training (distributed training)
python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_addr="localhost" --master_port=1112 main.py --train_corpus .data/wikitext-103/wiki.train --vocab_file wiki103.vocab --pretrained_sp_model wiki103.model --pretrain --distributed

# fine-tuning (distributed training)
python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_addr="localhost" --master_port=1112 main.py --train_corpus .data/aclImdb/imdb.train --test_corpus .data/aclImdb/imdb.test --vocab_file wiki103.vocab --pretrained_sp_model wiki103.model --pretrained_model .model/model.ep22 --finetune --do_eval --distributed

# tensorboard
tensorboard --logdir=runs

