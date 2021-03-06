# OpenAI GPT
[![LICENSE](https://img.shields.io/github/license/lyeoni/gpt-pytorch?style=flat-square)](https://github.com/lyeoni/gpt-pytorch/blob/master/LICENSE)
[![GitHub issues](https://img.shields.io/github/issues/lyeoni/gpt-pytorch?style=flat-square)](https://github.com/lyeoni/gpt-pytorch/issues)
[![GitHub stars](https://img.shields.io/github/stars/lyeoni/gpt-pytorch?style=flat-square&color=important)](https://github.com/lyeoni/gpt-pytorch/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/lyeoni/gpt-pytorch?style=flat-square&color=blueviolet)](https://github.com/lyeoni/gpt-pytorch/network/members)

PyTorch Implementation of OpenAI GPT

<p align="center"><img width= 70 src="https://pytorch.org/assets/images/logo-icon.svg"></p>

## Quick Start
### 0. Install dependencies
PreNLP is Preprocessing Library for Natural Language Processing. It provides sentencepiece tokenizer.
```
$ pip install prenlp
$ git clone https://github.com/LiyuanLucasLiu/RAdam
$ python RAdam/setup.py install
```
<br>

### 1. Setup input pipeline

#### Building vocab based on your corpus
```
$ python vocab.py --corpus <YOUR_CORPUS> --prefix <VOCAB_NAME> --vocab_size <YOUR_VOCAB_SIZE>
```

or you can download WikiText-103 corpus using below command, and build vocab based on this.
```
$ python -c "import prenlp; prenlp.data.WikiText103()"
$ ls .data/wikitext-103
wiki.test  wiki.train  wiki.valid
$ python vocab.py --corpus .data/wikitext-103/wiki.train --prefix wiki103
```
<br>

### 2. Unsupervised pre-training
```
$ python main.py --train_corpus <TRAIN_CORPUS> --vocab_file <VOCAB_FILE> --pretrained_sp_model <PRETRAINED_SP_MODEL> --pretrain
```

#### Distributed training with torch.distributed (Recommended)
You can apply to both single-node(multi-GPU) and multi-node distributed training.
```
$ python -m torch.distributed.launch --nproc_per_node=<NPROC_PER_NODE> --nnodes=<NNODES> --node_rank=<NODE_RANK> --master_addr=<MASTER_ADDR> --master_port=<MASTER_PORT> main.py --train_corpus <TRAIN_CORPUS> \
                                    --vocab_file <VOCAB_FILE> \
                                    --pretrained_sp_model <PRETRAINED_SP_MODEL> \
                                    --pretrain --distributed
```
<br>

### 3. Supervised fine-tuning
```
$ python main.py --train_corpus <TRAIN_CORPUS> --test_corpus <TEST_CORPUS>  --vocab_file <VOCAB_FILE> --pretrained_sp_model <PRETRAINED_SP_MODEL> --pretrained_model <PRETRAINED_MODEL> --finetune --do_eval
```

#### Distributed training with torch.distributed (Recommended)
You can apply to both single-node(multi-GPU) and multi-node distributed training.
```
$ python -m torch.distributed.launch --nproc_per_node=<NPROC_PER_NODE> --nnodes=<NNODES> --node_rank=<NODE_RANK> --master_addr=<MASTER_ADDR> --master_port=<MASTER_PORT> main.py --train_corpus <TRAIN_CORPUS> --test_corpus <TEST_CORPUS> \
                                    --vocab_file <VOCAB_FILE> \
                                    --pretrained_sp_model <PRETRAINED_SP_MODEL> \
                                    --pretrained_model <PRETRAINED_MODEL> \
                                    --finetune --do_eval --distributed
```
<br>

## Questions and Discussions
### Does auxiliary objective function have a bigger impact?
GPT authors mentioned that "We additionally found that including language modeling as an auxiliary objective to the fine-tuninghelped learning by (a) improving generalization of the supervised model, and (b) accelerating convergence".

And, in our experiments on IMDb dataset, it shows that the auxiliary objective function improves test-accuracy as shown below.
The orange line is for _auxiliary weight = 0_, blue line is for _auxiliary weight = 0.25_, red line is for _auxiliary weight = 0.5_. And you can also see training logs for this in [here](https://github.com/lyeoni/gpt-pytorch/tree/master/logs).
<p align="center"><img width= 700 src="logs/tensorboard-visualization.png"></p>
<br>

## List of options
You may need to change below argument parameters.
```
$ python main.py -h
usage: main.py [-h] --train_corpus TRAIN_CORPUS --vocab_file VOCAB_FILE
               --pretrained_sp_model PRETRAINED_SP_MODEL [--pretrain]
               [--finetune] [--do_eval] [--test_corpus TEST_CORPUS]
               [--pretrained_model PRETRAINED_MODEL]
               [--output_model_prefix OUTPUT_MODEL_PREFIX]
               [--batch_size BATCH_SIZE] [--max_seq_len MAX_SEQ_LEN]
               [--n_workers N_WORKERS] [--epochs EPOCHS] [--lr LR]
               [--auxiliary_ratio AUXILIARY_RATIO] [--local_rank LOCAL_RANK]
               [--no_cuda] [--distributed] [--hidden HIDDEN]
               [--n_layers N_LAYERS] [--n_attn_heads N_ATTN_HEADS]
               [--embd_dropout EMBD_DROPOUT] [--resid_dropout RESID_DROPOUT]
               [--attn_dropout ATTN_DROPOUT] [--ffn_hidden FFN_HIDDEN]
               [--cached_label_dict CACHED_LABEL_DICT]

optional arguments:
  -h, --help            show this help message and exit
  --train_corpus TRAIN_CORPUS
                        corpus for either pre-train or fine-tune
  --vocab_file VOCAB_FILE
                        pretrained vocabulary
  --pretrained_sp_model PRETRAINED_SP_MODEL
                        pretrained sentencepiece model
  --pretrain
  --finetune
  --do_eval
  --test_corpus TEST_CORPUS
                        corpus for either pre-train or fine-tune evaluation
  --pretrained_model PRETRAINED_MODEL
                        pretrained GPT model path
  --output_model_prefix OUTPUT_MODEL_PREFIX
                        output model name prefix
  --batch_size BATCH_SIZE
                        batch size
  --max_seq_len MAX_SEQ_LEN
                        the maximum size of the input sequence
  --n_workers N_WORKERS
                        the number of workers
  --epochs EPOCHS       the number of epochs
  --lr LR               initial learning rate
  --auxiliary_ratio AUXILIARY_RATIO
                        weight of auxiliary objective
  --local_rank LOCAL_RANK
                        node rank for distributed training
  --no_cuda
  --distributed
  --hidden HIDDEN       the number of expected features in the transformer
                        decoder
  --n_layers N_LAYERS   the number of decoder layers
  --n_attn_heads N_ATTN_HEADS
                        the number of multi-head attention heads
  --embd_dropout EMBD_DROPOUT
                        embedding dropout value
  --resid_dropout RESID_DROPOUT
                        residual dropout value
  --attn_dropout ATTN_DROPOUT
                        attention dropout value
  --ffn_hidden FFN_HIDDEN
                        dimension of the feedforward network
  --cached_label_dict CACHED_LABEL_DICT
```

### References
- [Improving Language Understandingby Generative Pre-Training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)
- [openai / finetune-transformer-lm](https://github.com/openai/finetune-transformer-lm)