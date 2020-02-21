# OpenAI GPT
PyTorch Implementation of OpenAI GPT

<p align="center"><img width= 70 src="https://pytorch.org/assets/images/logo-icon.svg"></p>


## Quick Start
### 0. Install PreNLP library
PreNLP is Preprocessing Library for Natural Language Processing. It provides sentencepiece tokenizer.
```
$ pip install prenlp
```

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

### 2. Unsupervised pre-training


### 3. Supervised fine-tuning


### References
- [Improving Language Understandingby Generative Pre-Training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)
- [openai / finetune-transformer-lm](https://github.com/openai/finetune-transformer-lm)