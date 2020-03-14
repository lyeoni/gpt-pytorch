from typing import Iterable, Union, List
from pathlib import Path
import json

import torch
import torch.distributed as dist
from torch.utils.data import TensorDataset

class PretrainInputExample:
    """A single example for unsupervised pre-training.
    """
    def __init__(self, text: str):
        self.text = text

class ClsInputExample:
    """A single example for supervised fine-tuning (classification).
    """
    def __init__(self, text: str, label: str):
        self.text = text
        self.label = label

class PretrainInputFeatures:
    """A single set of features of pre-training data.
    """
    def __init__(self, input_ids: List[int]):
        self.input_ids = input_ids

class ClsInputFeatures:
    """A single set of features of fine-tuning data (classification).
    """
    def __init__(self, input_ids: List[int], label_id: int):
        self.input_ids = input_ids
        self.label_id = label_id

def convert_examples_to_features(examples,
                                 tokenizer,
                                 args,
                                 mode):
    bos_token = tokenizer.bos_token
    eos_token = tokenizer.eos_token
    pad_token = tokenizer.pad_token

    # Build label dict(vocab) with examples
    if args.finetune:
        if mode == 'train':
            labels = sorted(list(set([example.label for example in examples])))
            label_dict = {label: i for i, label in enumerate(labels)}
            with open(args.cached_label_dict, 'w') as file:
                json.dump(label_dict, file,  indent=4)
        elif mode == 'test':
            with open(args.cached_label_dict, 'r') as file:
                label_dict = json.load(file)

    # Create features
    features = []
    for i, example in enumerate(examples):
        tokens = tokenizer.tokenize(example.text)
        tokens = [bos_token] + tokens[:args.max_seq_len-2] + [eos_token] # BOS, EOS
        tokens += [pad_token] * (args.max_seq_len - len(tokens))

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        if args.finetune:
            label_id = label_dict.get(example.label)

        if args.pretrain:
            feature = PretrainInputFeatures(input_ids)
        elif args.finetune:
            feature = ClsInputFeatures(input_ids, label_id)

        features.append(feature)

    return features

def create_examples(args, tokenizer, mode='train'):
    if args.local_rank not in [-1, 0]: # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        dist.barrier()

    # Load data features from cache or dataset file
    assert mode in ('train', 'test')
    cached_features_file = Path('cached_features_{}_{}_{}'.format('pretrain' if args.pretrain else 'finetune', mode, args.max_seq_len))

    if cached_features_file.exists():
        print('Loading features from cached file', cached_features_file)
        features = torch.load(cached_features_file)
    else:
        corpus_path = args.train_corpus if mode=='train' else args.test_corpus
        with open(corpus_path, 'r', encoding='utf-8') as reader:
            corpus = reader.readlines()
        
        # Create examples
        if args.pretrain:
            corpus = list(map(lambda x: x.strip(), corpus))
            corpus = list(filter(lambda x: len(x) > 0, corpus))
            examples = [PretrainInputExample(text) for text in corpus]
        elif args.finetune:
            corpus = list(map(lambda x: x.split('\t'), corpus))
            corpus = list(map(lambda x: list(map(lambda y: y.strip(), x)), corpus))
            corpus = list(map(lambda x: list(filter(lambda y: len(y) > 0, x)), corpus))
            examples = [ClsInputExample(text, label) for label, text in corpus]

        # Convert examples to features
        features = convert_examples_to_features(examples, tokenizer, args, mode)

        print('Saving features into cached file', cached_features_file)
        torch.save(features, cached_features_file)

    if args.local_rank == 0: # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        dist.barrier()
    
    # Create dataset with features
    if args.pretrain:
        all_input_ids = torch.tensor([feature.input_ids for feature in features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids)
    elif args.finetune:
        all_input_ids = torch.tensor([feature.input_ids for feature in features], dtype=torch.long)
        all_label_ids = torch.tensor([feature.label_id for feature in features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_label_ids)
    
    return dataset