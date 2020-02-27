from typing import Iterable, Union, List

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class InputExample:
    """A single example for unsupervised pre-training.
    """
    def __init__(self, text: str):
        self.text = text

class InputFeatures:
    """A single set of features of pre-training data.
    """
    def __init__(self, input_ids: List[int]):
        self.input_ids = input_ids

class GPTPretrainDataset(Dataset):
    def __init__(self, input_ids):
        self.input_ids = input_ids

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return torch.tensor(self.input_ids[idx])

def pretrain_collate_fn(inputs):
    inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
    return inputs

def convert_examples_to_features(examples: List[InputExample],
                                 tokenizer,
                                 max_seq_len): #-> List[InputFeatures]:
    bos_token = tokenizer.bos_token
    eos_token = tokenizer.eos_token
    
    features = []
    for i, example in enumerate(examples):
        tokens = tokenizer.tokenize(example.text)
        tokens = [bos_token] + tokens[:max_seq_len-2] + [eos_token] # BOS, EOS
        
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        
        feature = InputFeatures(input_ids)
        features.append(feature)

    return features

def create_examples(args, tokenizer):
    with open(args.corpus, 'r', encoding='utf-8') as reader:
        corpus = reader.readlines()
    corpus = list(map(lambda x: x.strip(), corpus))
    corpus = list(filter(lambda x: len(x) > 0, corpus))
    
    examples = [InputExample(text) for text in corpus]
    
    features = convert_examples_to_features(examples, tokenizer, args.max_seq_len)
    
    all_input_ids = [feature.input_ids for feature in features]
    
    dataset = GPTPretrainDataset(all_input_ids)
    
    return dataset
    