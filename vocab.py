"""Code from
    - https://github.com/lyeoni/nlp-tutorial/blob/master/translation-transformer/vocab.py
    - https://github.com/lyeoni/nlp-tutorial/blob/master/text-classification-transformer/vocab.py
"""
import argparse
from collections import Counter, OrderedDict

from prenlp.tokenizer import SentencePiece

def build(args):
    tokenizer = SentencePiece.train(input = args.corpus, model_prefix = args.prefix,
                                    vocab_size = args.vocab_size,
                                    model_type = args.model_type,
                                    character_coverage = args.character_coverage,
                                    max_sentence_length = args.max_sentence_length,
                                    pad_token = args.pad_token,
                                    unk_token = args.unk_token,
                                    bos_token = args.bos_token,
                                    eos_token = args.eos_token)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--corpus',      required=True,           type=str, help='one-sentence-per-line corpus file')
    parser.add_argument('--prefix',      required=True,           type=str, help='output vocab(or sentencepiece model) name prefix')

    parser.add_argument('--vocab_size',          default=16000,   type=int, help='the maximum size of the vocabulary')
    parser.add_argument('--character_coverage',  default=1.0,     type=float,
                        help='amount of characters covered by the model, good defaults are: 0.9995 for languages with rich character set\
                             like Japanse or Chinese and 1.0 for other languages with small character set')
    parser.add_argument('--model_type',          default='bpe',   type=str, help='sentencepiece model type. Choose from unigram, bpe, char, or word')
    parser.add_argument('--max_sentence_length', default=100000,  type=int, help='The maximum input sequence length')
    parser.add_argument('--pad_token',           default='[PAD]', type=str, help='token that indicates padding')
    parser.add_argument('--unk_token',           default='[UNK]', type=str, help='token that indicates unknown word')
    parser.add_argument('--bos_token',           default='[BOS]', type=str, help='token that indicates beginning of sentence')
    parser.add_argument('--eos_token',           default='[EOS]', type=str, help='token that indicates end of sentence')

    args = parser.parse_args()

    build(args)