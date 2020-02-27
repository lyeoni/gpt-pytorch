import argparse
from torch.utils.data import DataLoader, DistributedSampler

from data_utils import create_examples, pretrain_collate_fn
from tokenization import PretrainedTokenizer
from trainer import Trainer

def main(args):
    print(args)
    
    # Load tokenizer
    tokenizer = PretrainedTokenizer(pretrained_model = args.pretrained_sp_model, vocab_file = args.vocab_file)
    
    # Build DataLoader
    dataset = create_examples(args, tokenizer)
    if args.distributed:
        distributed_sampler = DistributedSampler(dataset)
        lm_loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=pretrain_collate_fn, num_workers=args.n_workers, sampler=distributed_sampler)
    else:
        lm_loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=pretrain_collate_fn, num_workers=args.n_workers, shuffle=True)

    # Build Trainer for unsupervised pre-training
    trainer = Trainer(args, lm_loader, tokenizer)

    # Train
    for epoch in range(1, args.epochs+1):
        trainer.train(epoch)
        trainer.save(epoch, args.output_model_prefix)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus',                 required=True, type=str, help='corpus name to pre-train')
    parser.add_argument('--vocab_file',             default='wiki103.vocab',     type=str, help='vocabulary path')
    parser.add_argument('--pretrained_sp_model',    default='wiki103.model',     type=str, help='pretrained sentencepiece model path')
    parser.add_argument('--output_model_prefix',    default='model',             type=str, help='output model name prefix')
    parser.add_argument('--pretrain',               action='store_true')
    parser.add_argument('--finetune',               action='store_true')
    # Input parameters
    parser.add_argument('--batch_size',     default=64,    type=int,   help='batch size')
    parser.add_argument('--max_seq_len',    default=384,   type=int,   help='the maximum size of the input sequence')
    parser.add_argument('--n_workers',      default=4,     type=int,   help='the number of workers')
    # Train parameters
    parser.add_argument('--epochs',         default=100,       type=int,   help='the number of epochs')
    parser.add_argument('--lr',             default=1.5e-4,    type=float, help='initial learning rate')
    parser.add_argument('--local_rank',     default=0,         type=int,   help='node rank for distributed training')
    parser.add_argument('--no_cuda',        action='store_true')
    parser.add_argument('--distributed',    action='store_true')
    # Model parameters
    parser.add_argument('--hidden',         default=768,  type=int,   help='the number of expected features in the transformer decoder')
    parser.add_argument('--n_layers',       default=12,   type=int,   help='the number of decoder layers')
    parser.add_argument('--n_attn_heads',   default=12,   type=int,   help='the number of multi-head attention heads')
    parser.add_argument('--embd_dropout',   default=0.1,  type=float, help='the embedding dropout value')
    parser.add_argument('--resid_dropout',  default=0.1,  type=float, help='the residual dropout value')
    parser.add_argument('--attn_dropout',   default=0.1,  type=float, help='the attention dropout value')
    parser.add_argument('--ffn_hidden',     default=3072, type=int,   help='the dimension of the feedforward network')

    args = parser.parse_args()

    main(args)