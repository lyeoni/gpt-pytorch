from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from radam import RAdam
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from model import GPT, GPTLMHead, GPTClsHead

class Trainer:
    def __init__(self, args, data_loader, tokenizer):
        self.args = args
        self.data_loader = data_loader
        self.vocab_size = tokenizer.vocab_size
        self.pad_id = tokenizer.pad_token_id
        self.eos_id = tokenizer.eos_token_id
        self.device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu', args.local_rank)
        self.check_args()

        self.gpt = GPT(vocab_size=self.vocab_size,
                       seq_len=args.max_seq_len,
                       d_model=args.hidden,
                       n_layers=args.n_layers,
                       n_heads=args.n_attn_heads,
                       d_ff=args.ffn_hidden,
                       embd_pdrop=args.embd_dropout,
                       attn_pdrop=args.attn_dropout,
                       resid_pdrop=args.resid_dropout,
                       pad_id=self.pad_id)

        if args.pretrain:
            self.model = GPTLMHead(self.gpt)
            self.model.to(self.device)
        if args.finetune:
            self.model = GPTClsHead(self.gpt, n_class=2, cls_token_id=self.eos_id)
            self.model.to(self.device)
        
        if args.distributed:
            torch.cuda.set_device(args.local_rank)
            dist.init_process_group(backend='nccl')
            if args.pretrain:
                self.model = DistributedDataParallel(self.model, device_ids=[args.local_rank], output_device=args.local_rank)

        self.optimizer = RAdam(self.model.parameters(), args.lr)
        # self.optimizer = optim.Adam(self.model.parameters(), args.lr)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_id).to(self.device)
    
    def train(self, epoch):    
        if self.args.pretrain:
            self.pretrain(epoch)
        if self.args.finetune:
            self.finetune(epoch)

    def pretrain(self, epoch):
        losses = 0
        n_batches, n_samples = len(self.data_loader), len(self.data_loader.dataset)
        
        self.model.train()
        for i, batch in enumerate(self.data_loader):
            inputs = batch.to(self.device)
            targets = inputs[:, 1:].contiguous()
            # |inputs| : (batch_size, seq_len), |targets| : (batch_size, seq_len-1)
            
            lm_logits = self.model(inputs)
            lm_logits = lm_logits[:, :-1].contiguous()
            # |lm_logitis| : (batch_size, seq_len-1, vocab_size)
            
            loss = self.criterion(lm_logits.view(-1, self.vocab_size), targets.view(-1))
            losses += loss.item()
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if i % (n_batches//5) == 0 and i != 0:
                print('Iteration {} ({}/{})\tLoss: {:.4f}'.format(i, i, n_batches, losses/i))
        
        print('Train Epoch: {}\t>\tLoss: {:.4f}'.format(epoch, losses/n_batches))

    def finetune(self, epoch):
        losses, accs = 0, 0
        n_batches, n_samples = len(self.data_loader), len(self.data_loader.dataset)

        self.model.train()
        for i, batch in enumerate(self.data_loader):
            inputs, labels = map(lambda x: x.to(self.device), batch)
            
            cls_logits = self.model(inputs)
            # |cls_logits| : (batch_size, n_class)
            
            loss = self.criterion(cls_logits, labels)
            losses += loss.item()
            acc = (outputs.argmax(dim=-1) == labels).sum()
            accs += acc.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if i % (n_batches//5) == 0 and i != 0:
                print('Iteration {} ({}/{})\tLoss: {:.4f} Acc: {:4f}%'.format(
                    i, i, n_batches, losses/i, accs/(i*self.args.batch_size)*100.))

        print('Train Epoch: {}\t>\tLoss: {:.4f} / Acc: {:.1f}%'.format(epoch, losses/n_batches, accs/n_samples*100.))
    
    def save(self, epoch, model_prefix='model', root='.model'):
        path = Path(root) / (model_prefix + '.ep%d' % epoch)
        if not path.parent.exists():
            path.parent.mkdir()
        
        torch.save(self.gpt, path)
    
    def check_args(self):
        if (self.args.finetune and self.args.pretrain) or (not self.args.finetune and not self.args.pretrain):
            print("*****************************************\n" \
                  "Please avoid to set both finetune and pretrain arguments to True or False\n" \
                  "*****************************************")
            raise