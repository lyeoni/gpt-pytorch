import time
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from radam import RAdam

from model import GPT, GPTLMHead, GPTClsHead

def timeit(method):
    def timed(*args, **kw):
        _args = args[0].args
            
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        if _args.distributed:
            if _args.local_rank == 0:
                print('Function Time: {}\t>\t{:.0f} min {:.0f} sec'.format(method.__name__, (te-ts)//60, (te-ts)%60))
        else:
            print('Function Time: {}\t>\t{:.0f} min {:.0f} sec'.format(method.__name__, (te-ts)//60, (te-ts)%60))
        
        return result
    return timed

class Trainer:
    def __init__(self, args, train_loader, test_loader, tokenizer):
        self.args = args
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        self.pad_id = tokenizer.pad_token_id
        self.eos_id = tokenizer.eos_token_id
        self.device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu', args.local_rank)
        self.writer = SummaryWriter() if args.local_rank in [-1, 0] else None
        self.n_gpus = torch.distributed.get_world_size() if args.distributed else torch.cuda.device_count()
        assert args.pretrain != args.finetune # Do not set both finetune and pretrain arguments to the same (True, False)

        if args.pretrained_model:
            self.gpt = torch.load(args.pretrained_model)
        else:
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
            with open(args.cached_label_dict, 'r') as file:
                label_dict = json.load(file)
            self.model = GPTClsHead(self.gpt, n_class=len(label_dict), cls_token_id=self.eos_id)
            self.model.to(self.device)
        
        if args.distributed:
            self.model = DistributedDataParallel(self.model, device_ids=[args.local_rank], output_device=args.local_rank)

        self.optimizer = RAdam(self.model.parameters(), args.lr)
        self.criterion = nn.CrossEntropyLoss(ignore_index = self.pad_id).to(self.device)
        self.cls_criterion = nn.CrossEntropyLoss().to(self.device)
    
    @timeit
    def train(self, epoch):    
        if self.args.pretrain:
            self.pretrain(epoch)
        if self.args.finetune:
            self.finetune(epoch)

    def pretrain(self, epoch):
        losses = 0
        n_batches, n_samples = len(self.train_loader), len(self.train_loader.dataset)
        
        self.model.train()
        for i, batch in enumerate(self.train_loader):
            inputs = batch[0].to(self.device)
            targets = inputs[:, 1:].contiguous()
            # |inputs| : (batch_size, seq_len), |targets| : (batch_size, seq_len-1)
            
            lm_logits = self.model(inputs)
            lm_logits = lm_logits[:, :-1].contiguous()
            # |lm_logits| : (batch_size, seq_len-1, vocab_size)
            
            loss = self.criterion(lm_logits.view(-1, self.vocab_size), targets.view(-1))
            losses += loss.item()
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.args.local_rank in [-1, 0]:
                self.writer.add_scalar('Loss/pre-train', loss.item(), ((epoch-1)*n_batches)+i)
                if i % (n_batches//5) == 0 and i != 0:
                    print('Iteration {} ({}/{})\tLoss: {:.4f}'.format(i, i, n_batches, losses/i))
        
        print('Train Epoch {} [rank: {}]\t>\tLoss: {:.4f}'.format(epoch, self.args.local_rank, losses/n_batches))

    def finetune(self, epoch):
        losses, accs = 0, 0
        n_batches, n_samples = len(self.train_loader), len(self.train_loader.dataset) # n_batches = batch size per GPU
        
        self.model.train()
        for i, batch in enumerate(self.train_loader):
            inputs, labels = map(lambda x: x.to(self.device), batch)
            # |inputs| : (batch_size, seq_len), |labels| : (batch_size)
            
            lm_logits, cls_logits = self.model(inputs)
            lm_logits = lm_logits[:, :-1].contiguous()
            # |lm_logits| : (batch_size, seq_len-1, vocab_size), |cls_logits| : (batch_size, n_class)
            
            lm_loss = self.criterion(lm_logits.view(-1, self.vocab_size), inputs[:, 1:].contiguous().view(-1))
            cls_loss = self.cls_criterion(cls_logits, labels)
            loss = cls_loss + (self.args.auxiliary_ratio * lm_loss)

            losses += loss.item()
            acc = (cls_logits.argmax(dim=-1) == labels).to(dtype=cls_logits.dtype).mean()
            accs += acc
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.args.local_rank in [-1, 0]:
                self.writer.add_scalar('Loss/fine-tune', loss.item(), ((epoch-1)*n_batches)+i)
                self.writer.add_scalar('Accuracy/fine-tune', acc, ((epoch-1)*n_batches)+i)
                if i % (n_batches//5) == 0 and i != 0:
                    print('Iteration {} ({}/{})\tLoss: {:.4f} Acc: {:.1f}%'.format(i, i, n_batches, losses/i, accs/i*100.))

        print('Train Epoch {} [rank: {}]\t>\tLoss: {:.4f} / Acc: {:.1f}%'.format(epoch, self.args.local_rank, losses/n_batches, accs/n_batches*100.))

    def evaluate(self, epoch):
        losses, accs = 0, 0
        n_batches, n_samples = len(self.test_loader), len(self.test_loader.dataset)

        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(self.test_loader):
                if self.args.pretrain:
                    inputs = batch.to(self.device)
                    targets = inputs[:, 1:].contiguous()
                    
                    lm_logits = self.model(inputs)
                    lm_logits = lm_logits[:, :-1].contiguous()
                    
                    loss = self.criterion(lm_logits.view(-1, self.vocab_size), targets.view(-1))
                    losses += loss.item()
                    
                    if self.args.local_rank in [-1, 0]:
                        self.writer.add_scalar('Loss/pre-train(eval)', loss.item(), ((epoch-1)*n_batches)+i)

                elif self.args.finetune:
                    inputs, labels = map(lambda x: x.to(self.device), batch)

                    lm_logits, cls_logits = self.model(inputs)
                    lm_logits = lm_logits[:, :-1].contiguous()

                    lm_loss = self.criterion(lm_logits.view(-1, self.vocab_size), inputs[:, 1:].contiguous().view(-1))
                    cls_loss = self.cls_criterion(cls_logits, labels)
                    loss = cls_loss + (self.args.auxiliary_ratio * lm_loss)

                    losses += loss.item()
                    acc = (cls_logits.argmax(dim=-1) == labels).to(dtype=cls_logits.dtype).mean()
                    accs += acc

                    if self.args.local_rank in [-1, 0]:
                        self.writer.add_scalar('Loss/fine-tune(eval)', loss.item(), ((epoch-1)*n_batches)+i)
                        self.writer.add_scalar('Accuracy/fine-tune(eval)', acc, ((epoch-1)*n_batches)+i)

        print('Eval Epoch {} [rank: {}]\t>\tLoss: {:.4f} / Acc: {:.1f}%'.format(epoch, self.args.local_rank, losses/n_batches, accs/n_batches*100.))

    def save(self, epoch, model_prefix='model', root='.model'):
        path = Path(root) / (model_prefix + '.ep%d' % epoch)
        if not path.parent.exists():
            path.parent.mkdir()

        if self.args.distributed:
            if self.args.local_rank == 0:
                torch.save(self.gpt, path)
        else:
            torch.save(self.gpt, path)
