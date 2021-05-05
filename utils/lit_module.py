
import os
from PIL import Image
from argparse import ArgumentParser

# torch
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import random_split,DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau,CosineAnnealingLR,MultiplicativeLR

# lightning
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint,LearningRateMonitor

# utils
from .models import get_model
from .datasets import SIDDDataset,SIDDValidDataset
from .metrics import BatchPSNR,BatchSSIM,calc_batch_ssim

# wandb
import wandb



class LitDenoiser(pl.LightningModule):
    def __init__(self,args):
        super().__init__()
        self.model = get_model(args)
        self.psnr = BatchPSNR()
        self.save_hyperparameters()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--model',type=str,default='selfdncnn')
        parser.add_argument('--num_layers',type=int,default=17)
        parser.add_argument('--num_rrg', type=int, default=4//2)
        parser.add_argument('--num_dab', type=int, default=8//2)
        parser.add_argument('--nfeats', type=int, default=64)
        parser.add_argument('--batch_size', type=int,nargs=3,default=[4,4,16])
        parser.add_argument('--num_workers', type=int,default=4)
        parser.add_argument('--patch_size',type=int,default=80)
        parser.add_argument('--patches_per_image',type=int,default=512) # 512 => 160k training patches, 3200 => 1M training patches
        parser.add_argument('--patches_per_batch',type=int,default=32)
        parser.add_argument('--lr', type=float, default=0.001)
        parser.add_argument('--q', type=int, default=3)
        return parser

    def setup(self,stage):
        self.train_ds = SIDDDataset(
                path=self.hparams.args.data_path,
        		patches_per_batch = self.hparams.args.patches_per_batch,
        		patches_per_image = self.hparams.args.patches_per_image,
        		patch_size = self.hparams.args.patch_size
        	)
        train_len = round(len(self.train_ds)*0.9)
        val_len = len(self.train_ds)-train_len
        torch.manual_seed(27)
        self.train_ds,self.val_ds = random_split(self.train_ds,[train_len,val_len])
       	self.test_ds = SIDDValidDataset(path=self.hparams.args.data_path)
        print("Train:",train_len*self.hparams.args.patches_per_batch)
        print("Val:",val_len*self.hparams.args.patches_per_batch)
        print("Test:",len(self.test_ds))
        
        self.top_val = -1e9
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),lr=self.hparams.args.lr) #,momentum=0.9)
        lmbda = lambda epoch: 1.05
        scheduler = MultiplicativeLR(optimizer, lr_lambda=lmbda)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler
            #'monitor': 'train_loss'
        }
    
    def train_dataloader(self): 
        num_workers = self.hparams.args.num_workers if self.hparams.args.overfit_batches==0 else 0
        return DataLoader(self.train_ds, batch_size=self.hparams.args.batch_size[0],pin_memory=True,num_workers=num_workers,shuffle=True)
    
    def val_dataloader(self): 
        num_workers = self.hparams.args.num_workers if self.hparams.args.overfit_batches==0 else 0
        return DataLoader(self.val_ds, batch_size=self.hparams.args.batch_size[1],pin_memory=True,num_workers=num_workers)
    
    def test_dataloader(self): 
        num_workers = self.hparams.args.num_workers if self.hparams.args.overfit_batches==0 else 0
        return DataLoader(self.test_ds, batch_size=self.hparams.args.batch_size[2],pin_memory=True,num_workers=num_workers)
    
    def forward(self, x): return self.model(x)

    def training_step(self, batch, batch_idx):
        noisy, clean = batch
        _, _, c, h, w = noisy.size()
        noisy = noisy.view(-1,c,h,w)
        clean = clean.view(-1,c,h,w)
        cleaned = self.model(noisy)
        loss = F.mse_loss(cleaned, clean,reduction='sum')
        return loss

    def training_epoch_end(self,losses):
        loss_epoch = torch.stack([x['loss'] for x in losses]).mean()
        if hasattr(self.logger.experiment,'log'): self.logger.experiment.log({
            'loss':loss_epoch,
            'epoch':self.current_epoch,
            'global_step':self.global_step
        })
    
    def validation_step(self, batch, batch_idx):
        noisy, clean = batch
        _, _, c, h, w = noisy.size()
        noisy = noisy.view(-1,c,h,w)
        clean = clean.view(-1,c,h,w)
        cleaned = self.model(noisy)
        self.psnr.update(cleaned,clean)

        cleaned_img = cleaned[0,...].permute(1,2,0).detach().cpu().numpy()
        return cleaned_img    
            
 
    
    def validation_epoch_end(self, outs):
        # log epoch metric
        val_psnr = self.psnr.compute()
        self.top_val = max(val_psnr,self.top_val)
        self.log('val_psnr',val_psnr,prog_bar=True)
        if hasattr(self.logger.experiment,'log'): 
            self.logger.experiment.log({
                'val_psnr':val_psnr,
                'epoch':self.current_epoch,
                'global_step':self.global_step,
                'best_val_psnr':self.top_val,
                #'examples': [wandb.Image(outs[0], caption="Cleaned")]
            })
            
        self.psnr.reset()

    def test_step(self,batch,batch_idx):
        noisy, clean = batch
        if noisy.dim()>4:
            _, _, c, h, w = noisy.size()
            noisy = noisy.view(-1,c,h,w)
            clean = clean.view(-1,c,h,w)

        cleaned = self.model(noisy)
        self.psnr.update(cleaned,clean)
        return calc_batch_ssim(cleaned,clean)
    
    def test_epoch_end(self,outs):
        test_psnr = self.psnr.compute()
        self.psnr.reset()
        correct = 0
        total = 0
        for (c,t) in outs: correct+=c; total+=t;
        print(test_psnr,correct/total)