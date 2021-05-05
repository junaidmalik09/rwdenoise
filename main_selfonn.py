# common
import os
from PIL import Image
from argparse import ArgumentParser

# torch
import torch
from torch import nn

# utils
from utils.datasets import SIDDDataset,SIDDValidDataset
from utils.models import get_model
from utils.metrics import BatchPSNR
from utils.lit_module import LitDenoiser,WandbLogger,ModelCheckpoint,Trainer,LearningRateMonitor
    
# main function
def main():
	# Injection of arguments
    parser = ArgumentParser()
    parser = LitDenoiser.add_model_specific_args(parser)

    # training args
    parser.add_argument('--max_epochs',type=int,default=100)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--gpu_id', type=int, default=None)
    parser.add_argument('--overfit_batches', type=float, default=0)
    parser.add_argument('--fast_dev_run', type=bool, default=False)
    parser.add_argument('--checkpointing',type=bool,default=False)
    parser.add_argument('--resume_from_checkpoint', type=str, default=None)
    parser.add_argument('--load_from_checkpoint',type=str,default=None)

    # wandb args
    parser.add_argument('--log_to_wandb',type=bool,default=False)
    parser.add_argument('--name',type=str,default=None)
    parser.add_argument('--project',type=str,default='debugging')
    parser.add_argument('--version',type=str,default=None)

    # data args
    parser.add_argument('--data_path',type=str,default='C:/Users/PA_hm17901/')
    
    args = parser.parse_args()

    # Init
    wandb_logger = WandbLogger(
        name=args.name,
        project=args.project,
        log_model=False
    )
    
    callbacks = []
    checkpoint_callback = ModelCheckpoint(
        monitor='val_psnr',
        save_top_k=1,
        mode='max',
        verbose=False
    )
    
    lr_callback = LearningRateMonitor(logging_interval='epoch')
    
    if args.checkpointing: callbacks.append(checkpoint_callback)
    callbacks.append(lr_callback)
    
    

    trainer = Trainer(
        max_epochs=args.max_epochs, # maximum number of epochs
        gpus=args.gpus if args.gpu_id is None else str(args.gpu_id), # gpus
        overfit_batches=args.overfit_batches, # for debugging
        fast_dev_run=args.fast_dev_run, # for debugging
        logger=wandb_logger if args.log_to_wandb else True, # logger
        checkpoint_callback=args.checkpointing, # if checkpointing is enabled
        callbacks=callbacks, # checkpoint callback to use
        distributed_backend='dp' if args.gpu_id is None else None, # callback
        resume_from_checkpoint=args.resume_from_checkpoint, # resume checkpoing
    )

    if args.load_from_checkpoint is not None:
        print("Loading from checkpoint..")
        denoiser = LitDenoiser.load_from_checkpoint(args.load_from_checkpoint)
        denoiser.hparams.args = args
    else:
        denoiser = LitDenoiser(args)
    
    # Start training
    trainer.fit(denoiser)
    
if __name__=='__main__':
    main()
