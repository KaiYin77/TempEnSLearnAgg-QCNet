from tqdm import tqdm
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch

from datasets import ArgoverseV2Dataset
from datamodules import ArgoverseV2DataModule
from predictors import TempEnsLearnAgg 
from transforms import TargetBuilder

if __name__ == '__main__':
    pl.seed_everything(2024, workers=True)

    parser = ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--train_batch_size', type=int, required=True)
    parser.add_argument('--val_batch_size', type=int, required=True)
    parser.add_argument('--test_batch_size', type=int, required=True)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--persistent_workers', type=bool, default=True)
    parser.add_argument('--accelerator', type=str, default='auto')
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--train_raw_dir', type=str, default=None)
    parser.add_argument('--val_raw_dir', type=str, default=None)
    parser.add_argument('--max_epochs', type=int, default=64)
    args = parser.parse_args()

    model = {
        'TempEnsLearnAgg': TempEnsLearnAgg,
    }[args.model]()
    datamodule = {
        'argoverse_v2': ArgoverseV2DataModule,
    }[args.dataset](**vars(args))
    model_checkpoint = ModelCheckpoint(monitor='val_minFDE', save_top_k=3, mode='min')
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    trainer = pl.Trainer(accelerator=args.accelerator, devices=args.devices,
                         strategy=DDPStrategy(find_unused_parameters=False, gradient_as_bucket_view=True),
                         callbacks=[model_checkpoint, lr_monitor], max_epochs=args.max_epochs, 
                         accumulate_grad_batches=8) # make sure devices * batch_size = 32 or accumulate_grad_batches * batch_size = 32

    # Freeze the base model parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Make sure the temporal aggregate layer parameters are trainable
    for param in model.temporal_aggregate_layer.parameters():
        param.requires_grad = True

    trainer.fit(model, datamodule)
