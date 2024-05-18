import torch
from ..lr_scheduler.sgdr import CosineAnnealingWarmUpRestarts
from ..launcher.base_launcher import BaseLauncher
from ..models.bit_transformer import ChannelTransformerBitLevelLargeVariable
from ..trainer.sgdr_bit_variable_trainer import SGDR_Bit_Variable_Trainer
from ..dataloader.dataloader import DeepMIMOSampleDataset
from ..loss.nmse import MSE_loss, NMSE_loss, Cosine_distance


def channeltransformer_sgdr_bit_variable():
    launcher = BaseLauncher
    model = ChannelTransformerBitLevelLargeVariable
    trainer = SGDR_Bit_Variable_Trainer
    data = DeepMIMOSampleDataset
    options = {
        'wandb_project_name': 'BitChannelTransformer',
        'save_dir': '/home/automatic/projects/ConcreteFeedbackLayers/checkpoints/O1_140/',
        'batch_size': 256,
        'data_options': {
            'files_dir':'/home/automatic/DeepMIMO_Datasets/O1_140_8b84b4/'
        },
        'data_split': {
            'train': 0.6, 
            'val': 0.2, 
            'test': 0.2
        },
        'model_options': {
            'n_blocks':3, 'd_model':512, 'nhead':8, 'dim_feedforward':2048, 
            'n_tx':64, 'n_rx':16, 'n_carrier':128, 'dim_feedback':256
        },
        'temp_schedule': {
            'start_temp': 10,
            'final_temp': 0.01
        },
        'trainer_options': {
            'max_bits' : 256,
            'eval_bits' : 128,
            'epochs' : 500, 
            'loss' : MSE_loss,
             'optimizer_cls' : torch.optim.AdamW,
             'gpu' : 0, 
             'metrics' : {
                 'cosine' : (Cosine_distance, 'max'),
                 'NMSE' : (NMSE_loss, 'min'),
             },
        },
        'optimizer_options': {
            'lr' : 1e-5,
            
        },
        'train_schedulers': CosineAnnealingWarmUpRestarts,
        'train_scheduler_options': 
            {
                'T_0' : 500,
                'T_mult' : 2,
                'T_up': 5,
                'eta_max': 0.001,
                'last_epoch':-1,
                'gamma': 0.8
            },
        
    }
    return [(launcher, model, trainer, data, options)]

def channeltransformer_sgdr_bit_variable_blockage():
    launcher = BaseLauncher
    model = ChannelTransformerBitLevelLargeVariable
    trainer = SGDR_Bit_Variable_Trainer
    data = DeepMIMOSampleDataset
    options = {
        'wandb_project_name': 'BitChannelTransformer',
        'save_dir': '/home/automatic/projects/ConcreteFeedbackLayers/checkpoints/O1_28B/',
        'batch_size': 256,
        'data_options': {
            'files_dir':'/home/automatic/DeepMIMO_Datasets/O1_28B/'
        },
        'data_split': {
            'train': 0.6, 
            'val': 0.2, 
            'test': 0.2
        },
        'model_options': {
            'n_blocks':3, 'd_model':512, 'nhead':8, 'dim_feedforward':2048, 
            'n_tx':64, 'n_rx':16, 'n_carrier':128, 'dim_feedback':512
        },
        'temp_schedule': {
            'start_temp': 10,
            'final_temp': 0.01
        },
        'trainer_options': {
            'max_bits' : 512,
            'eval_bits' : 128,
            'epochs' : 2000, 
            'loss' : MSE_loss,
             'optimizer_cls' : torch.optim.AdamW,
             'gpu' : 0, 
             'metrics' : {
                 'cosine' : (Cosine_distance, 'max'),
                 'NMSE' : (NMSE_loss, 'min'),
             },
        },
        'optimizer_options': {
            'lr' : 1e-5,
            
        },
        'train_schedulers': CosineAnnealingWarmUpRestarts,
        'train_scheduler_options': 
            {
                'T_0' : 2000,
                'T_mult' : 2,
                'T_up': 5,
                'eta_max': 0.001,
                'last_epoch':-1,
                'gamma': 0.8
            },
        
    }
    return [(launcher, model, trainer, data, options)]
