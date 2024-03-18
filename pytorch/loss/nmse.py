import torch
from torch.linalg import matrix_norm
from einops import rearrange

def MSE_loss(output, target):
    mse = torch.mean((output - target) ** 2)
    # mse = torch.sum((output - target) ** 2, dim = (1,2))
    
    return mse

def NMSE_loss(output, target):
    channel_dim = output.shape[-1] // 2
    output = output[:,:,:,:,0] + 1j * output[:,:,:,:,1]
    target = target[:,:,:,:,0] + 1j * target[:,:,:,:,1]
    
    mse = (torch.abs(output - target)**2).sum(dim = (1,2))
    
    power = torch.sum(torch.abs(target) ** 2, dim = (1,2))
    nmse = mse / power
    return nmse.mean(dim = 1)

def Cosine_distance(output, target):
    channel_dim = output.shape[-1] // 2
    output = output[:,:,:,:,0] + 1j * output[:,:,:,:,1]
    target = target[:,:,:,:,0] + 1j * target[:,:,:,:,1]
    
    inner_product = torch.abs(torch.sum(torch.conj(output) * target, dim = (1,2)))
    target_norm = torch.abs(torch.sqrt((target * torch.conj(target)).sum(dim = (1,2))))
    output_norm = torch.abs(torch.sqrt((output * torch.conj(output)).sum(dim = (1,2))))
    cosine_dist = inner_product / (target_norm * output_norm)
    cosine_dist = cosine_dist.mean(dim = 1)
    
    return cosine_dist