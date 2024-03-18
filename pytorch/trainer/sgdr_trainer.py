from .base_trainer import BaseTrainer
import torch
import numpy as np

class SGDR_Trainer(BaseTrainer):
    def __init__(self, model, epochs, loss, optimizer_cls, gpu, metrics, options) -> None:
        super().__init__(model, epochs, loss, optimizer_cls, gpu, metrics, options)
        

    def train_step(self, epoch):
        step_losses = []
        for i, data in enumerate(self.loaders['train']):
            self.model.train()
            data = data.to(self.gpu)
            output = self.model(data)
            step_loss = self.loss(output, data).mean()
            step_losses.append(step_loss.detach().cpu().float())
            self.optimizer.zero_grad()
            step_loss.backward()
            self.optimizer.step()
            self.scheduler.step(epoch + i / self.iters)
        self.eval_metric_history['train_loss'].append(np.mean(step_losses))
    
    def build_optimizer(self):
        super().build_optimizer()
        self.scheduler = self.options['train_schedulers'](
            optimizer = self.optimizer, **self.options['train_scheduler_options']
            )
        
            