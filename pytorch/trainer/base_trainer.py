from .trainer import Trainer
import torch
import numpy as np
import wandb
from copy import deepcopy
import time

class BaseTrainer(Trainer):
    def __init__(self, model, epochs, loss, optimizer_cls, gpu, metrics, options) -> None:
        super().__init__(model, epochs, loss, optimizer_cls, gpu, metrics, options)
        self.best_loss = None
        self.best_model = None
    def train_step(self, epoch):
        step_losses = []
        for data in self.loaders['train']:
            self.model.train()
            data = data.to(self.gpu)
            output = self.model(data)
            step_loss = self.loss(output, data).mean()
            step_losses.append(step_loss.detach().cpu().float().numpy())
            self.optimizer.zero_grad()
            step_loss.backward()
            self.optimizer.step()
        self.eval_metric_history['train_loss'].append(np.mean(step_losses))
    
    def save_best_model(self, cur_loss):
        if self.best_loss is None or self.best_loss > cur_loss:
            self.best_loss = cur_loss
            self.best_model = deepcopy(self.model)
            torch.save(self.best_model.state_dict(), self.options['save_dir'] + self.best_model.get_save_name() + '.ckpt')

    def train(self):
        self.model = self.model.to(device=self.gpu)
        for i in range(self.epochs):
            self.train_step(i)
            self.eval()
            self.save_best_model(self.eval_metric_history['val_loss'][-1])
            self.print_eval_metrics(i)
        self.test()
    
    def eval(self):
        eval_metrics = {}
        for key in self.metrics.keys():
            eval_metrics[key] = []
        eval_metrics['val_loss'] = []

        for data in self.loaders['val']:
            self.model.eval()
            data = data.to(self.gpu)
            output = self.model(data).detach()
            for key in self.metrics.keys():
                eval_metrics[key].append(self.metrics[key](output, data).mean().cpu().float().numpy())
            eval_metrics['val_loss'] = self.loss(output, data).mean().cpu().float().numpy()

        for key in eval_metrics.keys():
            eval_metrics[key] = np.mean(eval_metrics[key])
            self.eval_metric_history[key].append(eval_metrics[key])
        
        
    
    def test(self):
        test_metrics = {}
        for key in self.metrics.keys():
            test_metrics[key] = []
        test_metrics['loss'] = []
        test_times = []
        for data in self.loaders['test']:
            self.best_model.eval()
            data = data.to(self.gpu)
            start = time.time()
            output = self.best_model(data)
            end = time.time()
            output = output.detach()
            test_times.append(end-start)
            for key in self.metrics.keys():
                test_metrics[key].append(self.metrics[key](output, data).mean().cpu().float().numpy())
            test_metrics['loss'] = self.loss(output, data).mean().cpu().float().numpy()
        
        wandb.run.summary['batch_inference_time'] = np.mean(test_times)
        for key in test_metrics.keys():
            wandb.run.summary['test_'+key] = np.mean(test_metrics[key])