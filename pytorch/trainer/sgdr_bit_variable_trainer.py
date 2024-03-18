from .sgdr_bit_trainer import SGDR_Bit_Trainer
import torch
import numpy as np
import wandb
import time

class SGDR_Bit_Variable_Trainer(SGDR_Bit_Trainer):
    def __init__(self, model, epochs, loss, optimizer_cls, gpu, metrics, options, max_bits, eval_bits) -> None:
        super().__init__(model, epochs, loss, optimizer_cls, gpu, metrics, options)
        self.max_bits = max_bits
        self.eval_bits = eval_bits
        
    def train_step(self, epoch):
        self.model.set_temperature(self.get_current_temp(epoch))
        step_losses = []
        for i, data in enumerate(self.loaders['train']):
            bits = torch.randint(low = 1, high = self.max_bits+1, size=(data.shape[0],))
            self.model.train()
            data = data.to(self.gpu)
            output = self.model(data, bits)
            step_loss = self.loss(output, data).mean()
            step_losses.append(step_loss.detach().cpu().float())
            self.optimizer.zero_grad()
            step_loss.backward()
            self.optimizer.step()
            self.scheduler.step(epoch + i / self.iters)
        self.eval_metric_history['train_loss'].append(np.mean(step_losses))
        
    def eval(self):
        eval_metrics = {}
        for key in self.metrics.keys():
            eval_metrics[key] = []
        eval_metrics['val_loss'] = []

        for data in self.loaders['val']:
            self.model.eval()
            data = data.to(self.gpu)
            output = self.model(data, [self.eval_bits for _ in range(data.shape[0])]).detach()
            for key in self.metrics.keys():
                eval_metrics[key].append(self.metrics[key](output, data).mean().cpu().float().numpy())
            eval_metrics['val_loss'] = self.loss(output, data).mean().cpu().float().numpy()

        for key in eval_metrics.keys():
            eval_metrics[key] = np.mean(eval_metrics[key])
            self.eval_metric_history[key].append(eval_metrics[key])
    
    def test(self):
        test_metrics = {}
        test_times = []
        for key in self.metrics.keys():
                test_metrics[key] = []
        test_metrics['loss'] = []    
        for i in range(self.max_bits):
            for key in self.metrics.keys():
                test_metrics[key].append([])
            test_metrics['loss'].append([])
            
            for data in self.loaders['test']:
                self.best_model.eval()
                data = data.to(self.gpu)
                start = time.time()
                output = self.best_model(data, [i+1 for _ in range(data.shape[0])])
                end = time.time()
                output = output.detach()
                test_times.append(end-start)
                for key in self.metrics.keys():
                    test_metrics[key][-1].append(self.metrics[key](output, data).mean().cpu().float().numpy())
                test_metrics['loss'][-1].append(self.loss(output, data).mean().cpu().float().numpy())
            
            for key in test_metrics.keys():
                test_metrics[key][-1] = np.mean(test_metrics[key][-1])
            
        wandb.run.summary['batch_inference_time'] = np.mean(test_times)
        for key in test_metrics.keys():
            wandb.run.summary['test_'+key] = test_metrics[key]