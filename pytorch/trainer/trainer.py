import torch
import wandb
class Trainer:
    def __init__(self, model, epochs, loss, optimizer_cls, gpu, metrics, options) -> None:
        self.options = options
        self.epochs = epochs
        self.model = model
        self.loss = loss
        self.optimizer_cls = optimizer_cls
        self.gpu = gpu
        self.metrics = {}
        self.init_wandb(metrics)
        
        self.scheduler = None
        self.eval_metric_history = {'train_loss' : [], 'val_loss' : []}
        for key in self.metrics.keys():
            self.eval_metric_history[key] = []
    def init_wandb(self, metrics):
        wandb.define_metric('train_loss', summary = 'min')
        wandb.define_metric('val_loss', summary = 'min')
        for name in metrics.keys():
            wandb.define_metric(name, summary = metrics[name][1])
            self.metrics[name] = metrics[name][0]
        
    def build_optimizer(self):
            self.optimizer = self.optimizer_cls(self.model.parameters(), **self.options['optimizer_options'])

    def set_loaders(self, loaders):
        self.loaders = loaders
        self.iters = len(self.loaders['train'])
    
    def train_step(self, epoch):
        raise NotImplementedError
    
    def train(self):
        raise NotImplementedError
    
    def eval(self):
        raise NotImplementedError
    
    def test(self):
        raise NotImplementedError
    
    def print_eval_metrics(self, epoch):
        print_string = ''
        print_string += "epoch {}, ".format(epoch)
        wandb_dictionary = {}
        for key in self.eval_metric_history.keys():
            print_string += "{} : {}, ".format(key, self.eval_metric_history[key][-1])
            wandb_dictionary[key] = self.eval_metric_history[key][-1]
        if self.scheduler is not None:
            print_string += "current LR : {}".format(self.scheduler.get_last_lr())
            wandb_dictionary["Current LR"] = float(self.scheduler.get_last_lr()[0])
        wandb.log(wandb_dictionary)
        print(print_string)