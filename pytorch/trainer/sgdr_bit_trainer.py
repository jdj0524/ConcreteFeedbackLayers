from .sgdr_trainer import SGDR_Trainer
import wandb

class SGDR_Bit_Trainer(SGDR_Trainer):
    def __init__(self, model, epochs, loss, optimizer_cls, gpu, metrics, options) -> None:
        super().__init__(model, epochs, loss, optimizer_cls, gpu, metrics, options)
        self.temp_schedule = options['temp_schedule']
        self.temp_anneal_factor = (self.temp_schedule['final_temp'] / self.temp_schedule['start_temp']) ** (1/epochs)
    def get_current_temp(self, epoch):
        temperature = self.temp_schedule['start_temp'] * self.temp_anneal_factor ** epoch
        return temperature
    
    def train_step(self, epoch):
        self.model.set_temperature(self.get_current_temp(epoch))
        super().train_step(epoch)
        
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
        wandb_dictionary['Current Temperature'] = float(self.get_current_temp(epoch))
        wandb.log(wandb_dictionary)
        print(print_string)