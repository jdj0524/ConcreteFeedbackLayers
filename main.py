"""
Usage:
    main.py --config=config_name [--gpu=<gpu>]
    
Options:
    --gpu=<gpu>     Choice of GPU [default:0]
"""

from pytorch.configs.experiment_config import *
from docopt import docopt
import torch

torch.manual_seed(42)
if __name__ == '__main__':
    args = docopt(__doc__)
    configs = eval(args['--config'])()
    for config in configs:
        launcher_cls, model, trainer, data, options = config
        options['trainer_options']['gpu'] = int(args['--gpu'])
        launcher = launcher_cls(options, data, model, trainer)
        launcher.run()
