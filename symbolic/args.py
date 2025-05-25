import argparse
from math import inf


class BoolArg(argparse.Action):
    """
    Take an argparse argument that is either a boolean or a string and return a boolean.
    """
    def __init__(self, default=None, nargs=None, *args, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")

        # Set default
        if default is None:
            raise ValueError("Default must be set!")

        default = _arg_to_bool(default)

        super().__init__(*args, default=default, nargs='?', **kwargs)

    def __call__(self, parser, namespace, argstring, option_string):

        if argstring is not None:
            # If called with an argument, convert to bool
            argval = _arg_to_bool(argstring)
        else:
            # BoolArg will invert default option
            argval = True

        setattr(namespace, self.dest, argval)

def _arg_to_bool(arg):
    # Convert argument to boolean

    if type(arg) is bool:
        # If argument is bool, just return it
        return arg

    elif type(arg) is str:
        # If string, convert to true/false
        arg = arg.lower()
        if arg in ['true', 't', '1']:
            return True
        elif arg in ['false', 'f', '0']:
            return False
        else:
            return ValueError('Could not parse a True/False boolean')
    else:
        raise ValueError('Input must be boolean or string! {}'.format(type(arg)))


#### Argument parser ####

def init_argparse():
    """
    Sets up the argparse object for the sudoku dataset
    """
    parser = argparse.ArgumentParser()
    
    # Optimizer options
    parser.add_argument('--num-epoch', type=int, default=255, metavar='N',
                        help='number of epochs to train (default: 511)')
    parser.add_argument('--batch-size', '-bs', type=int, default=25, metavar='N',
                        help='Mini-batch size (default: 25)')
    parser.add_argument('--alpha', type=float, default=0.9, metavar='N',
                        help='Value of alpha to use for exponential moving average of training loss. (default: 0.9)')

    parser.add_argument('--weight-decay', type=float, default=0, metavar='N',
                        help='Set the weight decay used in optimizer (default: 0)')
    parser.add_argument('--cutoff-decay', type=float, default=0, metavar='N',
                        help='Set the weight decay used in optimizer for learnable radial cutoffs (default: 0)')
    parser.add_argument('--lr-init', type=float, default=1e-3, metavar='N',
                        help='Initial learning rate (default: 1e-3)')
    parser.add_argument('--lr-final', type=float, default=1e-5, metavar='N',
                        help='Final (held) learning rate (default: 1e-5)')
    parser.add_argument('--lr-decay', type=int, default=inf, metavar='N',
                        help='Timescale over which to decay the learning rate (default: inf)')
    parser.add_argument('--lr-decay-type', type=str, default='cos', metavar='str',
                        help='Type of learning rate decay. (cos | linear | exponential | pow | restart) (default: cos)')
    parser.add_argument('--lr-minibatch', '--lr-mb', action=BoolArg, default=True,
                        help='Decay learning rate every minibatch instead of epoch.')
    parser.add_argument('--sgd-restart', type=int, default=-1, metavar='int',
                        help='Restart SGD optimizer every (lr_decay)^p epochs, where p=sgd_restart. (-1 to disable) (default: -1)')

    parser.add_argument('--optim', type=str, default='amsgrad', metavar='str',
                        help='Set optimizer. (SGD, AMSgrad, Adam, RMSprop)')

    # Dataloader and randomness options
    parser.add_argument('--shuffle', action=BoolArg, default=True,
                        help='Shuffle minibatches.')
    parser.add_argument('--seed', type=int, default=1, metavar='N',
                        help='Set random number seed. Set to -1 to set based upon clock.')

    ### Arguments for files to save things to
    # Job prefix is used to name checkpoint/best file
    parser.add_argument('--prefix', '--jobname', type=str, default='nosave',
                        help='Prefix to set load, save, and logfile. (default: nosave)')

    # Allow to manually specify file to load
    parser.add_argument('--loadfile', type=str, default='',
                        help='Set checkpoint file to load. Leave empty to auto-generate from prefix. (default: (empty))')
    # Filename to save model checkpoint to
    parser.add_argument('--checkfile', type=str, default='',
                        help='Set checkpoint file to save checkpoints to. Leave empty to auto-generate from prefix. (default: (empty))')
    # Filename to best model checkpoint to
    parser.add_argument('--bestfile', type=str, default='',
                        help='Set checkpoint file to best model to. Leave empty to auto-generate from prefix. (default: (empty))')
    # Filename to save logging information to
    parser.add_argument('--logfile', type=str, default='',
                        help='Duplicate logging.info output to logfile. Set to empty string to generate from prefix. (default: (empty))')
    # Filename to save predictions to
    parser.add_argument('--predictfile', type=str, default='',
                        help='Save predictions to file. Set to empty string to generate from prefix. (default: (empty))')

    # Working directory to place all files
    parser.add_argument('--workdir', type=str, default='./',
                        help='Working directory as a default location for all files. (default: ./)')
    # Directory to place logging information
    parser.add_argument('--logdir', type=str, default='log/',
                        help='Directory to place log and savefiles. (default: log/)')
    # Directory to place saved models
    parser.add_argument('--modeldir', type=str, default='model/',
                        help='Directory to place log and savefiles. (default: model/)')
    # Directory to place model predictions
    parser.add_argument('--predictdir', type=str, default='predict/',
                        help='Directory to place log and savefiles. (default: predict/)')
    
    # Dataset options
    parser.add_argument('--task', type=str, default='sudoku')
    parser.add_argument('--dataset', type=str, default='big_kaggle',
                        help='Directory to look up data from. (default: data/big_kaggle/)')
    parser.add_argument('--size', type=int, default=10)
    parser.add_argument('--grid_size', type=int, default=None)
    parser.add_argument('--train_ratio', type=float, default=0.7, metavar='N',
                        help='Number of samples to train on. Set to -1 to use entire dataset. (default: -1)')
    parser.add_argument('--valid_ratio', type=float, default=0.2, metavar='N',
                        help='Number of validation samples to use. Set to -1 to use entire dataset. (default: -1)')
    parser.add_argument('--test_ratio', type=float, default=0.1, metavar='N',
                        help='Number of test samples to use. Set to -1 to use entire dataset. (default: -1)')
    parser.add_argument('--resize', type=bool, default=False)

    # Computation options
    parser.add_argument('--cuda', dest='cuda', action='store_true',
                        help='Use CUDA (default)')
    parser.set_defaults(cuda=True)

    parser.add_argument('--float', dest='dtype', action='store_const', const='float',
                        help='Use floats.')
    parser.add_argument('--double', dest='dtype', action='store_const', const='double',
                        help='Use doubles.')
    parser.set_defaults(dtype='float')

    parser.add_argument('--num_workers', type=int, default=2,
                        help='Set number of workers in dataloader. (Default: 8)')

    # Model options
    parser.add_argument('--dims', type=int, default=64,
                        help='Set the dim of UNet model. (Default: 64)')
    parser.add_argument('--timestep', type=int, default=20,
                        help='Set the timesteps of Diffusion model. (Default: 20)')
    parser.add_argument('--cond', type=bool, default=False)
    
    
    args = parser.parse_args()
    return args
