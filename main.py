import jax, argparse, sys
from pathlib import Path

from utils import parameters, load_dataset, plot_training
from models import create_state, train_model, MLP#, load_model



def check_params(args=None):
  
  parser = argparse.ArgumentParser(description='MLP for MNIST')

  parser.add_argument('-model', metavar='model', default = parameters.model, 
      help='Model to be run')  
      
  parser.add_argument('-mode', metavar='model', default = 'train', 
      help='Process to be run')

  parser.add_argument('-lr', metavar='lrate', type = float, default=parameters.lr, 
      help='Learning rate')

  parser.add_argument('-m', metavar='momentum', type=float, default=parameters.momentum,
      help='Momentum Optimizer')
  
  parser.add_argument('-bs', metavar='batch_zise', type=int, default=parameters.batch_size,
      help='Batch Size')

  parser.add_argument('-e', metavar='epoch', type = int, default = parameters.epoches,
      help='Epoch')

  parser.add_argument('-o', metavar='output', default='out',
      help='directory to save outputs')
  
  parser.add_argument('-opt', metavar='opt', default= parameters.opt,
    help='optimizer to use')  

  parser.add_argument('-decay', metavar='decay', type=float, default=parameters.decay,
    help='optimizer to use')

  return parser.parse_args(args)


if __name__ == '__main__':

  params = check_params(sys.argv[1:])

  model = params.model
  lr = params.lr
  mommentum = params.m
  batch_size = params.bs
  epoches = params.e
  output = params.o
  opt = params.opt
  decay = params.decay
  mode = params.mode


  Path(output).mkdir(parents=True, exist_ok=True)

  trainloader, devloader = load_dataset(batch_size)
  key = jax.random.PRNGKey(seed=parameters.seed)

  if model == 'mlp':

    model_state = create_state(key, MLP, opt, next(iter(trainloader))[0].numpy().shape, lr, mommentum,epoches=epoches, decay=decay)
    
    if mode == 'train':
      history = train_model(model_state=model_state, epoches=epoches, batch_size=batch_size,
                trainloader=trainloader, devloader=devloader)
      plot_training(history, output)
    else:
      pass#model_state = load_model(model_state, output)



  