import typing, jax, flax, optax
from tqdm import tqdm
import numpy as np


from functools import partial
from utils import compute_macro_f1, compute_accuracy

from flax.training import train_state as flax_train_state, checkpoints

from jax.nn.initializers import normal as normal_init
from jax import numpy as jnp

class TrainState(flax_train_state.TrainState):
  batch_stats: typing.Any

class MLP(flax.linen.Module):

  units: typing.Sequence[ int ]

  def setup(self):
    self.layers = [flax.linen.Dense(neurons, kernel_init=normal_init(0.02), bias_init=normal_init(0.01)) for neurons in self.units]
    self.batch_norm = [flax.linen.BatchNorm(axis=-1, scale_init=normal_init(0.02), dtype=jnp.float32) for i in range(len(self.units) - 1)]

  def __call__(self,  x:jnp.ndarray, train: bool = True) -> jnp.ndarray:

    y_hat = x
    for i, layer in enumerate(self.layers):

      if i < len(self.layers) - 1:

        y_hat = layer(y_hat)
        y_hat = self.batch_norm[i](y_hat, use_running_average=not train)
        y_hat = flax.linen.leaky_relu(y_hat)

      else: y_hat = layer(y_hat)
    return y_hat


def create_state(rng, model_cls, opt, input_shape, learning_rate, momentum, decay=None): 

  """Create the training state given a model class. """ 

  model = model_cls([512, 512, 10])   

  if opt == 'sgd':
    tx = optax.sgd(learning_rate=optax.exponential_decay(init_value=learning_rate, decay_rate=0.5, transition_steps=20), momentum=momentum) 
  elif opt == 'adam':
    tx = optax.adam(learning_rate=learning_rate)

  variables = model.init(rng, jnp.ones(input_shape))   

  state = TrainState.create(apply_fn=model.apply, tx=tx, 
      params=variables['params'], batch_stats=variables['batch_stats'])
  
  return state

@jax.jit
def train_step(model_state, data_batch):

  def loss_fn(params, data):

    logits, mutables = model_state.apply_fn( {'params': params, 
      'batch_stats': model_state.batch_stats},
      data, mutable=['batch_stats'])

    labels = jax.nn.one_hot(data_batch['labels'], 10)
    loss = optax.softmax_cross_entropy(logits, labels).mean()
    return loss, {'logits':logits, 'mutables':mutables}


  (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(model_state.params, data_batch['data'])
  new_model_state = model_state.apply_gradients(grads=grads, batch_stats=aux['mutables']['batch_stats'])
  
  return new_model_state, loss, aux['logits']


@jax.jit
def eval_step(model_sate, data_batch):

  logits = model_sate.apply_fn( {'params': model_sate.params, 
        'batch_stats': model_sate.batch_stats},
        data_batch['data'], train=False, mutable=False)

  return flax.linen.softmax(logits)


def eval_dev_data(devloader, model_state):

  eval_preds = None
  eval_labels = None

  for batch, batch_data in enumerate(devloader):

    if eval_preds is None:
      eval_preds = eval_step(model_state, {'data':batch_data[0].numpy()})
      eval_labels = batch_data[1].numpy()
    else:
      eval_preds = jnp.concatenate([eval_preds, eval_step(model_state, {'data':batch_data[0].numpy()})])
      eval_labels = np.concatenate([eval_labels, batch_data[1].numpy()])

  dev_f1 = compute_macro_f1(logits=eval_preds,  labels=eval_labels)
  dev_error = 1. - compute_accuracy(logits=eval_preds, labels=eval_labels)

  eval_labels = jax.nn.one_hot(eval_labels, 10)
  dev_loss = optax.softmax_cross_entropy(eval_preds, eval_labels).mean().item()
  

  return dev_loss, dev_f1, dev_error

def restore_checkpoint(state, workdir): 
  return checkpoints.restore_checkpoint(workdir, state) 

def save_model(model_state, save_path, step = 0):
   checkpoints.save_checkpoint(ckpt_dir=save_path, target=model_state, overwrite=True, step=step)

def train_model(model_state, epoches, batch_size, trainloader, devloader):

  eloss = []
  edev_loss = []
  eerror = []
  edev_error = []

  best_error = 1e30

  for epoch in range(epoches):

    itr = tqdm(enumerate(trainloader))
    itr.set_description(f'Epoch: {epoch}')

    running_loss = None
    running_f1 = None
    running_error = None

    for batch, batch_data in itr:

      model_state, loss, logits = train_step(model_state,
                          {'data':batch_data[0].numpy(), 'labels':batch_data[1].numpy()})

      if running_loss is None:
        running_loss = loss.item()
        running_f1 = compute_macro_f1(logits=logits, labels=batch_data[1].numpy())
        running_error = 1. - compute_accuracy(logits=logits, labels=batch_data[1].numpy())
      else:
        running_loss = (running_loss + loss.item())/2.
        running_f1  = (running_f1 + compute_macro_f1(logits=logits, labels=batch_data[1].numpy()))/2.
        running_error = (running_error + 1. - compute_accuracy(logits=logits, labels=batch_data[1].numpy()))/2.

      if batch == len(trainloader) - 1:
        l, f1, err = eval_dev_data(devloader, model_state)
        running_loss, running_f1, running_error = eval_dev_data(trainloader, model_state)
        itr.set_postfix_str(f'loss: {running_loss:.2f} f1: {running_f1:.2f} errorX100: {running_error*100:.2f} dev_loss: {l:.2f} dev_f1: {f1:.2f} dev_errorX100: {err*100:.2f}' )
        eloss += [running_loss]
        edev_loss += [l]
        eerror += [running_error]
        edev_error += [err]

        if err < best_error:
          best_error = err
          save_model(model_state, save_path='model')

      else:
        itr.set_postfix_str(f'loss: {running_loss:.2f} f1: {running_f1:.2f}')
  
  return {'loss': eloss, 'dev_loss': edev_loss, 'error': eerror, 'dev_error': edev_error}
    


