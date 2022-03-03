# %% [markdown]
# # Notebook for developing diffrax/JAX based reconstruction package for XNH

# %% [markdown]
# First, imports:
#SET GPU TO USE:
from functools import partial
import os
from queue import Empty
os.environ['CUDA_VISIBLE_DEVICES'] = '1,3'
import time
from tqdm import trange

from diffrax import *
import diffrax as dx
from diffrax.functional import *
from diffrax.elements.sources import *
from diffrax import LightField

# from jax.config import config
# config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import grad, jit, vmap, random, lax
import jax

import optax
import numpy as np
from einops import rearrange

from flax.traverse_util import flatten_dict
from flax.core import unfreeze
import flax.linen as nn

import matplotlib.pyplot as plt

from plots import *
from data import *
from models import *
from steps import *
from losses import *

# %% [markdown]
# Let's try working with some real data:
prefix='CBxs_lobV_bottomp100um_30nm'
path='./example_ESRF_data'

ESRF_data = get_ESRF_angle(path, prefix)

# %%
gt_dark_images = ESRF_data['dark_images']
gt_images_ESRF = ESRF_data['images'] - gt_dark_images
gt_empty_ESRF = jnp.expand_dims(jnp.median(ESRF_data['empty_images'] - gt_dark_images, 0), 0) # Let's just take the median for now

show_images(gt_images_ESRF, gt_empty_ESRF)

# %%
# First let's try to estimate the empty beam
empty = EmptyBeam(field_shape=gt_empty_ESRF.shape[1], prop_fcn=exact_propagate)
key = random.PRNGKey(42)
variables = empty.init(key)
state, empty_params = variables.pop('params') # Split state and params to optimize for
del variables # Delete variables to avoid wasting resources
init_empty_params = empty_params

flatten_dict(unfreeze(empty_params)).keys()
show_fields(empty.input_field(init_empty_params['u']))

# %%
naive_loss, naive_grads = jax.value_and_grad(loss)(init_empty_params, model=empty, empty=gt_empty_ESRF)

print(f'Loss: {naive_loss}')
show_fields(empty.input_field(naive_grads['u']))

# %% [markdown]
# Now let's give training a shot, using an Adam ~~AdaBelief~~ optimizer:
init_lr = 1e-6
num_epochs = 10

# initialize optimizer
# optimizer = optax.sgd(init_lr)
optimizer = optax.adam(init_lr)
# optimizer = optax.adamw(init_lr)
# optimizer = optax.adabelief(init_lr)
opt_state = optimizer.init(empty_params)
step = get_step_fn(loss, optimizer, model=empty, empty=gt_empty_ESRF)
# step = get_step_fn(random_window_loss, optimizer, model=empty, key=key, empty=gt_empty_ESRF)

for epoch in range(num_epochs):
  start_time = time.time()
  empty_params, opt_state, current_loss, grads = step(empty_params, opt_state)
  epoch_time = time.time() - start_time
  
  print("Epoch {} in {:0.2e} sec, loss = {:0.2e}".format(epoch, epoch_time, current_loss))

# %% [markdown]
# Any improvement?
# What do the loss and gradients look like?
sim_empty = empty.apply({'params': empty_params}).pop('empty')
show_images(sim_empty.intensity, gt_empty_ESRF)
show_loss((sim_empty.intensity, gt_empty_ESRF))
show_fields(empty.input_field(grads['u']), empty.input_field(empty_params['u']))

# %% [markdown]
# Now if we run it for a while:
num_epoch = 5000
num_stop = 100 # how many iterations with delta_loss == 0 to have before quitting

no_loss_cnt = 0
losses = [naive_loss]
progress = trange(num_epoch)
for epoch in progress:
  empty_params, opt_state, current_loss, grads = step(empty_params, opt_state)  
  delta_loss = current_loss - losses[-1]
  no_loss_cnt += (delta_loss == 0.)
  no_loss_cnt *= (delta_loss == 0.)
  if no_loss_cnt >= num_stop: break
  progress.set_postfix({'delta_loss': delta_loss, 'current_loss': current_loss, 'no_loss_cnt': no_loss_cnt})
  losses.append(current_loss)

plt.plot(losses)

# %% # Any better?
sim_empty = empty.apply({'params': empty_params}).pop('empty')
show_images(sim_empty.intensity, gt_empty_ESRF)
show_loss((sim_empty.intensity, gt_empty_ESRF))
show_fields(empty.input_field(grads['u']), empty.input_field(empty_params['u']))


# %%
#Try LBFGS instead
import jaxopt as jop
loss_partial = partial(loss, model=empty, empty=gt_empty_ESRF)    

# lbfgs = jop.LBFGS(loss_partial, jit=True)
# bfgs_params, state = lbfgs.run(init_empty_params)
lbfgsb = jop.ScipyBoundedMinimize(fun=loss_partial, method="l-bfgs-b")
lb = empty_params.unfreeze()
ub = empty_params.unfreeze()
lb['u'] = lb['u'].at[:].set(complex(0, 0))
ub['u'] = ub['u'].at[:].set(complex(13, 13))
bfgs_params, info = lbfgsb.run(init_empty_params, bounds=(lb, ub))

# %% # Any better?
sim_empty = empty.apply({'params': bfgs_params}).pop('empty')
show_images(sim_empty.intensity, gt_empty_ESRF)
show_loss((sim_empty.intensity, gt_empty_ESRF))
show_fields(empty.input_field(bfgs_params['u']))
# %%
