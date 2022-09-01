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
# Now, let's define some aspects of our system (based on CBxs_lobV_top_30nm at ESRF beamline i16a):

mul_factor = 1e6 # 1000 # conversion factor

lambda_ =  7.25146e-5 # wavelength in microns for 17kEV (~3.62573e-05 um for 33kEV)
lambda_ratios = 1.0 # ratio of wavelengths
# du_eff = 0.03 # effective pixel size of sample on detector (microns)
# du = 3e-6 * mul_factor # (microns) documented pixelsize_detector in ESRF script
z_total = 1.208 * mul_factor # distance from focus point to detector (microns)
zs_f2s = jnp.array([0.012080, 0.012598, 0.014671, 0.018975]) * mul_factor # distances source to sample (microns) 
dz = 1. # depth of slice in microns
zs_s2d = z_total - zs_f2s - dz # distances sample to detector (microns) 
n = 1 - 1e-9 # refractive index of x-rays is slightly below unity according to ESRF
D = 30e-3 # aperture width in microns (corresponding to *smallest beam width at focal point post Kirkpatrick-Baez mirrors)

# And pick a propagation calculation method (transfer, transform, or exact)
prop_fcn = exact_propagate
# prop_fcn = transform_propagate
# prop_fcn = transfer_propagate

# %% [markdown]
# Then get and display Siemens star test object (then make it into a phase object) and use it as ground truth (gt):
# values chosen based on Osmium using the following online calculator: https://henke.lbl.gov/optical_constants/getdb2.html
# @ 17.8 keV
d = 1.16845949e-05 # 1 minus the refractive index corresponding to osmium
b = 1.14571696e-06 # the extinction coefficient
# # @ 30 keV
# d = 4.13493717e-06 # 1 minus the refractive index corresponding to osmium
# b = 1.68511178e-07 # the extinction coefficient
delta, beta = siemens_star(n, d, b, True)

#Let's crop this sample for convenience
# mask = np.zeros_like(data)
# mid = n_pixels // 2
# pad = mid // 2
# mask[:, mid - pad: mid + pad, mid - pad: mid + pad, :] = 1
# data *= mask

n_pixels = delta.shape[1]
Nf = jnp.max((D / 2) ** 2 / (lambda_ * z_total)) # Fresnel number
Q = n_pixels / (4 * Nf)
N = int(jnp.ceil((Q * n_pixels) / 2) * 2)
# N_pad = max(0, int(N - n_pixels))
N_pad = int(jnp.ceil((n_pixels - (4 * Nf)) / 2) * 2)
assert (N_pad + n_pixels) > (4 * Nf)

print(f'N_pad = {N_pad}')
plt.imshow(delta[0,...,0].squeeze())

print(f'Delta range = [{delta.min()}, {delta.max()}]')
print(f'Beta range = [{beta.min()}, {beta.max()}]')

# %% [markdown]
# Now let's make the lightfields hitting the sample and display their intensity and phase:

src_field = create_field(pointsource(dz, D, n), (n_pixels, n_pixels), D / n_pixels, lambda_, lambda_ratios)
in_fields = prop_fcn(field=src_field, z=zs_f2s.reshape((len(zs_f2s),1,1,1)), n=n, N_pad=N_pad)
show_fields(in_fields)

# %% [markdown]
# Now let's make some projections:

def phasechange(field: LightField, delta: jnp.ndarray, beta: jnp.ndarray, dz: float) -> LightField:
    k = 2 * jnp.pi / field.wavelengths
    u = field.u * jnp.exp(-1j * k * dz * (delta - 1j * beta))
    return field.replace(u=u)

out_fields = phasechange(in_fields, delta, beta, dz=dz)
show_fields(out_fields)

# %% [markdown]
# Let's add free space propagation after the sample:

sensor_fields = prop_fcn(field=out_fields, z=zs_s2d.reshape((len(zs_s2d),1,1,1)), n=n, N_pad=N_pad)
show_fields(sensor_fields)

# %% [markdown]
# Finally, let's collect everything as ground truths for different projection distances and the empty beam image:

gt_sample_fields = sensor_fields
gt_sample_images = sensor_fields.intensity # these would be the sensor readings for each distance
gt_empty_beam = prop_fcn(field=src_field, z=z_total, n=n, N_pad=N_pad)
gt_empty_beam_image = gt_empty_beam.intensity
gt_params = {'delta': delta, 'beta': beta, 'input_field': src_field}

show_fields(sensor_fields, gt_empty_beam)

# %% [markdown]
# ### Now let's see if we can reconstruct the sample from the sensor images
# Instantiate system and extract states vs. parameters. (Parameters are differentiable/trainable, states are not). Here we see the trainable parameters for the phase object: ```delta```, ```beta```, ```z1s``` (distance from beam focus to sample) and ```input_field```:

multi = MultiDistance()
key = random.PRNGKey(42)
variables = multi.init(key)
state, params = variables.pop('params') # Split state and params to optimize for
del variables # Delete variables to avoid wasting resources
init_params = params

flatten_dict(unfreeze(params)).keys()

# %% [markdown]
# Okay. Let's take a look at our naive outputs before training:
show_results(multi, params, gt_params)

# %% [markdown]
# Well, that's not right. So let's define loss and step functions to train with and see where we stand before any training:
naive_loss, naive_grads = jax.value_and_grad(loss)(params, multi, sample=gt_sample_images, empty=gt_empty_beam_image)

print(f'Loss: {naive_loss}')
print(f"Gradients - max Delta: {naive_grads['delta'].max()}, max Beta: {naive_grads['beta'].max()}")
show_params(naive_grads)

# %% [markdown]
# Now let's give training a shot, using an Adam ~~AdaBelief~~ optimizer:
init_lr = 1e-6
num_epochs = 10

# initialize optimizer
# optimizer = optax.sgd(init_lr)
optimizer = optax.adam(init_lr)
# optimizer = optax.adamw(init_lr)
# optimizer = optax.adabelief(init_lr)
opt_state = optimizer.init(params)
step = get_step_fn(loss, optimizer, model=multi, sample=gt_sample_images)

for epoch in range(num_epochs):
  start_time = time.time()
  params, opt_state, current_loss, grads = step(params, opt_state)
  epoch_time = time.time() - start_time
  
  print("Epoch {} in {:0.2f} sec, loss = {:0.3f}".format(epoch, epoch_time, current_loss))

# %% [markdown]
# Any improvement?
# What do the gradients look like?
show_results(multi, params, gt_params, grads)

# %% [markdown]
# Now if we run it for a while:
num_epoch = 1000
num_stop = 100 # how many iterations with delta_loss == 0 to have before quitting

no_loss_cnt = 0
losses = [naive_loss]
progress = trange(num_epoch)
for epoch in progress:
  params, opt_state, current_loss, grads = step(params, opt_state)  
  delta_loss = current_loss - losses[-1]
  no_loss_cnt += (delta_loss == 0.)
  no_loss_cnt *= (delta_loss == 0.)
  if no_loss_cnt >= num_stop: break
  progress.set_postfix({'delta_loss': delta_loss, 'current_loss': current_loss, 'no_loss_cnt': no_loss_cnt})
  losses.append(current_loss)

plt.plot(losses)

# %% [markdown]
# Let's see how we did:
show_results(multi, params, gt_params, grads)

# %% [markdown]
# What happens if we sample random windows for calculating loss?
RW_size = 256
# rw_params = params
rw_params = init_params
naive_loss, naive_grads = jax.value_and_grad(random_window_loss)(rw_params, multi, key, sample=gt_sample_images)

print(f'Loss: {naive_loss}')
print(f"Gradients - max Delta: {naive_grads['delta'].max()}, max Beta: {naive_grads['beta'].max()}")
show_params(naive_grads)

# %%
# initialize optimizer and run
init_lr = 1e-8
# optimizer = optax.sgd(init_lr)
optimizer = optax.adam(init_lr)
# optimizer = optax.adamw(init_lr)
# optimizer = optax.adabelief(init_lr)
opt_state = optimizer.init(rw_params)

step = get_step_fn(random_window_loss, optimizer, model=multi, key=key, sample=gt_sample_images, size=RW_size)

num_epoch = 1000
num_stop = 100 # how many iterations with delta_loss == 0 to have before quitting

losses = jnp.zeros(num_epoch)
losses = losses.at[0].set(naive_loss)
no_loss_cnt = 0
progress = trange(num_epoch)
for epoch in progress:
    rw_params, opt_state, current_loss, grads, key = step(rw_params, opt_state, key=key)  
    delta_loss = current_loss - losses[epoch]
    no_loss_cnt += (delta_loss == 0.)
    no_loss_cnt *= (delta_loss == 0.)
    if no_loss_cnt >= num_stop: break
    progress.set_postfix({'delta_loss': delta_loss, 'current_loss': current_loss, 'no_loss_cnt': no_loss_cnt})
    losses = losses.at[epoch+1].set(current_loss)

plt.plot(losses)

# %%

#Show Window:
show_random_window_results(multi, rw_params, gt_params, key, grads, size=RW_size)


# %%
#Try LBFGS instead
import jaxopt as jop
loss_partial = partial(loss, model=multi, sample=gt_sample_images)    

# lbfgs = jop.LBFGS(loss_partial, jit=True)
# bfgs_params, state = lbfgs.run(init_params)
lbfgsb = jop.ScipyBoundedMinimize(fun=loss_partial, method="l-bfgs-b")
lb = params.unfreeze()
ub = params.unfreeze()
lb['delta'] = lb['delta'].at[:].set(0)
lb['beta'] = lb['beta'].at[:].set(0)
ub['delta'] = ub['delta'].at[:].set(0.001)
ub['beta'] = ub['beta'].at[:].set(0.001)
bfgs_params, info = lbfgsb.run(init_params, bounds=(lb, ub))

# %%
show_results(multi, bfgs_params, gt_params)
