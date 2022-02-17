# Notebook for developing diffrax/JAX based reconstruction package for XNH

# %%
import diffrax as dx
from diffrax.functional import *
from diffrax import LightField, TransFormPropagate
from diffrax.utils.types import LazyGrid, Spectrum
from diffrax.utils import OpticalElement, OpticalLayer

import jax.numpy as jnp
from jax import random
import jax

from einops import rearrange, repeat, reduce
from flax.traverse_util import flatten_dict
from flax.core import unfreeze
import flax.linen as nn

from typing import Tuple, Callable, Sequence, Union
import matplotlib.pyplot as plt

# %%
# Let's define some aspects of our system (based on CBxs_lobV_top_30nm at ESRF beamline i16a):
du = 0.03 # effective pixel size of detector (microns) (note: documented as pixelsize_detector = 3e-6 in ESRF script?)
z_total = 1.208 * 1000 # distance from focus point to detector (microns)
zs_f2s = jnp.array([0.012080, 0.012598, 0.014671, 0.018975]) * 1000 # distances source to sample (microns) 
dz = 1. # depth of slice in microns
zs_s2d = z_total - zs_f2s - dz # distances sample to detector (microns) 
lambda_ =  7.25146e-5 # wavelength in microns for 17kEV (~3.62573e-05 um for 33kEV)
lambda_ratios = 1.0 # ratio of wavelengths
n = 0.999999999 # refractive index of x-rays is slightly below unity according to ESRF

# Get and display Siemens star test object (then make it into a phase object)

data = plt.imread('/n/groups/htem/users/jlr54/raygun/diffrax/Siemens_star.svg.png')
data = jnp.expand_dims(jnp.expand_dims(jnp.abs(jnp.mean(data, axis=-1)), 0), -1) # 'h w -> b h w 1'

# delta and beta should be 4d Tensors
delta = data * 1.2600 # phase change
beta = data * (1 - 0.98200) # 1 minus the extinction coefficient

# Now let's make the lightfields hitting the sample and display their intensity and phase:
n_pixels = data.shape[1]
D = n_pixels * du
in_fields = [pointsource((n_pixels, du), z0, D, n, lambda_, lambda_ratios) for z0 in zs_f2s]

def show_fields(fields):
    _, axs = plt.subplots(2, len(fields), figsize=(20, 3*len(fields)))
    for i, ax in enumerate(axs.T):
        ax[0].imshow(fields[i].intensity.squeeze())
        ax[1].imshow(fields[i].phase.squeeze())

show_fields(in_fields)

# Now let's make some projections:
def phasechange_cache(field: LightField, delta: jnp.ndarray, beta: jnp.ndarray, dz: float) -> Tuple[jnp.ndarray]:
    k = 2 * jnp.pi / field.spectrum.wavelength
    phase_factor = jnp.exp(-1j * k * dz * (delta - 1j * beta))
    return (phase_factor,)


def phasechange_apply(field: LightField, cache: Tuple[jnp.ndarray]) -> LightField:
    (phase_factor,) = cache
    u = field.u * phase_factor
    return field.replace(u=u)

# Register as functions and layers:
phasechange = OpticalElement(phasechange_apply, phasechange_cache)
PhaseChange = OpticalLayer.init_layer("phase change", phasechange)

# %%
out_fields = [phasechange(in_field, delta, beta, dz=dz) for in_field in in_fields]
show_fields(out_fields)

# Let's add free space propagation after the sample:
# %%
def propagate(field, dz, n, D): # here dz is distance to propogate wave
    Nf = jnp.max((D / 2) ** 2 / (field.spectrum.wavelength * dz))
    M = field.u.shape[1]
    Q = M / (4 * Nf)
    N = int(jnp.ceil((Q * M) / 2) * 2)
    # N_pad = int((N - M) / 2)
    return transform_propagate(field, z=dz, n=n, N_pad=N)

sensor_fields = [propagate(in_field, z_s2d, n, D) for in_field, z_s2d in zip(out_fields, zs_s2d)]
sensor_images = [sensor_field.intensity for sensor_field in sensor_fields] # these would be the sensor readings for each distance
show_fields(sensor_fields)

# %% [markdown]
# ### Now let's see if we can reconstruct the sample from the sensor images
# First, let's make some necessary variables, including making objects for parts of our imaging system.
# Let's first remember the elements of our system, using the thin slice approximation for modeling the sample volume:
# 1) ```source```: Coherent monochromatic X-ray beam of wavelength ```lambda_``` and refractive index ```n``` in the system, focused by Kirkpatrick-Baez mirrors distances ```[zs_f2s]``` from the sample, and distance ```z_total``` from the detector/sensor.
# 2) ```phase_object```: The sample volume to estimate, modeled as a thin phase object with ```delta``` refractive index and ```beta``` extinction coefficient, ```[zs_f2s]``` from the beam focal point, and ```[zs_s2d]``` from the detector, with depth ```dz```.
# 3) ```propagation```: Propgation of the beam exiting the ```phase_object``` some ```[zs_s2d]``` distances to the detector/sensor, which should have some shot noise/manufacturing defects that produces the final images. TODO: Add noise OR **make sensor object incorporating noise/response model**

# %% [markdown]
# Instantiate system and extract states vs. parameters. (Parameters are differentiable/trainable, states are not). Here we see the trainable parameters for the phase object: ```delta``` and ```beta```:
# %%
class SingleProjection(nn.Module):    
    du: float = 0.03 # effective pixel size of detector (microns) (note: documented as pixelsize_detector = 3e-6 in ESRF script?)
    z_total: float = 1.208 * 1000 # distance from focus point to detector (microns)
    zs_f2s: jnp.array = jnp.array([0.012080, 0.012598, 0.014671, 0.018975]) * 1000 # distances source to sample (microns) 
    dz: float = 1. # depth of slice in microns
    # zs_s2d = z_total - zs_f2s - dz # distances sample to detector (microns) 
    lambda_: jnp.array =  7.25146e-5 # wavelength in microns for 17kEV (~3.62573e-05 um for 33kEV)
    lambda_ratios: jnp.array = 1.0 # ratio of wavelengths
    n: float = 0.999999999 # refractive index of x-rays is slightly below unity according to ESRF
    field_shape: tuple = (512, 512)

    @property
    def max_Npad(self):
        M = self.field_shape[1]
        Nf = jnp.max(((M * self.du) / 2) ** 2 / (self.lambda_ * self.z_total))
        Q = M / (4 * Nf)
        N = int(jnp.ceil((Q * M) / 2) * 2)
        # return int((N - M) / 2)
        return N

    def setup(self):
        self.input_field = self.param('input_field_param', 
            lambda key, shape, du, lambda_, lambda_ratios: LightField(
                jnp.ones(shape, dtype=jnp.complex64), 
                LazyGrid(shape[0], du), 
                Spectrum(lambda_, lambda_ratios)
            ), 
            self.field_shape, self.du, self.lambda_, self.lambda_ratios)
        
        self.delta = self.param('delta_param', 
            lambda key, shape: jnp.ones(shape), 
            self.field_shape)

        self.beta = self.param('beta_param', 
            lambda key, shape: jnp.ones(shape), 
            self.field_shape)        

        self.z1s = self.param('z1s_param', 
            lambda key, zs: zs, 
            self.zs_f2s)
    
    def __call__(self):
        empty_beam_field = transform_propagate(self.input_field, self.z_total, self.n, self.max_Npad)
        # sample_fields = jax.vmap(transform_propagate, in_axes=(None, 0, None, None))(self.input_field, self.z1s, self.n, self.max_Npad) # 4 x h x w
        # sample_fields = jax.vmap(phasechange, in_axes=(0, None, None, None))(sample_fields, self.delta, self.beta, dz=self.dz)
        # sample_fields = jax.vmap(transform_propagate, in_axes=(0, 0, None, None))(sample_fields, self.z_total - self.z1s - self.dz, self.n, self.max_Npad)
        
        sample_fields = [
            transform_propagate(self.input_field, z, self.n, self.max_Npad) 
            for z in self.z1s
        ] # 4 x h x w
        sample_fields = [
            phasechange(sample_field, self.delta, self.beta, dz=self.dz) 
            for sample_field in sample_fields
        ]
        sample_fields = [
            transform_propagate(sample_field, z, self.n, self.max_Npad) 
            for sample_field, z in zip(sample_fields, self.z_total - self.z1s - self.dz)
        ]

        return empty_beam_field, sample_fields

# %%
projector = SingleProjection()
key = random.PRNGKey(42)

# %%
variables = projector.init(key)
state, params = variables.pop('params') # Split state and params to optimize for
# del variables # Delete variables to avoid wasting resources

flatten_dict(unfreeze(params)).keys()
