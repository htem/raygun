import flax.linen as nn
import jax.numpy as jnp
from diffrax import LightField
from diffrax.functional import *
from diffrax.elements.sources import *
import numpy as np

class MultiDistance(nn.Module):    
    # du_eff: float = 0.03 # effective pixel size of detector (microns)
    # du: float = 3e-6 # (microns) documented pixelsize_detector in ESRF script
    z_total: float = 1.208 * 1e6 #1000 # distance from focus point to detector (microns)
    zs_f2s: jnp.array = jnp.array([0.012080, 0.012598, 0.014671, 0.018975]) * 1e6 #1000 # distances source to sample (microns) 
    dz: float = 1. # depth of slice in microns
    lambda_: jnp.array =  7.25146e-5 # wavelength in microns for 17kEV (~3.62573e-05 um for 33kEV)
    lambda_ratios: jnp.array = 1.0 # ratio of wavelengths
    n: float = 1 - 1e-9 # refractive index of x-rays is slightly below unity according to ESRF
    field_shape: int = 2048 #TODO: Determine if there should also be n_pixels
    D: float = 30e-3 # aperture width in microns (corresponding to *smallest beam width at focal point post Kirkpatrick-Baez mirrors)
    prop_fcn: Callable = exact_propagate
    
    @property
    def max_Npad(self):
        Nf = np.max((self.D / 2) ** 2 / (self.lambda_ * self.z_total)) # Fresnel number
        # N = np.ceil(1 / (8 * Nf)) * 2
        # N_pad = int(np.clip(N - self.field_shape, 0, None))
        return int(np.clip(np.ceil((self.field_shape - (4 * Nf)) / 2) * 2, 0, None))
    
    @classmethod
    def phasechange(self, delta, beta, field: LightField) -> LightField:
        k = 2 * jnp.pi / field.wavelengths
        u = field.u * jnp.exp(-1j * k * self.dz * (delta - 1j * beta))
        return field.replace(u=u)

    def setup(self):        
        self.input_field = create_field(pointsource(self.dz, self.D, self.n), (self.field_shape, self.field_shape), self.D / self.field_shape, self.lambda_, self.lambda_ratios)

        # self.input_field = self.param('input_field', 
        #     lambda key, dz, shape, D, n, lambda_, lambda_ratios: create_field(pointsource(dz, D, n), (shape, shape), D / shape, lambda_, lambda_ratios), 
        #     self.dz, self.field_shape, self.D, self.n, self.lambda_, self.lambda_ratios)

        self.delta = self.param('delta', 
            lambda key, shape: jnp.zeros((1, shape, shape, 1)), # ~corresponding to air
            self.field_shape)

        self.beta = self.param('beta', 
            lambda key, shape: jnp.zeros((1, shape, shape, 1)),  # ~corresponding to air
            self.field_shape)        

        self.z1s = jnp.array(self.zs_f2s).reshape((len(self.zs_f2s),1,1,1))
        # self.z1s = self.param('z1s', 
        #     lambda key, zs: jnp.array(zs).reshape((len(zs),1,1,1)), # make properly dimensioned array (b x 1 x 1 x 1)
        #     self.zs_f2s)
    
    def __call__(self) -> LightField:
        empty_beam_field = self.prop_fcn(field = self.input_field, z = self.z_total, n = self.n, N_pad = self.max_Npad) # 1 x h x w x 1
        sample_fields = self.prop_fcn(field = self.input_field, z = self.z1s, n = self.n, N_pad = self.max_Npad) 
        sample_fields = self.phasechange(self.delta, self.beta, sample_fields)
        sample_fields = self.prop_fcn(field = sample_fields, z = self.z_total - self.z1s - self.dz, n = self.n, N_pad = self.max_Npad)
        
        return {'empty': empty_beam_field, 'sample': sample_fields}

class SingleProjection(nn.Module):    
    # du_eff: float = 0.03 # effective pixel size of detector (microns)
    # du: float = 3e-6 # (microns) documented pixelsize_detector in ESRF script
    z_total: float = 1.208 * 1e6 #1000 # distance from focus point to detector (microns)
    zs_f2s: jnp.array = jnp.array([0.012080, 0.012598, 0.014671, 0.018975]) * 1e6 #1000 # distances source to sample (microns) 
    dz: float = 1. # depth of slice in microns
    lambda_: jnp.array =  7.25146e-5 # wavelength in microns for 17kEV (~3.62573e-05 um for 33kEV)
    lambda_ratios: jnp.array = 1.0 # ratio of wavelengths
    n: float = 1 - 1e-9 # refractive index of x-rays is slightly below unity according to ESRF
    field_shape: int = 2048 #TODO: Determine if there should also be n_pixels
    D: float = 30e-3 # aperture width in microns (corresponding to *smallest beam width at focal point post Kirkpatrick-Baez mirrors)
    prop_fcn: Callable = exact_propagate

    @property
    def max_Npad(self):
        Nf = jnp.max((self.du / 2) ** 2 / (self.lambda_ * self.z_total))
        N = jnp.ceil(1 / (8 * Nf)) * 2
        return jnp.clip(N - self.field_shape, 0).astype(int)    
    
    @classmethod
    def phasechange(self, field: LightField) -> LightField:
        k = 2 * jnp.pi / field.wavelengths
        u = field.u * jnp.exp(-1j * k * self.dz * (self.delta - 1j * self.beta))
        return field.replace(u=u)

    @classmethod
    def project(self, z1):
        sample_field = self.prop_fcn(field = self.input_field, z = z1, n = self.n, N_pad = self.max_Npad)
        sample_field = self.phasechange(sample_field)
        return self.prop_fcn(field = sample_field, z = self.z_total - z1 - self.dz, n = self.n, N_pad = self.max_Npad)

    def setup(self):     
        self.input_field = create_field(pointsource(self.dz, self.D, self.n), (self.field_shape, self.field_shape), self.D / self.field_shape, self.lambda_, self.lambda_ratios)

        # self.input_field = self.param('input_field', 
        #     lambda key, dz, shape, D, n, lambda_, lambda_ratios: create_field(pointsource(dz, D, n), (shape, shape), D / shape, lambda_, lambda_ratios), 
        #     self.dz, self.field_shape, self.D, self.n, self.lambda_, self.lambda_ratios)

        self.delta = self.param('delta', 
            lambda key, shape: jnp.zeros((1, shape, shape, 1)), # ~corresponding to air
            self.field_shape)

        self.beta = self.param('beta', 
            lambda key, shape: jnp.zeros((1, shape, shape, 1)),  # ~corresponding to air
            self.field_shape)        

        self.z1s = jnp.array(self.zs_f2s).reshape((len(self.zs_f2s),1,1,1))
        # self.z1s = self.param('z1s', 
        #     lambda key, zs: jnp.array(zs).reshape((len(zs),1,1,1)), # make properly dimensioned array (b x 1 x 1 x 1)
        #     self.zs_f2s)
    
    def __call__(self, z1) -> LightField:
        if z1 is None:
            field = self.prop_fcn(field = self.input_field, z = self.z_total, n = self.n, N_pad = self.max_Npad) # 1 x h x w x 1
        else:
            field = self.project(z1)
                
        return {'sample': field}

class EmptyBeam(nn.Module):    
    # du_eff: float = 0.03 # effective pixel size of detector (microns)
    # du: float = 3e-6 # (microns) documented pixelsize_detector in ESRF script
    z_total: float = 1.208 * 1e6 # distance from focus point to detector (microns)
    lambda_: jnp.array =  7.25146e-5 # wavelength in microns for 17kEV (~3.62573e-05 um for 33kEV)
    lambda_ratios: jnp.array = 1.0 # ratio of wavelengths
    n: float = 1 - 1e-9 # refractive index of x-rays is slightly below unity according to ESRF
    field_shape: int = 2048 #TODO: Determine if there should also be n_pixels
    D: float = 30e-3 # aperture width in microns (corresponding to *smallest beam width at focal point post Kirkpatrick-Baez mirrors)
    prop_fcn: Callable = transfer_propagate
    
    @classmethod
    def input_field(self, u):
        field = create_field(pointsource(1., self.D, self.n), (self.field_shape, self.field_shape), self.D / self.field_shape, self.lambda_, self.lambda_ratios)
        return field.replace(u=u)

    @property
    def max_Npad(self):
        Nf = np.max((self.D / 2) ** 2 / (self.lambda_ * self.z_total)) # Fresnel number
        # N = np.ceil(1 / (8 * Nf)) * 2
        # N_pad = int(np.clip(N - self.field_shape, 0, None))
        return int(np.clip(np.ceil((self.field_shape - (4 * Nf)) / 2) * 2, 0, None))

    def setup(self):
        base_field = create_field(pointsource(1., self.D, self.n), (self.field_shape, self.field_shape), self.D / self.field_shape, self.lambda_, self.lambda_ratios)
        
        # base_field = create_field(square_aperture(self.max_Npad), (self.field_shape, self.field_shape), self.D / self.field_shape, self.lambda_, self.lambda_ratios)

        self.u = self.param('u', lambda key, field: field.u, base_field)        
        # self.input_field = self.param('input_field', 
        #     lambda key, shape, D, n, lambda_, lambda_ratios: create_field(pointsource(1., D, n), (shape, shape), D / shape, lambda_, lambda_ratios), self.field_shape, self.D, self.n, self.lambda_, self.lambda_ratios)
    
    def __call__(self) -> LightField:
        empty_beam_field = self.prop_fcn(field = self.input_field(self.u), z = self.z_total, n = self.n, N_pad = self.max_Npad) # 1 x h x w x 1
        
        return {'empty': empty_beam_field}


class Sensor(nn.Module):
    n_x_pixels: int = 2048 # must be even number
    n_y_pixels: int = 2048 # must be even number
    pixel_size: float = 3 # (microns) documented pixelsize_detector in ESRF script (as 3e-6, assumed in meters)
    dark_image: jnp.array = jnp.zeros((2048, 2048))

    # Grid properties
    @property
    def grid(self):
        half_size = jnp.array([self.n_x_pixels, self.n_y_pixels]) / 2
        # We must (!!) use a linspace here as mgrid the
        # num of arguments can be set similar to the shape
        # of the field allowing jit.

        grid = jnp.meshgrid(
            jnp.linspace(-half_size[0], half_size[0] - 1, num=self.n_x_pixels),
            jnp.linspace(-half_size[1], half_size[1] - 1, num=self.n_y_pixels),
            indexing="ij",
        )
        grid = rearrange(grid, "d h w -> d 1 h w 1")
        return self.pixel_size * grid

    def setup(self):
        # TODO: add noise, beam_offset, etc. parameters for training
        ...

    def __call__(self, field: LightField):
        # Integrate field.intensity over grid the size of the sensor
        # 1) Crop out / pad incoming field to match world unit size of sensor --> in_array
        # 2) Integrate / interpolate in_array to match n_pixels of sensor
        ...