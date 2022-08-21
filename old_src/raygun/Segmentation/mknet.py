import os
import jax
import jax.numpy as jnp
from jax import jit
import haiku as hk
import optax
import jmp
import time
import json
from funlib.learn.jax.models import UNet, ConvPass
from typing import Tuple, Any, NamedTuple, Dict


class GenericJaxModel():
    '''Generic training model for Gunpowder'''

    def __init__(self):
        pass

    def initialize(self, rng_key, inputs, is_training):
        raise RuntimeError("Unimplemented")

    def forward(self, params, inputs):
        raise RuntimeError("Unimplemented")

    def train_step(self, params, inputs, pmapped=False):
        raise RuntimeError("Unimplemented")


# PARAMETERS
mp_training = True  # mixed-precision training using `jmp`
# mp_training = False  # mixed-precision training using `jmp`
learning_rate = 0.5e-4 * 0.25
pmap_use_psum = False


class Params(NamedTuple):
    weight: jnp.ndarray
    opt_state: jnp.ndarray
    loss_scale: jmp.LossScale


class Model(GenericJaxModel):

    def __init__(self):
        super().__init__()

        # we encapsulate the UNet and the ConvPass in one hk.Module
        # to make assigning precision policy easier
        class MyModel(hk.Module):

            def __init__(self, name=None):
                super().__init__(name=name)
                self.unet = UNet(
                    num_fmaps=24,
                    fmap_inc_factor=3,
                    downsample_factors=[[2,2,2],[2,2,2],[2,2,2]],
                    # downsample_factors=[],
                    )
                self.conv = ConvPass(
                    kernel_sizes=[[1,1,1]],
                    # out_channels=3,
                    out_channels=12,  # long range affs
                    activation='sigmoid',
                    )

            def __call__(self, x):
                return self.conv(self.unet(x))

        def _forward_fn(x):
            net = MyModel()
            return net(x)

        if mp_training:
            policy = jmp.get_policy('p=f32,c=f16,o=f32')
        else:
            policy = jmp.get_policy('p=f32,c=f32,o=f32')
        hk.mixed_precision.set_policy(MyModel, policy)

        self.model = hk.without_apply_rng(hk.transform(_forward_fn))
        self.opt = optax.adam(learning_rate, b1=0.95, b2=0.999)
        # self.opt = optax.adam(learning_rate)

        @jit
        def _forward(params, inputs):
            return {'affs': self.model.apply(params.weight, inputs['raw'])}

        self.forward = _forward

        @jit
        def _loss_fn(weight, raw, gt, mask, loss_scale):
            pred_affs = self.model.apply(weight, x=raw)
            loss = optax.l2_loss(predictions=pred_affs, targets=gt)
            loss = loss*2*mask  # optax divides loss by 2 so we mult it back
            loss_mean = loss.mean(where=mask)
            return loss_scale.scale(loss_mean), (pred_affs, loss, loss_mean)

        self.loss_fn = _loss_fn

        @jit
        def _apply_optimizer(params, grads):
            updates, new_opt_state = self.opt.update(grads, params.opt_state)
            new_weight = optax.apply_updates(params.weight, updates)
            return new_weight, new_opt_state

        def _train_step(params, inputs, pmapped=False) -> Tuple[Params, Dict[str, jnp.ndarray], Any]:

            raw, gt, mask = inputs['raw'], inputs['gt'], inputs['mask']
            #raw, gt = inputs['raw'], inputs['gt']

            grads, (pred_affs, loss, loss_mean) = jax.grad(
                _loss_fn, has_aux=True)(params.weight, raw, gt,
                                        mask,
                                        params.loss_scale)

            if pmapped:
                # sync grads, casting to compute precision (f16) for efficiency
                grads = policy.cast_to_compute(grads)
                if pmap_use_psum:
                    grads = jax.lax.psum(grads, axis_name='num_devices')
                else:
                    grads = jax.lax.pmean(grads, axis_name='num_devices')
                grads = policy.cast_to_param(grads)

            # dynamic mixed precision loss scaling
            grads = params.loss_scale.unscale(grads)
            new_weight, new_opt_state = _apply_optimizer(params, grads)

            # if any update is non-finite, skip updates
            grads_finite = jmp.all_finite(grads)
            new_loss_scale = params.loss_scale.adjust(grads_finite)
            new_weight, new_opt_state = jmp.select_tree(
                                            grads_finite,
                                            (new_weight, new_opt_state),
                                            (params.weight, params.opt_state))

            new_params = Params(new_weight, new_opt_state, new_loss_scale)
            outputs = {'affs': pred_affs[0:3], 'grad': loss}
            return new_params, outputs, loss_mean

        self.train_step = _train_step

    def initialize(self, rng_key, inputs, is_training=True):
        weight = self.model.init(rng_key, inputs['raw'])
        opt_state = self.opt.init(weight)
        if mp_training:
            loss_scale = jmp.DynamicLossScale(jmp.half_dtype()(2 ** 15),
                                              period=1000)
            # loss_scale = jmp.DynamicLossScale(jmp.half_dtype()(2 ** 15))
        else:
            loss_scale = jmp.NoOpLossScale()
        return Params(weight, opt_state, loss_scale)


def split(arr, n_devices):
  """Splits the first axis of `arr` evenly across the number of devices."""
  return arr.reshape(n_devices, arr.shape[0] // n_devices, *arr.shape[1:])


def create_network():
    return Model()


def make_config(input_shape, config_f):
    model = Model()

    raw = jnp.ones((1, 1) + tuple(input_shape))
    # gt = jnp.zeros([1, 3, 40, 40, 40])
    # mask = jnp.ones([1, 3, 40, 40, 40])
    rng = jax.random.PRNGKey(42)
    params = model.initialize(
                    rng, {'raw': raw}, is_training=True)

    output = model.forward(params, {'raw': raw})
    output_shape = output['affs'].shape[2:]

    config = {
        'input_shape': input_shape,
        'output_shape': output_shape,
    }

    config['inputs'] = ['raw']

    config['outputs'] = {
        'affs': {
            'out_dims': 3,
            'out_dtype': 'uint8',
            # 'scale_shift': [1, 0],
        }
    }

    with open(config_f + '.json', 'w') as f:
        json.dump(config, f)


if __name__ == "__main__":

    # make_config((132, 132, 132), 'train_net')
    # make_config((244, 244, 244), 'test_net')  # new jax, 8GB, full precision
    # make_config((268, 268, 268), 'test_net')  # new jax, 8GB
    # make_config((284, 284, 284), 'test_net')  # old jax, 8GB
    # make_config((292, 292, 292), 'test_net')
    # d = -8*4  # failed 2080Ti
    # d = -8*2  # passed
    # d = 8*0  # failed 2080
    # d = 8*2  # 
    # # d = 8*1  # 
    # make_config((292-d, 292-d, 292-d), 'test_net')

    # d = -8*0
    # make_config((268+d, 268+d, 268+d), 'test_net')
    d = 8*2
    make_config((268+d, 268+d, 268+d), 'test_net')

    exit(0)

    my_model = Model()

    n_devices = jax.local_device_count()
    batch_size = 4*n_devices

    raw = jnp.ones([batch_size, 1, 132-32, 132-32, 132-32])
    gt = jnp.zeros([batch_size, 3, 8, 8, 8])
    mask = jnp.ones([batch_size, 3, 8, 8, 8])
    # raw = jnp.ones([batch_size, 1, 132-32, 132-32, 132-32])
    # gt = jnp.zeros([batch_size, 24, 128-32, 128-32, 128-32])
    # mask = jnp.ones([batch_size, 24, 128-32, 128-32, 128-32])
    # raw = jnp.ones([batch_size, 1, 132-32, 132-32, 132-32])
    # gt = jnp.zeros([batch_size, 3, 132-32, 132-32, 132-32])
    # mask = jnp.ones([batch_size, 3, 132-32, 132-32, 132-32])
    inputs = {
        'raw': raw,
        'gt': gt,
        'mask': mask,
    }
    rng = jax.random.PRNGKey(42)

    # init model
    if n_devices > 1:
        # split input for pmap
        raw = split(raw, n_devices)
        gt = split(gt, n_devices)
        mask = split(mask, n_devices)
        inputs = {
            'raw': raw,
            'gt': gt,
            'mask': mask,
        }
        rng = jnp.broadcast_to(rng, (n_devices,) + rng.shape)
        model_params = jax.pmap(my_model.initialize)(rng, inputs)
    else:
        model_params = my_model.initialize(rng, inputs, is_training=True)

    for _ in range(10):
        t0 = time.time()

        if n_devices > 1:
            model_params, outputs, loss = jax.pmap(my_model.parallel_train_step,
                                axis_name='num_devices',donate_argnums=(0,))(
                            params=model_params, inputs=inputs)
        else:
            model_params, outputs, loss = jax.jit(my_model.train_step)(
                            params=model_params, inputs=inputs)

        print(f'Took {time.time()-t0}s, loss: {loss}')
