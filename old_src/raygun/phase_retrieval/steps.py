from functools import partial
import optax
import jax
from jax import jit, random

def get_step_fn(loss_fn, optimizer, **init_kwargs):
    """Returns update function function."""
    loss_fn = partial(loss_fn, **init_kwargs)

    def step(params, opt_state, *args, **kwargs):
        if 'key' in kwargs: # step RNG key if necessary
            kwargs['key'] = random.split(kwargs['key'])

        loss, grads = jax.value_and_grad(loss_fn)(params, *args, **kwargs)

        # Applying updates
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        outs = [params, opt_state, loss, grads]
        if 'key' in kwargs: # step RNG key if necessary
            outs.append(kwargs['key'])
        return outs

    return jit(step)

