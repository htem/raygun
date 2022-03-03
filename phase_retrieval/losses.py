import jax.numpy as jnp
from jax import random, lax
import optax
from utils import *

def loss_(pairs, method='huber'):
    
    if method.lower() == 'l1':
        loss_fn = lambda sim, gt: jnp.nanmean(abs(gt - sim))

    if method.lower() == 'l2':
        loss_fn = lambda sim, gt: optax.l2_loss(sim, gt).mean()
        
    if method.lower() == 'huber':
        loss_fn = lambda sim, gt: optax.huber_loss(sim, gt).mean()

    total_num = 0
    total_loss = 0
    for (sim, gt) in pairs:
        num = sim.shape[0]
        total_num += num
        total_loss += loss_fn(sim, gt) * num
    
    return total_loss / total_num # normalize loss so it doesn't depend on the number of sample images


def loss(params, model, method='huber', **gts):
    gts = get_lf_prop('intensity', gts)

    sims = model.apply({'params': params})
    sims = get_lf_prop('intensity', sims)
    
    pairs = []
    for key, sim_val in sims.items():
        if key in gts:
            pairs.append((sim_val, gts[key]))
    
    assert len(pairs) > 0, f'No matching sim and gt data found. Check keys for correspondence (sim: {sims.keys()} vs. gts: {gts.keys()} should match).'
    return loss_(pairs, method)


def random_window_loss(params, model, key, method='huber', size=512, **gts):
    gts = get_lf_prop('intensity', gts)

    sims = model.apply({'params': params})
    sims = get_lf_prop('intensity', sims)
    
    # Get random window:
    gt_ex_val = list(gts.values())[0]
    x, y = random.randint(key, [2,], 0, jnp.array([gt_ex_val.shape[1] - size, gt_ex_val.shape[2] - size]))

    pairs = []
    for key, sim_val in sims.items():
        if key in gts:
            sim_val = lax.dynamic_slice(sim_val, [0, x, y, 0], [sim_val.shape[0], size, size, sim_val.shape[3]])
            gt_val = lax.dynamic_slice(gts[key], [0, x, y, 0], [gts[key].shape[0], size, size, gts[key].shape[3]])
            pairs.append((sim_val, gt_val))
    
    assert len(pairs) > 0, f'No matching sim and gt data found. Check keys for correspondence (sim: {sims.keys()} vs. gts: {gts.keys()} should match).'
    return loss_(pairs, method=method)