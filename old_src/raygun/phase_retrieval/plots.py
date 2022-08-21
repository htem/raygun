import matplotlib.pyplot as plt
import numpy as np

from jax import lax
import jax.numpy as jnp
from jax import random
import optax

from diffrax import LightField

def show_fields(fields, empty_field=None, orientation='landscape'):
    eb = empty_field is not None
    if fields.u.shape[0] + eb > 6: orientation='portrait'
    intensity = fields.intensity.squeeze(-1)
    phase = fields.phase.squeeze(-1)

    if orientation.lower() == 'portrait':
        _, axs = plt.subplots(fields.u.shape[0] + eb, 2, figsize=(10, 5*(fields.u.shape[0] + eb)))
        if eb:
            axs[0,0].imshow(empty_field.intensity.squeeze(), cmap='viridis')
            axs[0,1].imshow(empty_field.phase.squeeze(), cmap='hsv')

        if len(axs.shape) == 2:
            for i, ax in enumerate(axs[eb:, :]):
                ax[0].imshow(intensity[i,...], cmap='viridis')
                ax[1].imshow(phase[i,...], cmap='hsv')
            
            axs[0,0].set_title('Intensity')
            axs[0,1].set_title('Phase')

        else:            
            axs[0].imshow(intensity[0,...], cmap='viridis')
            axs[1].imshow(phase[0,...], cmap='hsv')            
            axs[0].set_title('Intensity')
            axs[1].set_title('Phase')
    else:
        _, axs = plt.subplots(2, fields.u.shape[0] + eb, figsize=(5*(fields.u.shape[0] + eb), 10))
        if eb:
            axs[0,0].imshow(empty_field.intensity.squeeze(), cmap='viridis')
            axs[1,0].imshow(empty_field.phase.squeeze(), cmap='hsv')

        if len(axs.shape) == 2:
            for i, ax in enumerate(axs[:, eb:].T):
                ax[0].imshow(intensity[i,...], cmap='viridis')
                ax[1].imshow(phase[i,...], cmap='hsv')
            
            axs[0,0].set_title('Intensity')
            axs[1,0].set_title('Phase')
        else:            
            axs[0].imshow(intensity[0,...], cmap='viridis')
            axs[1].imshow(phase[0,...], cmap='hsv')            
            axs[0].set_title('Intensity')
            axs[1].set_title('Phase')


def show_images(images, empty_image=None, orientation='landscape'):
    eb = empty_image is not None
    intensity = images.squeeze(-1)
    if intensity.shape[0] + eb > 6: orientation='portrait'

    if orientation.lower() == 'portrait':
        _, axs = plt.subplots(intensity.shape[0] + eb, 1, figsize=(5, 5*(intensity.shape[0] + eb)))
    else:
        _, axs = plt.subplots(1, intensity.shape[0] + eb, figsize=(5*(intensity.shape[0] + eb), 5))

    if (intensity.shape[0] + eb) > 1:
        if eb:
            axs[0].imshow(empty_image.squeeze(), cmap='viridis')
        for i, ax in enumerate(axs[eb:]):
            ax.imshow(intensity[i,...], cmap='viridis')
        axs[0].set_title('Intensity')
    else:
        axs.imshow(intensity.squeeze(), cmap='viridis')
        axs.set_title('Intensity')


def show_params(params):
    c = 2 + 2*('input_field' in params)
    fig, ax = plt.subplots(1, c, figsize=(5*c, 5))
    plt.subplot(1,c,1, title='Delta:')
    plt.imshow(params['delta'].squeeze())
    plt.colorbar()

    plt.subplot(1,c,2, title='Beta:')
    plt.imshow(params['beta'].squeeze())
    plt.colorbar()

    if 'input_field' in params:
        plt.subplot(1,c,3, title='Input Intensity:')
        plt.imshow(params['input_field'].intensity.squeeze(), cmap='viridis')
        plt.colorbar()

        plt.subplot(1,c,4, title='Input Phase:')
        plt.imshow(params['input_field'].phase.squeeze(), cmap='hsv')
        plt.colorbar()


def show_results(model, params, gt_params, grads=None, method='huber'):
    sim_empty, sim_sample = model.apply({'params': params}).values()
    gt_empty, gt_sample = model.apply({'params': gt_params}).values()

    print('GT Fields (top) vs. Simulated Fields (bottom)')
    show_fields(gt_sample, gt_empty)
    show_fields(sim_sample, sim_empty)

    print('GT parameters (top) vs. current trained parameters (bottom)')
    show_params(gt_params)
    show_params(params)

    if grads is not None:
        print('Current gradients:')
        show_params(grads)

    print(f'{method} Loss')
    show_loss((sim_sample.intensity, gt_sample.intensity), (sim_empty.intensity, gt_empty.intensity), method=method)


def show_random_window_results(model, params, gt_params, key, grads=None, size=512, method='huber'):
    show_results(model, params, gt_params, grads, method)

    sim_empty, sim_sample = model.apply({'params': params}).values()
    gt_empty, gt_sample = model.apply({'params': gt_params}).values()

    # Get random window:
    x, y = random.randint(key, [2,], 0, jnp.array([gt_sample.shape[1] - size, gt_sample.shape[2] - size]))

    print('Sampled Window:')
    win = np.zeros_like(gt_empty.intensity)
    win[:, x:x+size, y:y+size, :] = 1
    plt.figure()
    plt.imshow(win.squeeze())

    gt_empty = lax.dynamic_slice(gt_empty.intensity, [0, x, y, 0], [gt_empty.shape[0], size, size, gt_empty.shape[3]])
    gt_sample = lax.dynamic_slice(gt_sample.intensity, [0, x, y, 0], [gt_sample.shape[0], size, size, gt_sample.shape[3]])
    sim_empty = lax.dynamic_slice(sim_empty.intensity, [0, x, y, 0], [sim_empty.shape[0], size, size, sim_empty.shape[3]])
    sim_sample = lax.dynamic_slice(sim_sample.intensity, [0, x, y, 0], [sim_sample.shape[0], size, size, sim_sample.shape[3]])

    print(f'GT window (top) vs. Simulated window (bottom) vs. {method} Loss')
    show_images(gt_sample, gt_empty)
    show_images(sim_sample, sim_empty)
    show_loss((sim_sample, gt_sample), (sim_empty, gt_empty), method=method)


def show_loss(*pairs, method='huber'):
    if method.lower() == 'huber':
        loss_fcn = optax.huber_loss
    elif method.lower() == 'l2':
        loss_fcn = optax.l2_loss
    elif method.lower() == 'l1':
        loss_fcn = lambda sim, gt: abs(gt - sim)

    losses = []
    for (sim, gt) in pairs:
        losses.append(loss_fcn(sim, gt))

    show_images(*losses)