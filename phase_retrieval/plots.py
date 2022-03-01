import matplotlib.pyplot as plt

def show_fields(fields, empty_field=None, orientation='landscape'):
    eb = empty_field is not None
    intensity = fields.intensity.squeeze()
    phase = fields.phase.squeeze()

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
            axs[0].imshow(intensity, cmap='viridis')
            axs[1].imshow(phase, cmap='hsv')            
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
            axs[0].imshow(intensity, cmap='viridis')
            axs[1].imshow(phase, cmap='hsv')            
            axs[0].set_title('Intensity')
            axs[1].set_title('Phase')

def show_images(images, empty_image=None, orientation='landscape'):
    eb = empty_image is not None
    intensity = images.squeeze()

    if orientation.lower() == 'portrait':
        _, axs = plt.subplots(intensity.shape[0] + eb, 1, figsize=(5, 5*(intensity.shape[0] + eb)))
    else:
        _, axs = plt.subplots(1, intensity.shape[0] + eb, figsize=(5*(intensity.shape[0] + eb), 5))

    if eb:
        axs[0].imshow(empty_image.squeeze(), cmap='viridis')
    for i, ax in enumerate(axs[eb:]):
        ax.imshow(intensity[i,...], cmap='viridis')
    axs[0].set_title('Intensity')

def show_params(params):
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    plt.subplot(141, title='Delta:')
    plt.imshow(params['delta_param'].squeeze())
    plt.colorbar()

    plt.subplot(142, title='Beta:')
    plt.imshow(params['beta_param'].squeeze())
    plt.colorbar()

    if 'input_field_param' in params:
        plt.subplot(143, title='Input Intensity:')
        plt.imshow(params['input_field_param'].intensity.squeeze(), cmap='viridis')
        plt.colorbar()

        plt.subplot(144, title='Input Phase:')
        plt.imshow(params['input_field_param'].phase.squeeze(), cmap='hsv')
        plt.colorbar()
