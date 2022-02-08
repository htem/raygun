from gunpowder import *

def get_discriminator_FOV(net):
    # Returns the receptive field of one output neuron for a network (written for patch discriminators)
    # See https://distill.pub/2019/computing-receptive-fields/#solving-receptive-field-region for formula derivation
    
    L = 0 # num of layers
    k = [] # [kernel width at layer l]
    s = [] # [stride at layer i]
    for layer in net.model:
        if hasattr(layer, 'kernel_size'):
            L += 1
            k += [layer.kernel_size[-1]]
            s += [layer.stride[-1]]
    
    r = 1
    for l in range(L-1, 0, -1):
        r = s[l]*r + (k[l] - s[l])

    return r