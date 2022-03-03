# !conda activate n2v
import torch
from tri_utils import *

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.Logger('CycleGAN_Loss', 'INFO')

class SliceFill_CARE_Loss(torch.nn.Module):
    def __init__(self, 
                Gnet, 
                optimizer,
                l1_fun=torch.nn.L1Loss() # torch.nn.SmoothL1Loss()  
                 ):
        super(SliceFill_CARE_Loss, self).__init__()
        self.l1_fun = l1_fun
        self.Gnet = Gnet # network to generate missing slice
        self.optimizer = optimizer
        self.loss_dict = {
            'Loss/L1': float(),
        }

    def crop(self, x, shape):
        '''Center-crop x to match spatial dimensions given by shape.'''

        x_target_size = x.size()[:-2] + shape

        offset = tuple(
            (a - b)//2
            for a, b in zip(x.size(), x_target_size))

        slices = tuple(
            slice(o, o + s)
            for o, s in zip(offset, x_target_size))

        return x[slices]

    def forward(self, real, pred):
        # set G's gradients to zero
        self.optimizer.zero_grad()        

        real_mid_slice = real[:,1,:,:] 
        pred_mid_slice = pred[:,1,:,:] 

        # crop if necessary
        if pred_mid_slice.size()[-2:] != real_mid_slice.size()[-2:]:
            real_mid_slice = self.crop(real_mid_slice, pred_mid_slice.size()[-2:])

        # get l1 loss
        self.l1_loss = self.l1_fun(real_mid_slice, pred_mid_slice)

        #Calculate gradients
        self.l1_loss.backward()

        #Step optimizer
        self.optimizer.step()             # udpate G's weights

        self.loss_dict.update({
            'Loss/L1': float(self.l1_loss)
        })

        total_loss = self.l1_loss.detach()
        # define dummy backward pass to disable Gunpowder's Train node loss.backward() call
        total_loss.backward = lambda: None

        logger.info(self.loss_dict)
        return total_loss

class SliceFill_ConditionalGAN_Loss(torch.nn.Module):
    def __init__(self, 
                Gnet, 
                Dnet,
                optimizer_G,
                optimizer_D, 
                l1_lambda=1,
                l1_fun=torch.nn.L1Loss(), 
                gan_mode=None,
                 ):
        super(SliceFill_ConditionalGAN_Loss, self).__init__()
        self.Gnet = Gnet # network to generate missing slice
        self.Dnet = Dnet # network to discriminate between real and generated missing slice
        self.optimizer_G = optimizer_G
        self.optimizer_D = optimizer_D
        self.l1_lambda = l1_lambda
        self.l1_fun = l1_fun
        self.gan_mode = gan_mode
        self.gan_loss_fun = GANLoss(gan_mode=gan_mode)

        self.loss_dict = {
            'Loss/Gnet_Score': float(),
            'Loss/Dnet_Score': float(),
            'Loss/L1': float(),
        }
    
    def clamp_weights(self, net, min=-0.01, max=0.01):
        for module in net.model:
            if hasattr(module, 'weight') and hasattr(module.weight, 'data'):
                temp = module.weight.data
                module.weight.data = temp.clamp(min, max)

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def crop(self, x, shape):
        '''Center-crop x to match spatial dimensions given by shape.'''

        x_target_size = x.size()[:-2] + shape

        offset = tuple(
            (a - b)//2
            for a, b in zip(x.size(), x_target_size))

        slices = tuple(
            slice(o, o + s)
            for o, s in zip(offset, x_target_size))

        return x[slices]

    def _backward_D(self, pred, real):        
        # Fake; stop backprop to the generator by detaching fake
        pred_fake = self.Dnet(pred.detach())
        loss_D_fake = self.gan_loss_fun(pred_fake, False)
        
        # Real
        pred_real = self.Dnet(real)
        loss_D_real = self.gan_loss_fun(pred_real, True)
        
        self.loss_D = (loss_D_real + loss_D_fake) * 0.5
        self.loss_D.backward()
        return self.loss_D

    def backward_D(self, pred, real, n_loop=5):
        self.set_requires_grad([self.Dnet], True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero

        if self.gan_mode.lower() == 'wgangp': # Wasserstein Loss
            for _ in range(n_loop):
                self.loss_D = self._backward_D(pred, real)
                self.optimizer_D.step()          # update D's weights
                self.clamp_weights(self.Dnet)
        else:
            self.loss_D = self._backward_D(pred, real)
            self.optimizer_D.step()          # update D's weights            
        
        #return losses
        return self.loss_D

    def backward_G(self, pred, real_mid_slice, pred_mid_slice):
        self.set_requires_grad([self.Dnet], False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        
        # get l1 loss
        self.l1_loss = self.l1_fun(real_mid_slice, pred_mid_slice)        

        # get discriminator loss
        self.gan_loss = self.gan_loss_fun(self.Dnet(pred), True)
        
        #Sum all losses
        self.loss_G = self.gan_loss + self.l1_lambda * self.l1_loss

        #Calculate gradients
        self.loss_G.backward()

        #Step optimizer
        self.optimizer_G.step()             # udpate G's weights

        #return losses
        return self.gan_loss, self.l1_loss

    def forward(self, real, pred):
        
        real_mid_slice = real[:,1,:,:] 
        pred_mid_slice = pred[:,1,:,:] 

        # crop if necessary
        if pred_mid_slice.size()[-2:] != real_mid_slice.size()[-2:]:
            real_mid_slice = self.crop(real_mid_slice, pred_mid_slice.size()[-2:])

        # update Gs
        gan_loss, l1_loss = self.backward_G(pred, real_mid_slice, pred_mid_slice)
        
        # # update Ds
        loss_D = self.backward_D(pred, real)

        self.loss_dict = {
            'DiscrimLoss/Gnet': float(gan_loss),
            'DiscrimLoss/Dnet': float(loss_D),
            'Loss/Gnet_total': float(self.loss_G),
            'Loss/L1': float(l1_loss),
        }

        total_loss = self.loss_G.detach()
        # define dummy backward pass to disable Gunpowder's Train node loss.backward() call
        total_loss.backward = lambda: None

        logger.info(self.loss_dict)
        return total_loss

class SliceFill_UncertainGAN_Loss(torch.nn.Module):
    def __init__(self, 
                Gnet, 
                Dnet,
                optimizer_G,
                optimizer_D,
                nll_lambda=1,
                nll_fun=torch.nn.GaussianNLLLoss(), 
                gan_mode=None,
                 ):
        super(SliceFill_UncertainGAN_Loss, self).__init__()
        self.Gnet = Gnet # network to generate missing slice
        self.Dnet = Dnet # network to discriminate between real and generated missing slice
        self.optimizer_G = optimizer_G
        self.optimizer_D = optimizer_D
        self.nll_lambda = nll_lambda
        self.nll_fun = nll_fun
        self.gan_mode = gan_mode
        self.gan_loss_fun = GANLoss(gan_mode=gan_mode)

        self.loss_dict = {
            'Loss/Gnet_Score': float(),
            'Loss/Dnet_Score': float(),
            'Loss/NLL': float(),
        }
    
    def clamp_weights(self, net, min=-0.01, max=0.01):
        for module in net.model:
            if hasattr(module, 'weight') and hasattr(module.weight, 'data'):
                temp = module.weight.data
                module.weight.data = temp.clamp(min, max)

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def crop(self, x, shape):
        '''Center-crop x to match spatial dimensions given by shape.'''

        x_target_size = x.size()[:-2] + shape

        offset = tuple(
            (a - b)//2
            for a, b in zip(x.size(), x_target_size))

        slices = tuple(
            slice(o, o + s)
            for o, s in zip(offset, x_target_size))

        return x[slices]

    def _backward_D(self, pred, real):        
        # Fake; stop backprop to the generator by detaching fake
        pred_fake = self.Dnet(pred.detach())
        loss_D_fake = self.gan_loss_fun(pred_fake, False)
        
        # Real
        pred_real = self.Dnet(real)
        loss_D_real = self.gan_loss_fun(pred_real, True)
        
        self.loss_D = (loss_D_real + loss_D_fake) * 0.5
        self.loss_D.backward()
        return self.loss_D

    def backward_D(self, pred, real, n_loop=5):
        self.set_requires_grad([self.Dnet], True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero

        if self.gan_mode.lower() == 'wgangp': # Wasserstein Loss
            for _ in range(n_loop):
                self.loss_D = self._backward_D(pred, real)
                self.optimizer_D.step()          # update D's weights
                self.clamp_weights(self.Dnet)
        else:
            self.loss_D = self._backward_D(pred, real)
            self.optimizer_D.step()          # update D's weights            
        
        #return losses
        return self.loss_D

    def backward_G(self, pred, real_mid_slice, pred_mid_slice, pred_var):
        self.set_requires_grad([self.Dnet], False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        
        # get nll loss
        self.nll_loss = self.nll_fun(pred_mid_slice, real_mid_slice, pred_var)        

        # get discriminator loss
        self.gan_loss = self.gan_loss_fun(self.Dnet(pred), True)
        
        #Sum all losses
        self.loss_G = self.gan_loss + self.nll_lambda * self.nll_loss

        #Calculate gradients
        self.loss_G.backward()

        #Step optimizer
        self.optimizer_G.step()             # udpate G's weights

        #return losses
        return self.gan_loss, self.nll_loss

    def forward(self, real, pred, pred_var):
        
        real_mid_slice = real[:,1,:,:] 
        pred_mid_slice = pred[:,1,:,:] 

        # crop if necessary
        if pred_mid_slice.size()[-2:] != real_mid_slice.size()[-2:]:
            real_mid_slice = self.crop(real_mid_slice, pred_mid_slice.size()[-2:])

        # update Gs
        gan_loss, nll_loss = self.backward_G(pred, real_mid_slice, pred_mid_slice, pred_var[:,0,:,:])
        
        # # update Ds
        loss_D = self.backward_D(pred, real)

        self.loss_dict = {
            'DiscrimLoss/Gnet': float(gan_loss),
            'DiscrimLoss/Dnet': float(loss_D),
            'Loss/Gnet_total': float(self.loss_G),
            'Loss/NLL': float(nll_loss),
        }

        total_loss = self.loss_G.detach()
        # define dummy backward pass to disable Gunpowder's Train node loss.backward() call
        total_loss.backward = lambda: None

        logger.info(self.loss_dict)
        return total_loss
