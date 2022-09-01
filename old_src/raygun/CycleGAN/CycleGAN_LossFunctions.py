# !conda activate n2v
import torch
from utils import *

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.Logger('CycleGAN_Loss', 'INFO')

class CycleGAN_Loss(torch.nn.Module):
    def __init__(self, 
                netD1, 
                netG1, 
                netD2, 
                netG2, 
                optimizer_D, 
                optimizer_G, 
                dims,
                l1_loss = torch.nn.SmoothL1Loss(), 
                l1_lambda=100, 
                identity_lambda=0,
                gan_mode='lsgan'
                 ):
        super(CycleGAN_Loss, self).__init__()
        self.l1_loss = l1_loss
        self.gan_loss = GANLoss(gan_mode=gan_mode)
        self.netD1 = netD1 # differentiates between fake and real Bs
        self.netG1 = netG1 # turns As into Bs
        self.netD2 = netD2 # differentiates between fake and real As
        self.netG2 = netG2 # turns Bs into As
        self.optimizer_D = optimizer_D
        self.optimizer_G = optimizer_G
        self.l1_lambda = l1_lambda
        self.identity_lambda = identity_lambda
        self.gan_mode = gan_mode
        self.dims = dims
        self.loss_dict = {
            'Loss/D1': float(),
            'Loss/D2': float(),
            'Loss/cycle': float(),
            'GAN_Loss/G1': float(),
            'GAN_Loss/G2': float(),
        }

    def crop(self, x, shape):
        '''Center-crop x to match spatial dimensions given by shape.'''

        x_target_size = x.size()[:-self.dims] + shape

        offset = tuple(
            (a - b)//2
            for a, b in zip(x.size(), x_target_size))

        slices = tuple(
            slice(o, o + s)
            for o, s in zip(offset, x_target_size))

        return x[slices]

    def clamp_weights(self, net, min=-0.01, max=0.01):
        for module in net.model:
            if hasattr(module, 'weight') and hasattr(module.weight, 'data'):
                temp = module.weight.data
                module.weight.data = temp.clamp(min, max)

    def backward_D(self, Dnet, real, fake):
        # Real
        pred_real = Dnet(real)
        loss_D_real = self.gan_loss(pred_real, True)
        
        # Fake; stop backprop to the generator by detaching fake
        pred_fake = Dnet(fake.detach())
        loss_D_fake = self.gan_loss(pred_fake, False)

        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_Ds(self, real_A, fake_A, real_B, fake_B, n_loop=5):
        # self.set_requires_grad([self.netG1, self.netG2], False)  # G does not require gradients when optimizing D
        self.set_requires_grad([self.netD1, self.netD2], True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero

        if self.gan_mode.lower() == 'wgangp': # Wasserstein Loss
            for _ in range(n_loop):
                self.loss_D1 = self.backward_D(self.netD1, real_B, fake_B)
                self.loss_D2 = self.backward_D(self.netD2, real_A, fake_A)
                self.optimizer_D.step()          # update D's weights
                self.clamp_weights(self.netD1)
                self.clamp_weights(self.netD2)
        else:
            self.loss_D1 = self.backward_D(self.netD1, real_B, fake_B)
            self.loss_D2 = self.backward_D(self.netD2, real_A, fake_A)
            self.optimizer_D.step()          # update D's weights            
        
        #return losses
        return self.loss_D1, self.loss_D2

    def backward_G(self, real_A, fake_A, cycled_A, real_B, fake_B, cycled_B):
        self.set_requires_grad([self.netD1, self.netD2], False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero

        #get cycle loss for both directions (i.e. real == cycled, a.k.a. real_A == netG2(netG1(real_A)) for A and B)
        # crop if necessary
        if real_A.size()[-self.dims:] != cycled_A.size()[-self.dims:]:
            l1_loss_A = self.l1_loss(self.crop(real_A, cycled_A.size()[-self.dims:]), cycled_A)
            l1_loss_B = self.l1_loss(self.crop(real_B, cycled_B.size()[-self.dims:]), cycled_B)
        else:
            l1_loss_A = self.l1_loss(real_A, cycled_A)
            l1_loss_B = self.l1_loss(real_B, cycled_B)        
        self.loss_dict.update({
            'Cycle_Loss/A': float(l1_loss_A),                
            'Cycle_Loss/B': float(l1_loss_B),                
        })
        cycle_loss = self.l1_lambda * (l1_loss_A + l1_loss_B)

        #get identity loss (i.e. ||G_A(B) - B|| for G_A(A) --> B) if applicable
        if self.identity_lambda > 0:
            identity_B = self.netG1(real_B)
            identity_A = self.netG2(real_A)
            if real_A.size()[-self.dims:] != identity_A.size()[-self.dims:]:
                identity_loss_B = self.l1_loss(self.crop(real_B, identity_B.size()[-self.dims:]), identity_B)
                identity_loss_A = self.l1_loss(self.crop(real_A, identity_A.size()[-self.dims:]), identity_A)
            else:
                identity_loss_B = self.l1_loss(real_B, identity_B)#TODO: add ability to have unique loss function for identity
                identity_loss_A = self.l1_loss(real_A, identity_A)
            self.loss_dict.update({
                'Identity_Loss/A': float(identity_loss_A),                
                'Identity_Loss/B': float(identity_loss_B),                
            })
        else:
            identity_loss_B = 0
            identity_loss_A = 0
        identity_loss = self.identity_lambda * (identity_loss_A + identity_loss_B)

        #Then G1 discriminator loss first
        gan_loss_G1 = self.gan_loss(self.netD1(fake_B), True)

        #Then G2 discriminator loss
        gan_loss_G2 = self.gan_loss(self.netD2(fake_A), True)
        
        #Sum all losses
        self.loss_G = cycle_loss + identity_loss + gan_loss_G1 + gan_loss_G2

        #Calculate gradients
        self.loss_G.backward()

        #Step optimizer
        self.optimizer_G.step()             # udpate G's weights

        #return losses
        return cycle_loss, gan_loss_G1, gan_loss_G2

    def forward(self, real_A, fake_A, cycled_A, real_B, fake_B, cycled_B):

        # crop if necessary
        if real_A.size()[-self.dims:] != fake_B.size()[-self.dims:]:
            real_A = self.crop(real_A, fake_A.size()[-self.dims:])
            real_B = self.crop(real_B, fake_B.size()[-self.dims:])

        # update Gs
        cycle_loss, gan_loss_G1, gan_loss_G2 = self.backward_G(real_A, fake_A, cycled_A, real_B, fake_B, cycled_B)
        
        # update Ds
        loss_D1, loss_D2 = self.backward_Ds(real_A, fake_A, real_B, fake_B)

        self.loss_dict.update({
            'Loss/D1': float(loss_D1),
            'Loss/D2': float(loss_D2),
            'Loss/cycle': float(cycle_loss),
            'GAN_Loss/G1': float(gan_loss_G1),
            'GAN_Loss/G2': float(gan_loss_G2),
        })

        total_loss = self.loss_G.detach()
        # define dummy backward pass to disable Gunpowder's Train node loss.backward() call
        total_loss.backward = lambda: None

        logger.info(self.loss_dict)
        return total_loss

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

class SplitGAN_Loss(torch.nn.Module):
    def __init__(self, 
                netD1, 
                netG1, 
                netD2, 
                netG2, 
                optimizer_G1, 
                optimizer_G2, 
                optimizer_D, 
                dims,
                l1_loss = torch.nn.SmoothL1Loss(), 
                l1_lambda=100, 
                identity_lambda=0,
                gan_mode='lsgan'
                 ):
        super(SplitGAN_Loss, self).__init__()
        self.l1_loss = l1_loss
        self.gan_loss = GANLoss(gan_mode=gan_mode)
        self.netD1 = netD1 # differentiates between fake and real Bs
        self.netG1 = netG1 # turns As into Bs
        self.netD2 = netD2 # differentiates between fake and real As
        self.netG2 = netG2 # turns Bs into As
        self.optimizer_G1 = optimizer_G1
        self.optimizer_G2 = optimizer_G2
        self.optimizer_D = optimizer_D
        self.l1_lambda = l1_lambda
        self.identity_lambda = identity_lambda
        self.gan_mode = gan_mode
        self.dims = dims
        self.loss_dict = {
            'Loss/D1': float(),
            'Loss/D2': float(),
            'Loss/G1': float(),
            'Loss/G2': float(),
        }

    def crop(self, x, shape):
        '''Center-crop x to match spatial dimensions given by shape.'''

        x_target_size = x.size()[:-self.dims] + shape

        offset = tuple(
            (a - b)//2
            for a, b in zip(x.size(), x_target_size))

        slices = tuple(
            slice(o, o + s)
            for o, s in zip(offset, x_target_size))

        return x[slices]
                
    def clamp_weights(self, net, min=-0.01, max=0.01):
        for module in net.model:
            if hasattr(module, 'weight') and hasattr(module.weight, 'data'):
                temp = module.weight.data
                module.weight.data = temp.clamp(min, max)

    def backward_D(self, Dnet, real, fake):
        # Real
        pred_real = Dnet(real)
        loss_D_real = self.gan_loss(pred_real, True)
        
        # Fake; stop backprop to the generator by detaching fake
        pred_fake = Dnet(fake.detach())
        loss_D_fake = self.gan_loss(pred_fake, False)

        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_Ds(self, real_A, fake_A, real_B, fake_B, n_loop=5):
        # self.set_requires_grad([self.netG1, self.netG2], False)  # G does not require gradients when optimizing D
        self.set_requires_grad([self.netD1, self.netD2], True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero

        if self.gan_mode.lower() == 'wgangp': # Wasserstein Loss
            for _ in range(n_loop):
                self.loss_D1 = self.backward_D(self.netD1, real_B, fake_B)
                self.loss_D2 = self.backward_D(self.netD2, real_A, fake_A)
                self.optimizer_D.step()          # update D's weights
                self.clamp_weights(self.netD1)
                self.clamp_weights(self.netD2)
        else:
            self.loss_D1 = self.backward_D(self.netD1, real_B, fake_B)
            self.loss_D2 = self.backward_D(self.netD2, real_A, fake_A)
            self.optimizer_D.step()          # update D's weights            
        
        #return losses
        return self.loss_D1, self.loss_D2

    def backward_G(self, side, Gnet, Dnet, real, fake, cycled):
        """Calculate GAN and L1 loss for the generator"""        
        # First, G(A) should fake the discriminator
        pred_fake = Dnet(fake)
        gan_loss = self.gan_loss(pred_fake, True)
        
        # Include L1 loss for forward and reverse cycle consistency
        # crop if necessary
        if real.size()[-self.dims:] != cycled.size()[-self.dims:]:
            cycle_loss = self.l1_lambda * self.l1_loss(self.crop(real, cycled.size()[-self.dims:]), cycled)
        else:
            cycle_loss = self.l1_lambda * self.l1_loss(real, cycled)                 
        self.loss_dict.update({
            'Cycle_Loss/'+side: float(cycle_loss)            
        })
        
        # Combine losses
        loss_G = cycle_loss + gan_loss
        #get identity loss (i.e. ||G_A(B) - B|| for G_A(A) --> B) and add if applicable
        if self.identity_lambda > 0:
            identity = Gnet(real)
            if real.size()[-self.dims:] != identity.size()[-self.dims:]:
                identity_loss = self.l1_loss(self.crop(real, identity.size()[-self.dims:]), identity)
            else:
                identity_loss = self.l1_loss(real, identity)#TODO: add ability to have unique loss function for identity             
            self.loss_dict.update({
                'Identity_Loss/'+side: float(identity_loss)            
            })
            loss_G = loss_G + self.identity_lambda*identity_loss

        # calculate gradients
        loss_G.backward()
        return loss_G

    def backward_Gs(self, real_A, fake_A, cycled_A, real_B, fake_B, cycled_B):
        self.set_requires_grad([self.netD1, self.netD2], False)  # D requires no gradients when optimizing G

        #G1 first
        self.set_requires_grad([self.netG1], True)  # G1 requires gradients when optimizing
        self.set_requires_grad([self.netG2], False)  # G2 requires no gradients when optimizing G1
        self.optimizer_G1.zero_grad()        # set G1's gradients to zero
        loss_G1 = self.backward_G('B', self.netG1, self.netD1, real_B, fake_B, cycled_B)                   # calculate gradient for G
        self.optimizer_G1.step()             # udpate G1's weights

        #Then G2
        self.set_requires_grad([self.netG2], True)  # G2 requires gradients when optimizing
        self.set_requires_grad([self.netG1], False)  # G1 requires no gradients when optimizing G2
        self.optimizer_G2.zero_grad()        # set G2's gradients to zero
        loss_G2 = self.backward_G('A', self.netG2, self.netD2, real_A, fake_A, cycled_A)                   # calculate gradient for G
        self.optimizer_G2.step()             # udpate G2's weights

        # Turn gradients back on
        self.set_requires_grad([self.netG1], True)
        #return losses
        return loss_G1, loss_G2

    def forward(self, real_A, fake_A, cycled_A, real_B, fake_B, cycled_B):
        
        # crop if necessary
        if real_A.size()[-self.dims:] != fake_B.size()[-self.dims:]:
            real_A = self.crop(real_A, fake_A.size()[-self.dims:])
            real_B = self.crop(real_B, fake_B.size()[-self.dims:])

        # update Gs
        loss_G1, loss_G2 = self.backward_Gs(real_A, fake_A, cycled_A, real_B, fake_B, cycled_B)
        
        # update Ds
        loss_D1, loss_D2 = self.backward_Ds(real_A, fake_A, real_B, fake_B)
        
        self.loss_dict.update({
            'Loss/D1': float(loss_D1),
            'Loss/D2': float(loss_D2),
            'Loss/G1': float(loss_G1),
            'Loss/G2': float(loss_G2),
        })

        total_loss = loss_G1 + loss_G2 #+ loss_D1 + loss_D2
        # define dummy backward pass to disable Gunpowder's Train node loss.backward() call
        total_loss.backward = lambda: None

        logger.info(self.loss_dict)
        return total_loss

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


class Custom_Loss(torch.nn.Module):
    """Hyper-adjustable CycleGAN loss function based on Split-Loss"""
    def __init__(self, 
                netD1, 
                netG1, 
                netD2, 
                netG2, 
                optimizer_G1, 
                optimizer_G2, 
                optimizer_D, 
                dims,
                l1_loss = torch.nn.SmoothL1Loss(), 
                g_lambda_dict= {'A': {'l1_loss': {'cycled': 10, 'identity': 0},
                                    'gan_loss': {'fake': 1, 'cycled': 0},
                                    },
                                'B': {'l1_loss': {'cycled': 10, 'identity': 0},
                                    'gan_loss': {'fake': 1, 'cycled': 0},
                                    },
                            },
                d_lambda_dict= {'A': {'real': 1, 'fake': 1, 'cycled': 0},
                                'B': {'real': 1, 'fake': 1, 'cycled': 0},
                            },
                gan_mode='lsgan'
                 ):
        super(Custom_Loss, self).__init__()
        self.l1_loss = l1_loss
        self.gan_loss = GANLoss(gan_mode=gan_mode)
        self.netD1 = netD1 # differentiates between fake and real Bs
        self.netG1 = netG1 # turns As into Bs
        self.netD2 = netD2 # differentiates between fake and real As
        self.netG2 = netG2 # turns Bs into As
        self.optimizer_G1 = optimizer_G1
        self.optimizer_G2 = optimizer_G2
        self.optimizer_D = optimizer_D
        self.g_lambda_dict = g_lambda_dict
        self.d_lambda_dict = d_lambda_dict
        self.gan_mode = gan_mode
        self.dims = dims
        self.loss_dict = {}

    def crop(self, x, shape):
        '''Center-crop x to match spatial dimensions given by shape.'''

        x_target_size = x.size()[:-self.dims] + shape

        offset = tuple(
            (a - b)//2
            for a, b in zip(x.size(), x_target_size))

        slices = tuple(
            slice(o, o + s)
            for o, s in zip(offset, x_target_size))

        return x[slices]
                
    def clamp_weights(self, net, min=-0.01, max=0.01):
        for module in net.model:
            if hasattr(module, 'weight') and hasattr(module.weight, 'data'):
                temp = module.weight.data
                module.weight.data = temp.clamp(min, max)

    def backward_D(self, side, Dnet, data_dict):
        loss = 0
        for key, lambda_ in self.d_lambda_dict[side].items():
                if lambda_ != 0:
                    # if key == 'identity': # TODO: ADD IDENTITY SUPPORT
                    #     pred = Gnet(data_dict['real'])
                    # else:
                    #     pred = data_dict[key]
        
                    this_loss = self.gan_loss(Dnet(data_dict[key].detach()), key == 'real')
                    
                    self.loss_dict.update({f'Discriminator_{side}/{key}': this_loss})
                    loss += lambda_ * this_loss

        loss.backward()
        return loss

    def backward_Ds(self, data_dict, n_loop=5):
        self.set_requires_grad([self.netG1, self.netG2], False)  # G does not require gradients when optimizing D
        self.set_requires_grad([self.netD1, self.netD2], True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero

        if self.gan_mode.lower() == 'wgangp': # Wasserstein Loss
            for _ in range(n_loop):
                loss_D1 = self.backward_D('B', self.netD1, data_dict['B'])
                loss_D2 = self.backward_D('A', self.netD2, data_dict['A'])
                self.optimizer_D.step()          # update D's weights
                self.clamp_weights(self.netD1)
                self.clamp_weights(self.netD2)
        else:
            loss_D1 = self.backward_D('B', self.netD1, data_dict['B'])
            loss_D2 = self.backward_D('A', self.netD2, data_dict['A'])
            self.optimizer_D.step()          # update D's weights            
        
        self.set_requires_grad([self.netG1, self.netG2], True)  # Turn G gradients back on
        #return losses
        return loss_D1, loss_D2

    def backward_G(self, side, Gnet, Dnet, data_dict):
        """Calculate losses for the generator"""        
        
        loss = 0
        real = data_dict['real']
        for fcn_name, lambdas in self.g_lambda_dict[side].items():
            loss_fcn = getattr(self, fcn_name)
            for key, lambda_ in lambdas.items():
                if lambda_ != 0:
                    if key == 'identity' and key not in data_dict:
                        data_dict['identity'] = Gnet(data_dict['real'])
                    pred = data_dict[key]

                    if fcn_name == 'l1_loss':
                        if real.size()[-self.dims:] != pred.size()[-self.dims:]:
                            this_loss = loss_fcn(self.crop(real, pred.size()[-self.dims:]), pred)
                        else:
                            this_loss = loss_fcn(real, pred)
                    elif fcn_name == 'gan_loss':
                        this_loss = loss_fcn(Dnet(pred), True)
                    
                    self.loss_dict.update({f'{fcn_name}/{key}_{side}': this_loss})
                    loss += lambda_ * this_loss
        
        # calculate gradients
        loss.backward()
        return loss

    def backward_Gs(self, data_dict):
        self.set_requires_grad([self.netD1, self.netD2], False)  # D requires no gradients when optimizing G

        #G1 first
        self.set_requires_grad([self.netG1], True)  # G1 requires gradients when optimizing
        self.set_requires_grad([self.netG2], False)  # G2 requires no gradients when optimizing G1
        self.optimizer_G1.zero_grad()        # set G1's gradients to zero
        loss_G1 = self.backward_G('B', self.netG1, self.netD1, data_dict['B'])                   # calculate gradient for G
        self.optimizer_G1.step()             # udpate G1's weights

        #Then G2
        self.set_requires_grad([self.netG2], True)  # G2 requires gradients when optimizing
        self.set_requires_grad([self.netG1], False)  # G1 requires no gradients when optimizing G2
        self.optimizer_G2.zero_grad()        # set G2's gradients to zero
        loss_G2 = self.backward_G('A', self.netG2, self.netD2, data_dict['A'])                   # calculate gradient for G
        self.optimizer_G2.step()             # udpate G2's weights

        # Turn gradients back on
        self.set_requires_grad([self.netG1], True)
        #return losses
        return loss_G1, loss_G2

    def forward(self, real_A, fake_A, cycled_A, real_B, fake_B, cycled_B):
        
        # crop if necessary
        if real_A.size()[-self.dims:] != fake_B.size()[-self.dims:]:
            real_A = self.crop(real_A, fake_A.size()[-self.dims:])
            real_B = self.crop(real_B, fake_B.size()[-self.dims:])

        data_dict = {'A': {'real': real_A, 'fake': fake_A, 'cycled': cycled_A},
                     'B': {'real': real_B, 'fake': fake_B, 'cycled': cycled_B}
                    }
        # update Gs
        loss_G1, loss_G2 = self.backward_Gs(data_dict)
        
        # update Ds
        loss_D1, loss_D2 = self.backward_Ds(data_dict)        

        self.loss_dict.update({
            'Total_Loss/D1': float(loss_D1),
            'Total_Loss/D2': float(loss_D2),
            'Total_Loss/G1': float(loss_G1),
            'Total_Loss/G2': float(loss_G2),
        })

        total_loss = loss_G1 + loss_G2 #+ loss_D1 + loss_D2
        # define dummy backward pass to disable Gunpowder's Train node loss.backward() call
        total_loss.backward = lambda: None

        logger.info(self.loss_dict)
        return total_loss

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=False for all the networks to avoid unnecessary computations
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