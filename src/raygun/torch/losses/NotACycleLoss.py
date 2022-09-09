#%%
import torch
from raygun.torch.losses import BaseCompetentLoss
from raygun.utils import passing_locals

import logging
logger = logging.Logger(__name__, 'INFO')

class NotACycleLoss(BaseCompetentLoss):
    """NotACycleGAN loss function"""
    def __init__(self, 
                netD,  # differentiates between which of A or B inputs are fake
                NaC_nets,  # Encoders, latent, and decoders
                optimizer_G, 
                optimizer_D, 
                dims,
                l1_loss = torch.nn.SmoothL1Loss(), 
                g_lambda_dict= {'A': {'l1_loss': 3, #cycled
                                    'gan_loss': 1, #fake
                                    },
                                'B': {'l1_loss': 3,
                                    'gan_loss': 1,
                                    },
                            },
                d_lambda_dict= {'A': 1,
                                'B': 1,
                            },
                gan_mode='lsgan',
                **kwargs):        
        super().__init__(**passing_locals(locals()))

    def _backward_D(self, data_dict):
        
        A_test = torch.cat([data_dict['A']['real'], data_dict['B']['fake'].detach()], 1)
        loss_A = self.d_lambda_dict['A'] * self.gan_loss(self.netD(A_test), True)
        
        B_test = torch.cat([data_dict['B']['real'], data_dict['A']['fake'].detach()], 1)
        loss_B = self.d_lambda_dict['B'] * self.gan_loss(self.netD(B_test), False)

        loss = loss_A + loss_B
        loss.backward()
        self.optimizer_D.step()          # update D's weights

        return loss_A, loss_B

    def backward_D(self, data_dict, n_loop=5):
        """Calculate losses for discriminator"""        
        self.set_requires_grad(self.NaC_nets, False)  # G does not require gradients when optimizing D
        self.set_requires_grad([self.netD], True)  # D does require gradients when optimizing D

        self.optimizer_D.zero_grad(set_to_none=True)     # set D's gradients to zero

        if self.gan_mode.lower() == 'wgangp': # Wasserstein Loss
            for _ in range(n_loop):
                loss_A, loss_B = self._backward_D(data_dict)
                self.clamp_weights(self.netD1)
                self.clamp_weights(self.netD2)
        else:
            loss_A, loss_B = self._backward_D(data_dict)

        self.set_requires_grad(self.NaC_nets, True)  # Turn G gradients back on
        
        self.loss_dict.update({ 
            'Discriminator/A': float(loss_A),
            'Discriminator/B': float(loss_B),
        })

        #return losses
        return loss_A + loss_B

    def backward_G(self, data_dict):
        """Calculate losses for generators"""        
        self.set_requires_grad([self.netD], False)  # D requires no gradients when optimizing G
        self.set_requires_grad(self.NaC_nets, True)  # G does not require gradients when optimizing D

        self.optimizer_G.zero_grad(set_to_none=True)        # set G1's gradients to zero
        losses = {}

        #First L1 losses
        cyc_size = data_dict['A']['cycled'].size()[-self.dims:]
        if data_dict['A']['real'].size()[-self.dims:] != cyc_size:
            losses['A', 'l1_loss'] = self.l1_loss(self.crop(data_dict['A']['real'], cyc_size), data_dict['A']['cycled'])
            losses['B', 'l1_loss'] = self.l1_loss(self.crop(data_dict['B']['real'], cyc_size), data_dict['B']['cycled'])
        else:            
            losses['A', 'l1_loss'] = self.l1_loss(data_dict['A']['real'], data_dict['A']['cycled'])
            losses['B', 'l1_loss'] = self.l1_loss(data_dict['B']['real'], data_dict['B']['cycled'])

        #Then Discriminator losses        
        A_test = torch.cat([data_dict['A']['real'], data_dict['B']['fake'].detach()], 1)
        losses['A', 'gan_loss'] = self.gan_loss(self.netD(A_test), False)
        
        B_test = torch.cat([data_dict['B']['real'], data_dict['A']['fake'].detach()], 1)
        losses['B', 'gan_loss'] = self.gan_loss(self.netD(B_test), True)
        
        #Sum and add to loss dictionary for logging
        loss = 0
        for (side, fcn_name), this_loss in losses.items():
            self.loss_dict.update({f'{fcn_name}/{side}': this_loss})
            loss += self.g_lambda_dict[side][fcn_name] * this_loss
        
        # calculate gradients
        # loss.backward(retain_graph=True)
        loss.backward()
        self.optimizer_G.step()             # udpate G1's weights

        self.set_requires_grad([self.netD], True)  # re-enable backprop for D        
        return loss

    def forward(self, real_A, fake_A, cycled_A, real_B, fake_B, cycled_B):
        self.data_dict = {'real_A': real_A, 'fake_A': fake_A, 'cycled_A': cycled_A, 'real_B': real_B, 'fake_B': fake_B, 'cycled_B': cycled_B}
        
        # crop if necessary
        if real_A.size()[-self.dims:] != fake_B.size()[-self.dims:]:
            real_A = self.crop(real_A, fake_A.size()[-self.dims:])
            real_B = self.crop(real_B, fake_B.size()[-self.dims:])

        data_dict = {'A': {'real': real_A, 'fake': fake_A, 'cycled': cycled_A},
                     'B': {'real': real_B, 'fake': fake_B, 'cycled': cycled_B}
                    }

        # update D
        loss_D = self.backward_D(data_dict)

        # update G
        loss_G = self.backward_G(data_dict)        

        self.loss_dict.update({ 
            'Total_Loss/D': float(loss_D),
            'Total_Loss/G': float(loss_G),
        })

        total_loss = loss_D + loss_G
        # define dummy backward pass to disable Gunpowder's Train node loss.backward() call
        total_loss.backward = lambda: None

        logger.info(self.loss_dict)
        return total_loss
# %%
