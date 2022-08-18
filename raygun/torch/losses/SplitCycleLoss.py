
import torch
from raygun.torch.losses import BaseCompetentLoss
from raygun.utils import passing_locals

import logging
logger = logging.Logger(__name__, 'INFO')

class SplitCycleLoss(BaseCompetentLoss):
    """CycleGAN loss function"""
    def __init__(self, 
                netD1,  # differentiates between fake and real Bs
                netG1,  # turns As into Bs
                netD2,  # differentiates between fake and real As
                netG2,  # turns Bs into As
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
                gan_mode='lsgan',
                **kwargs):        
        super().__init__(**passing_locals(locals()))

    def backward_D(self, side, dnet, data_dict):
        """Calculate losses for a discriminator"""        
        loss = 0
        for key, lambda_ in self.d_lambda_dict[side].items():
            if lambda_ != 0:
                # if key == 'identity': # TODO: ADD IDENTITY SUPPORT
                #     pred = gnet(data_dict['real'])
                # else:
                #     pred = data_dict[key]

                this_loss = self.gan_loss(dnet(data_dict[key].detach()), key == 'real')
                
                self.loss_dict.update({f'Discriminator_{side}/{key}': this_loss})
                loss += lambda_ * this_loss

        loss.backward()
        return loss

    def backward_Ds(self, data_dict, n_loop=5):
        self.set_requires_grad([self.netG1, self.netG2], False)  # G does not require gradients when optimizing D
        self.optimizer_D.zero_grad(set_to_none=True)     # set D's gradients to zero

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

    def backward_G(self, side, gnet, dnet, data_dict):
        """Calculate losses for a generator"""        
        loss = 0
        real = data_dict['real']
        for fcn_name, lambdas in self.g_lambda_dict[side].items():
            loss_fcn = getattr(self, fcn_name)
            for key, lambda_ in lambdas.items():
                if lambda_ != 0:
                    if key == 'identity' and key not in data_dict:
                        data_dict['identity'] = gnet(real)
                    pred = data_dict[key]

                    if fcn_name == 'l1_loss':
                        if real.size()[-self.dims:] != pred.size()[-self.dims:]:
                            this_loss = loss_fcn(self.crop(real, pred.size()[-self.dims:]), pred)
                        else:
                            this_loss = loss_fcn(real, pred)
                    elif fcn_name == 'gan_loss':
                        this_loss = loss_fcn(dnet(pred), True)
                    
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
        self.optimizer_G1.zero_grad(set_to_none=True)        # set G1's gradients to zero
        loss_G1 = self.backward_G('B', self.netG1, self.netD1, data_dict['B'])                   # calculate gradient for G
        self.optimizer_G1.step()             # udpate G1's weights

        #Then G2
        self.set_requires_grad([self.netG2], True)  # G2 requires gradients when optimizing
        self.set_requires_grad([self.netG1], False)  # G1 requires no gradients when optimizing G2
        self.optimizer_G2.zero_grad(set_to_none=True)        # set G2's gradients to zero
        loss_G2 = self.backward_G('A', self.netG2, self.netD2, data_dict['A'])                   # calculate gradient for G
        self.optimizer_G2.step()             # udpate G2's weights

        # Turn gradients back on
        self.set_requires_grad([self.netG1, self.netD1, self.netD2], True)
        
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

        total_loss = loss_G1 + loss_G2 + loss_D1 + loss_D2
        # define dummy backward pass to disable Gunpowder's Train node loss.backward() call
        total_loss.backward = lambda: None

        logger.info(self.loss_dict)
        return total_loss