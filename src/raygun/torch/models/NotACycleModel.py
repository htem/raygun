#%%
# from raygun.models import FreezableModel
import itertools
from raygun.utils import passing_locals
from raygun.torch.losses import GANLoss
from raygun.torch.networks.utils import init_weights
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam

# class NotACycleModel(FreezableModel):
class NotACycleModel(nn.Module):
    def __init__(self, perseverate=3, **kwargs):
        # output_arrays = ['fake_B', 'cycled_B', 'fake_A', 'cycled_A']
        self.output_arrays = ["fake_B", "cycled_B", "fake_A", "cycled_A"]
        # super().__init__(**passing_locals(locals()))
        super().__init__()

        # self.crop_pad = None #TODO: Determine if this is depracated

        self.A_encoder = nn.Sequential(
            nn.Conv3d(1, 16, 9, padding="same"),
            nn.ReLU(),
            nn.Conv3d(16, 16, 7, padding="same"),
            nn.ReLU(),
            nn.Conv3d(16, 32, 5, 2, padding=2),  # Downsample with preservation
            nn.ReLU(),
            nn.Conv3d(32, 32, 3, padding="same"),
            nn.ReLU(),
            nn.Conv3d(32, 32, 3, padding="same"),
            nn.ReLU(),
            nn.Conv3d(32, 64, 3, 2, padding=1),  # Downsample with preservation
            nn.ReLU(),
            nn.Conv3d(64, 64, 3, padding="same"),
            nn.ReLU(),
            nn.Conv3d(64, 64, 3, padding="same"),
            nn.ReLU(),
        )

        self.B_encoder = nn.Sequential(
            nn.Conv3d(1, 16, 9, padding="same"),
            nn.ReLU(),
            nn.Conv3d(16, 16, 7, padding="same"),
            nn.ReLU(),
            nn.Conv3d(16, 32, 5, 2, padding=2),  # Downsample with "preservation"
            nn.ReLU(),
            nn.Conv3d(32, 32, 3, padding="same"),
            nn.ReLU(),
            nn.Conv3d(32, 32, 3, padding="same"),
            nn.ReLU(),
            nn.Conv3d(32, 64, 3, 2, padding=1),  # Downsample with "preservation"
            nn.ReLU(),
            nn.Conv3d(64, 64, 3, padding="same"),
            nn.ReLU(),
            nn.Conv3d(64, 64, 3, padding="same"),
            nn.ReLU(),
        )

        self.latent = nn.Sequential(
            nn.Conv3d(64, 128, 5, padding="same"),
            nn.ReLU(),
            nn.Conv3d(128, 256, 3, padding="same"),
            nn.ReLU(),
            nn.Conv3d(256, 128, 3, padding="same"),
            nn.ReLU(),
            nn.Conv3d(128, 128, 3, padding="same"),
            nn.ReLU(),
        )

        self.A_decoder = nn.Sequential(
            nn.Conv3d(128, 128, 3, padding="same"),
            nn.ReLU(),
            nn.Conv3d(128, 64, 3, padding="same"),
            nn.ReLU(),
            nn.ConvTranspose3d(64, 32, 2, 2),  # Upsample with "preservation"
            nn.ReLU(),
            nn.Conv3d(32, 32, 3, padding="same"),
            nn.ReLU(),
            nn.Conv3d(32, 32, 3, padding="same"),
            nn.ReLU(),
            nn.ConvTranspose3d(32, 16, 2, 2),  # Upsample with "preservation"
            nn.ReLU(),
            nn.Conv3d(16, 16, 7, padding="same"),
            nn.ReLU(),
            nn.Conv3d(16, 1, 9, padding="same"),
        )

        self.B_decoder = nn.Sequential(
            nn.Conv3d(128, 128, 3, padding="same"),
            nn.ReLU(),
            nn.Conv3d(128, 64, 3, padding="same"),
            nn.ReLU(),
            nn.ConvTranspose3d(64, 32, 2, 2),  # Upsample with preservation
            nn.ReLU(),
            nn.Conv3d(32, 32, 3, padding="same"),
            nn.ReLU(),
            nn.Conv3d(32, 32, 3, padding="same"),
            nn.ReLU(),
            nn.ConvTranspose3d(32, 16, 2, 2),  # Upsample with preservation
            nn.ReLU(),
            nn.Conv3d(16, 16, 7, padding="same"),
            nn.ReLU(),
            nn.Conv3d(16, 1, 9, padding="same"),
        )

        self.critic = nn.Sequential(
            nn.Conv3d(128, 256, 2),
            nn.InstanceNorm3d(256),
            nn.LeakyReLU(),
            nn.Conv3d(256, 128, 2),
            nn.InstanceNorm3d(128),
            nn.LeakyReLU(),
            nn.Conv3d(128, 64, 2),
            nn.InstanceNorm3d(64),
            nn.LeakyReLU(),
            nn.Conv3d(64, 1, 2),
        )

        self.critic_loss = GANLoss("lsgan")

        self.encode_optimizer = Adam(
            itertools.chain(
                self.A_encoder.parameters(),
                self.B_encoder.parameters(),
                self.latent.parameters(),
            ),
            lr=1e-4,
        )
        self.critic_optimizer = Adam(self.critic.parameters(), lr=1e-4)

        self.perseverate = perseverate

        self.loss_dict = {}
        self.NaC_nets = [
            self.A_encoder,
            self.B_encoder,
            self.latent,
            self.A_decoder,
            self.B_decoder,
        ]
        for net in self.NaC_nets + [self.critic]:
            init_weights(net, "kaiming")

    # def set_crop_pad(self, crop_pad, ndims):
    # self.crop_pad = (slice(None,None,None),)*2 + (slice(crop_pad,-crop_pad),)*ndims

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

    def add_log(self, writer, step):
        # add loss values
        for key, loss in self.loss_dict.items():
            writer.add_scalar(key, loss, step)

    def forward(self, real_A=None, real_B=None):
        assert (
            real_A is not None or real_B is not None
        ), "Must have some real input to generate outputs)"

        if real_A is not None and real_B is not None:  # Assume training

            for n in range(self.perseverate):
                latent_A = self.latent(self.A_encoder(real_A))
                latent_B = self.latent(self.B_encoder(real_B))

                # Update Critic
                self.critic_optimizer.zero_grad()
                self.set_requires_grad(
                    [self.A_encoder, self.B_encoder, self.latent], False
                )
                self.set_requires_grad([self.critic], True)
                loss_a = self.critic_loss(self.critic(latent_A.detach()), True)
                loss_b = self.critic_loss(self.critic(latent_B.detach()), False)
                loss = loss_a + loss_b
                loss.backward()
                self.critic_optimizer.step()
                self.loss_dict["Model/Critic"] = loss.item()

                # Update Encoders
                self.encode_optimizer.zero_grad()
                self.set_requires_grad(
                    [self.A_encoder, self.B_encoder, self.latent], True
                )
                self.set_requires_grad([self.critic], False)
                loss_a = self.critic_loss(self.critic(latent_A), False)
                loss_b = self.critic_loss(self.critic(latent_B), True)
                loss = loss_a + loss_b
                loss.backward()
                self.encode_optimizer.step()
                self.loss_dict["Model/latent"] = loss.item()

                # Reactivate Critic gradients
                self.set_requires_grad([self.critic], True)

        if (
            real_A is not None
        ):  # allow calling for single direction pass (i.e. prediction)
            latent_A = self.latent(self.A_encoder(real_A))
            fake_B = self.B_decoder(latent_A)
            cycled_A = self.A_decoder(latent_A)
        else:
            fake_B = None
            cycled_A = None

        if real_B is not None:
            latent_B = self.latent(self.B_encoder(real_B))
            fake_A = self.A_decoder(latent_B)
            cycled_B = self.B_decoder(latent_B)
        else:
            fake_A = None
            cycled_B = None

        return fake_B, cycled_B, fake_A, cycled_A


# %%
