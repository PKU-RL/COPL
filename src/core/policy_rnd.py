import torch
from torch import nn

from ..common.network import CNN
from ..segment.segmineclip import preprocess
from .policy import PPOMultiDiscretePolicy


class RNDPPOPolicy(PPOMultiDiscretePolicy):
    def __init__(
        self, 
        *args, 
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.predictor_net = CNN(
            input_shape = (3, 160, 256),
            conv_kwargs = (
                dict(kernel_size=7, stride=4, padding=0, out_channels=64),
                dict(kernel_size=5, stride=2, padding=0, out_channels=128),
                dict(kernel_size=3, stride=1, padding=0, out_channels=128),
                dict(kernel_size=3, stride=1, padding=0, out_channels=128),
            ),
            hiddens = (512, 512)
        )

    def loss(self, obs, *args, **kwargs):
        pg_loss, vf_loss, entropy, extra_out = super().loss(obs, *args, **kwargs)
        
        obs_rgb = preprocess(obs["obs_rgb"]).to(self.device).float()
        obs_emb = torch.as_tensor(obs["obs_emb"]).to(self.device).float()
        
        T, B = obs_rgb.shape[:2]
        pred_obs_emb = self.predictor_net(obs_rgb.reshape(T * B, *obs_rgb.shape[2:])).reshape(T, B, -1)
        rnd_loss = torch.norm(obs_emb - pred_obs_emb, dim=-1, p=2).mean()
        return pg_loss, vf_loss, entropy, rnd_loss, extra_out