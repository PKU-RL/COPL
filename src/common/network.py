from typing import Optional, Tuple, Type

import numpy as np
import torch
from more_itertools import pairwise
from torch import nn

mapping = {}


def register(name: str):
    def _thunk(network_class: Type[nn.Module]):
        mapping[name] = network_class
        return network_class

    return _thunk


def get_network_class(name: str):
    if name in mapping:
        return mapping[name]
    else:
        raise ValueError("Unknown network class: {}".format(name))
    

@torch.no_grad()
def infer_fc_input_dim(module, input_shape):
    dummy_input = torch.rand(2, *input_shape)
    dummy_output = module(dummy_input)
    infer_dim = torch.flatten(dummy_output, start_dim=1).shape[-1]
    return infer_dim


@register("mlp")
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hiddens: Tuple[int, ...] = (),
        activation: Type[nn.Module] = nn.ReLU,
        final_activation: Type[nn.Module] = nn.Identity,
    ):
        super().__init__()
        hiddens = (input_dim, *hiddens)
        n_layers = len(hiddens) - 1
        # layers = [nn.Flatten()]
        layers = []
        if n_layers > 0:
            for i, (n_in, n_out) in enumerate(pairwise(hiddens), start=1):
                layers.append(nn.Linear(in_features=n_in, out_features=n_out))
                layers.append(activation() if i < n_layers else final_activation())

        self.fc = nn.Sequential(*layers)
        self.output_dim = hiddens[-1]
        self.is_recurrent = False

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x


@register("emb_mlp")
class EMBMLP(nn.Module):
    def __init__(
        self,
        input_n: int,
        num_emb: int,
        emb_dim: int,
        padding_idx: Optional[int] = None,
        hiddens: Tuple[int, ...] = (),
        activation: Type[nn.Module] = nn.ReLU,
        final_activation: Type[nn.Module] = nn.Identity,
    ):
        super().__init__()
        self.embed = nn.Embedding(num_emb, emb_dim, padding_idx=padding_idx)
        self.mlp = MLP(
            input_dim=input_n * emb_dim,
            hiddens=hiddens,
            activation=activation,
            final_activation=final_activation,
        )
        self.output_dim = self.mlp.output_dim
        
    def forward(self, x):
        x = self.embed(x.long())
        x = x.view(x.shape[:-2] + (-1,))
        x = self.mlp(x)
        return x


@register("cnn")
class CNN(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        conv_kwargs: Tuple[dict, ...],
        activation: Type[nn.Module] = nn.ReLU,
        hiddens: Optional[Tuple[int, ...]] = (256,),
        final_activation: Type[nn.Module] = nn.Identity,
    ):
        super().__init__()
        c, h, w = input_shape

        # Build conv layers
        n_channels = (c, *(kwargs["out_channels"] for kwargs in conv_kwargs))
        convs = []
        for i, n_in in enumerate(n_channels[:-1]):
            convs.append(nn.Conv2d(in_channels=n_in, **conv_kwargs[i]))
            convs.append(activation())
        convs.append(nn.Flatten())
        self.convs = nn.Sequential(*convs)
        output_dim = infer_fc_input_dim(self.convs, input_shape)
        self.output_dim = output_dim

        # Build fully connected layers
        self.hiddens = hiddens
        if hiddens is not None:
            self.mlp = MLP(
                input_dim=output_dim,
                hiddens=hiddens,
                activation=activation,
                final_activation=final_activation,
            )
            self.output_dim = self.mlp.output_dim
        self.is_recurrent = False

    def forward(self, x):
        x = self.convs(x)
        if self.hiddens is not None:
            x = self.mlp(x)
        return x


@register("rnn")
class RNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        rnn_type: Type[nn.RNNBase] = nn.LSTM,
        rnn_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        self.recurrent_layers = rnn_type(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            **(rnn_kwargs or {}),
        )
        self.output_dim = hidden_size
        self.is_recurrent = True

    def forward(self, x, states=None):
        x, states = self.recurrent_layers(x, states)
        return x, states


@register("oh_trunc")
class OneHotTrunc(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = x[..., : self.input_dim]
        x = self.fc(x)
        return x
