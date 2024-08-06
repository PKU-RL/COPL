from collections import OrderedDict
from typing import Any, Dict, Tuple, Type, Union

import torch
from torch import nn

from ..config import load_config
from .extractor import MMExtractor
from .network import MLP, RNN


class ParamsMixin(nn.Module):
    def __init__(self, device: Union[str, torch.device] = "cpu") -> None:
        super().__init__()
        self.device = device

    def get_params(self) -> OrderedDict:
        params = OrderedDict(
            {name: weight.detach().cpu() for name, weight in self.state_dict().items()}
        )
        return params

    def set_params(self, params: OrderedDict) -> None:
        self.load_state_dict(params)
        self.to(self.device)


class ActorHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_outputs: int,
        hiddens: Tuple[int, ...] = (),
        activation: Type[nn.Module] = nn.ReLU,
        squash_output: bool = False,
    ) -> None:
        super().__init__()
        self.pi = MLP(
            input_dim=input_dim,
            hiddens=(*hiddens, n_outputs),
            activation=activation,
            final_activation=nn.Tanh if squash_output else nn.Identity,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        T, B, _ = features.shape
        return self.pi(features.view(T * B, -1)).view(T, B, -1)


class VCriticHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hiddens: Tuple[int, ...] = (),
        activation: Type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        self.vf = MLP(input_dim=input_dim, hiddens=(*hiddens, 1), activation=activation)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        T, B, _ = features.shape
        return self.vf(features.view(T * B, -1)).view(T, B, -1)


class Actor(nn.Module):
    def __init__(
        self,
        n_outputs: int,
        extractor_config: Union[str, dict],
        use_rnn: bool = False,
        rnn_hidden: int = 256,
        hiddens: Tuple[int, ...] = (),
        activation: Type[nn.Module] = nn.ReLU,
        squash_output: bool = False,
    ) -> None:
        super().__init__()
        if isinstance(extractor_config, str):
            extractor_config = load_config(extractor_config)
        self.extractor = MMExtractor(extractor_config)
        output_dim = self.extractor.output_dim
        
        self._use_rnn = use_rnn
        if self._use_rnn:
            self.rnn = RNN(
                input_size=output_dim,
                hidden_size=rnn_hidden,
                num_layers=1,
                rnn_type=nn.GRU,
            )
            output_dim = rnn_hidden
        
        self.actor_head = ActorHead(
            input_dim=output_dim,
            n_outputs=n_outputs,
            hiddens=hiddens,
            activation=activation,
            squash_output=squash_output,
        )

    def forward(
        self, 
        obs: Union[torch.Tensor, Dict[str, torch.Tensor]], 
        first: torch.Tensor, 
        states=None
    ):
        features, _ = self.extractor.extract_features(obs)

        if self._use_rnn:
            # unroll RNN
            features_lst = []
            masks = ~first
            for feat, mask in zip(features.unbind(), masks.unbind()):
                if states is not None:
                    # if isinstance(states, tuple):
                    #     states = tuple(mask[..., None] * s for s in states)
                    # else:
                    #     states = mask[..., None] * states
                    states = mask[..., None] * states
                feat, states = self.rnn(feat[None, ...], states)
                features_lst.append(feat)
            features = torch.cat(features_lst, dim=0)
        
        pi = self.actor_head(features)
        return pi, states


class VCritic(nn.Module):
    def __init__(
        self,
        extractor_config: Union[str, dict],
        use_rnn: bool = False,
        rnn_hidden: int = 256,
        hiddens: Tuple[int, ...] = (),
        activation: Type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        if isinstance(extractor_config, str):
            extractor_config = load_config(extractor_config)
        self.extractor = MMExtractor(extractor_config)
        output_dim = self.extractor.output_dim
        critic_extra_output_dim = self.extractor.critic_extra_output_dim
        
        self._use_rnn = use_rnn
        if self._use_rnn:
            self.rnn = RNN(
                input_size=output_dim,
                hidden_size=rnn_hidden,
                num_layers=1,
                rnn_type=nn.GRU,
            )
            output_dim = rnn_hidden
        
        self.vcritic_head = VCriticHead(
            input_dim=output_dim + critic_extra_output_dim,
            hiddens=hiddens, 
            activation=activation
        )

    def forward(
        self, 
        obs: Union[torch.Tensor, Dict[str, torch.Tensor]], 
        first: torch.Tensor, 
        states=None
    ):
        features, extra_features = self.extractor.extract_features(obs)

        if self._use_rnn:
            # unroll RNN
            features_lst = []
            masks = ~first
            for feat, mask in zip(features.unbind(), masks.unbind()):
                if states is not None:
                    # if isinstance(states, tuple):
                    #     states = tuple(mask[..., None] * s for s in states)
                    # else:
                    #     states = mask[..., None] * states
                    states = mask[..., None] * states
                feat, states = self.rnn(feat[None, ...], states)
                features_lst.append(feat)
            features = torch.cat(features_lst, dim=0)
        
        value = self.vcritic_head(torch.cat((features, extra_features), dim=-1))
        return value, states


class ActorVCritic(nn.Module):
    def __init__(
        self,
        n_outputs: int,
        extractor_config: Union[str, dict],
        use_rnn: bool = False,
        rnn_hidden: int = 256,
        actor_hiddens: Tuple[int, ...] = (),
        critic_hiddens: Tuple[int, ...] = (),
        activation: Type[nn.Module] = nn.ReLU,
        squash_output: bool = False,
    ) -> None:
        super().__init__()
        if isinstance(extractor_config, str):
            extractor_config = load_config(extractor_config)
        self.extractor = MMExtractor(extractor_config)
        output_dim = self.extractor.output_dim
        critic_extra_output_dim = self.extractor.critic_extra_output_dim
        
        self._use_rnn = use_rnn
        if self._use_rnn:
            self.rnn = RNN(
                input_size=output_dim,
                hidden_size=rnn_hidden,
                num_layers=1,
                rnn_type=nn.GRU,
            )
            output_dim = rnn_hidden
        
        self.actor_head = ActorHead(
            input_dim=output_dim,
            n_outputs=n_outputs,
            hiddens=actor_hiddens,
            activation=activation,
            squash_output=squash_output,
        )
        self.vcritic_head = VCriticHead(
            input_dim=output_dim + critic_extra_output_dim,
            hiddens=critic_hiddens,
            activation=activation,
        )

    def forward(
        self, 
        obs: Union[torch.Tensor, Dict[str, torch.Tensor]], 
        first: torch.Tensor, 
        states=None
    ):
        features, extra_features = self.extractor.extract_features(obs)

        if self._use_rnn:
            # unroll RNN
            features_lst = []
            masks = ~first
            for feat, mask in zip(features.unbind(), masks.unbind()):
                if states is not None:
                    # if isinstance(states, tuple):
                    #     states = tuple(mask[..., None] * s for s in states)
                    # else:
                    #     states = mask[..., None] * states
                    states = mask[..., None] * states
                feat, states = self.rnn(feat[None, ...], states)
                features_lst.append(feat)
            features = torch.cat(features_lst, dim=0)
        
        pi = self.actor_head(features)
        value = self.vcritic_head(torch.cat((features, extra_features), dim=-1))
        return pi, value, states

    def forward_actor(
        self, 
        obs: Union[torch.Tensor, Dict[str, torch.Tensor]], 
        first: torch.Tensor, 
        states=None
    ):
        features, _ = self.extractor.extract_features(obs)

        if self._use_rnn:
            # unroll RNN
            features_lst = []
            masks = ~first
            for feat, mask in zip(features.unbind(), masks.unbind()):
                if states is not None:
                    # if isinstance(states, tuple):
                    #     states = tuple(mask[..., None] * s for s in states)
                    # else:
                    #     states = mask[..., None] * states
                    states = mask[..., None] * states
                feat, states = self.rnn(feat[None, ...], states)
                features_lst.append(feat)
            features = torch.cat(features_lst, dim=0)
        
        pi = self.actor_head(features)
        return pi, states

    def forward_critic(
        self, 
        obs: Union[torch.Tensor, Dict[str, torch.Tensor]], 
        first: torch.Tensor, 
        states=None
    ):
        features, extra_features = self.extractor.extract_features(obs)

        if self._use_rnn:
            # unroll RNN
            features_lst = []
            masks = ~first
            for feat, mask in zip(features.unbind(), masks.unbind()):
                if states is not None:
                    # if isinstance(states, tuple):
                    #     states = tuple(mask[..., None] * s for s in states)
                    # else:
                    #     states = mask[..., None] * states
                    states = mask[..., None] * states
                feat, states = self.rnn(feat[None, ...], states)
                features_lst.append(feat)
            features = torch.cat(features_lst, dim=0)
        
        value = self.vcritic_head(torch.cat((features, extra_features), dim=-1))
        return value, states


class ActorDualVCritic(nn.Module):
    def __init__(
        self,
        n_outputs: int,
        extractor_config: Union[str, dict],
        use_rnn: bool = False,
        rnn_hidden: int = 256,
        actor_hiddens: Tuple[int, ...] = (),
        critic_hiddens: Tuple[int, ...] = (),
        activation: Type[nn.Module] = nn.ReLU,
        squash_output: bool = False,
    ) -> None:
        super().__init__()
        if isinstance(extractor_config, str):
            extractor_config = load_config(extractor_config)
        self.extractor = MMExtractor(extractor_config)
        output_dim = self.extractor.output_dim
        
        self._use_rnn = use_rnn
        if self._use_rnn:
            self.rnn = RNN(
                input_size=output_dim,
                hidden_size=rnn_hidden,
                num_layers=1,
                rnn_type=nn.GRU,
            )
            output_dim = rnn_hidden
        
        self.actor_head = ActorHead(
            input_dim=output_dim,
            n_outputs=n_outputs,
            hiddens=actor_hiddens,
            activation=activation,
            squash_output=squash_output,
        )
        # critic head for episodic extrinsic reward
        self.ext_vcritic_head = VCriticHead(
            input_dim=output_dim,
            hiddens=critic_hiddens,
            activation=activation,
        )
        # critic hrad for non-episodic intrinsic reward
        self.int_vcritic_head = VCriticHead(
            input_dim=output_dim,
            hiddens=critic_hiddens,
            activation=activation,
        )
    
    def forward(
        self, 
        obs: Union[torch.Tensor, Dict[str, torch.Tensor]], 
        first: torch.Tensor, 
        states: Any = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        features = self.extractor.extract_features(obs)

        if self._use_rnn:
            # unroll RNN
            features_lst = []
            masks = ~first
            for feat, mask in zip(features.unbind(), masks.unbind()):
                if states is not None:
                    # if isinstance(states, tuple):
                    #     states = tuple(mask[..., None] * s for s in states)
                    # else:
                    #     states = mask[..., None] * states
                    states = mask[..., None] * states
                feat, states = self.rnn(feat[None, ...], states)
                features_lst.append(feat)
            features = torch.cat(features_lst, dim=0)
        
        pi = self.actor_head(features)
        vf_ext = self.ext_vcritic_head(features)
        vf_int = self.int_vcritic_head(features)
        return pi, vf_ext, vf_int, states
    
    def forward_actor(
        self, 
        obs: Union[torch.Tensor, Dict[str, torch.Tensor]], 
        first: torch.Tensor, 
        states: Any = None,
    ) -> Tuple[torch.Tensor, Any]:
        features = self.extractor.extract_features(obs)

        if self._use_rnn:
            # unroll RNN
            features_lst = []
            masks = ~first
            for feat, mask in zip(features.unbind(), masks.unbind()):
                if states is not None:
                    # if isinstance(states, tuple):
                    #     states = tuple(mask[..., None] * s for s in states)
                    # else:
                    #     states = mask[..., None] * states
                    states = mask[..., None] * states
                feat, states = self.rnn(feat[None, ...], states)
                features_lst.append(feat)
            features = torch.cat(features_lst, dim=0)
        
        pi = self.actor_head(features)
        return pi, states

    def forward_critic(
        self, 
        obs: Union[torch.Tensor, Dict[str, torch.Tensor]], 
        first: torch.Tensor, 
        states: Any = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Any]:
        features = self.extractor.extract_features(obs)

        if self._use_rnn:
            # unroll RNN
            features_lst = []
            masks = ~first
            for feat, mask in zip(features.unbind(), masks.unbind()):
                if states is not None:
                    # if isinstance(states, tuple):
                    #     states = tuple(mask[..., None] * s for s in states)
                    # else:
                    #     states = mask[..., None] * states
                    states = mask[..., None] * states
                feat, states = self.rnn(feat[None, ...], states)
                features_lst.append(feat)
            features = torch.cat(features_lst, dim=0)
        
        vf_ext = self.ext_vcritic_head(features)
        vf_int = self.int_vcritic_head(features)
        return vf_ext, vf_int, states
