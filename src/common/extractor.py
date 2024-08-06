from typing import Dict

import torch
from torch import nn

from .network import get_network_class


class MMExtractor(nn.Module):
    """A multi-modal extractor for MineCraft.
    
    """
    def __init__(
        self,
        config: dict,
    ):
        super().__init__()
        self.fusion_kwargs = config.pop('fuse', {})
        self.critic_extra_info_kwargs = config.pop('critic_extra_info', {})
        
        self.extractors = nn.ModuleDict()
        for k, v in config.items():
            self.extractors[k] = get_network_class(v['extractor'])(**v['extractor_kwargs'])
        output_dim = sum([v.output_dim for v in self.extractors.values()])
        
        self.fusion_kwargs["extractor_kwargs"]["input_dim"] = output_dim
        self.fusion = get_network_class(
            self.fusion_kwargs["extractor"]
        )(**self.fusion_kwargs["extractor_kwargs"])
        self.output_dim = self.fusion.output_dim

        # extra information extractor for critic (e.g. target indicator)
        self.critic_extractor = nn.ModuleDict()
        for k, v in self.critic_extra_info_kwargs.items():
            self.critic_extractor[k] = get_network_class(v['extractor'])(**v['extractor_kwargs'])
        self.critic_extra_output_dim = sum([v.output_dim for v in self.critic_extractor.values()])
        
    def extract_features(self, obs: Dict[str, torch.Tensor]):
        ext_embeddings = []
        for k, v in obs.items():
            if k not in self.extractors:
                continue
            T, B, *shape = v.shape
            ext_embeddings.append(
                self.extractors[k](v.reshape(T * B, *shape)))
        output = self.fusion(torch.cat(ext_embeddings, dim=-1)).view(T, B, -1)
            
        critic_extra_ext_embeddings = []
        for k, v in obs.items():
            if k not in self.critic_extractor:
                continue
            T, B, *shape = v.shape
            critic_extra_ext_embeddings.append(
                self.critic_extractor[k](v.reshape(T * B, *shape)))
        if len(critic_extra_ext_embeddings) == 0:
            critic_extra_output = torch.tensor([]).to(output.device)
        else:
            critic_extra_output = torch.cat(critic_extra_ext_embeddings, dim=-1).view(T, B, -1)

        return output, critic_extra_output


if __name__ == "__main__":
    from src.config import load_config
    ext_fn = MMExtractor(load_config('src/config/feats_all.json'))
    print(ext_fn)
