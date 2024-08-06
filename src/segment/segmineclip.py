import os
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..others.mineclip_official.modules import (GPT, AdapterHead,
                                                TemporalTransformer)
from ..utils.utils import get_yaml_data
from .modules import VisionTransformer
from .utils import get_similarity_map, prompt_denoising


class SegVideoCLIP(nn.Module):
    def __init__(
        self,
        # modules
        image_encoder: VisionTransformer, #| VisionTransformer_after_freeze,
        text_encoder: GPT ,#| GPT_after_freeze,
        temporal_encoder: TemporalTransformer,
        reward_adapter: AdapterHead,
        logit_scale: nn.Parameter,
        # parameters for segmentation
        prompt_denoise: float = 0.0,
        dcrf: bool = False,
        dcrf_kwargs: dict = None,
    ) -> None:
        super().__init__()

        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.temporal_encoder = temporal_encoder
        self.reward_adapter = reward_adapter
        self.logit_scale = logit_scale

        self.prompt_denoise = prompt_denoise
        
        self.dcrf = dcrf
        self.dcrf_kwargs = dcrf_kwargs

    def encode_video(
        self,
        videos: torch.Tensor,
    ) -> torch.Tensor:
        assert len(videos.shape) == 5   # BTCHW
        image_embeddings = self.image_encoder(videos)[0]
        video_embeddings = self.temporal_encoder(image_embeddings)
        return video_embeddings
    
    def encode_image(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        assert len(images.shape) == 4   # BCHW
        image_embeddings = self.image_encoder(images.unsqueeze(dim=1))[0]
        return image_embeddings.squeeze(dim=1)

    def encode_and_segment_image(
        self, 
        images: torch.Tensor, 
        target: Union[int, List[int]],
        text_embeddings: torch.Tensor,
        temporal_encode: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert len(images.shape) == 4   # BCHW
        image_embeddings, patch_embeddings, q, k = self.image_encoder(images.unsqueeze(dim=1))
        if temporal_encode:
            b, t, h, w, c = patch_embeddings.shape
            patch_embeddings = patch_embeddings.permute(0, 2, 3, 1, 4).reshape(b * h * w, t, c)
            patch_embeddings = self.temporal_encoder(patch_embeddings).reshape(b, h, w, -1)
        else:
            patch_embeddings = patch_embeddings[:, 0]

        patch_embeddings /= patch_embeddings.norm(dim=-1, keepdim=True)
        patch_logits = patch_embeddings @ text_embeddings.T
        if self.prompt_denoise > 0.0:
            patch_logits = prompt_denoising(patch_logits, self.logit_scale, self.prompt_denoise)
        patch_probs = F.softmax(patch_logits * self.logit_scale.exp(), dim=-1)
        
        if isinstance(target, int):
            target = [target] * images.shape[0]
        target = torch.as_tensor(target, device=images.device)[:, None, None, None]
        target = target.expand(-1, patch_logits.shape[1], patch_logits.shape[2], -1)
        if not self.dcrf:
            similarity_map = get_similarity_map(patch_logits, images.shape[2: 4], norm=False)
            similarity_map = similarity_map.cpu().numpy()
        else:
            raise NotImplementedError
            similarity_map = get_similarity_map(patch_logits, images.shape[2: 4], norm="bilinear")
            similarity_map = similarity_map.cpu().numpy()
            # Optional: DenseCRF
            similarity_map = densecrf(similarity_map, images.cpu().numpy(), len(text_embeddings), not self.key_smooth, **self.dcrf_kwargs)

        image_embeddings = image_embeddings.squeeze(dim=1)
        patch_logits_target = torch.gather(patch_logits, dim=-1, index=target)[..., 0]
        patch_probs_target = torch.gather(patch_probs, dim=-1, index=target)[..., 0]

        patch_labels = torch.argmax(patch_logits, dim=-1)
        patch_masks_target = patch_labels == target[..., 0]

        return image_embeddings, patch_logits_target, patch_probs_target, patch_labels, patch_masks_target

    def encode_text(self, text):
        text_embeddings = self.text_encoder(text)
        return text_embeddings
    
    def forward(self, videos, texts):
        videos = torch.as_tensor(videos,dtype=torch.float)
        images_embeddings, _, _, _ = self.image_encoder(videos)
        text_embeddings = self.text_encoder(texts)
        video_embeddings = self.temporal_encoder(images_embeddings)
        adapted_video, adapted_text = self.reward_adapter(video_embeddings, text_embeddings)
        
        video_features = adapted_video / adapted_video.norm(dim=1, keepdim=True)
        text_features = adapted_text / adapted_text.norm(dim=1, keepdim=True)
        
        return self.logit_scale.exp() * video_features, text_features
    
    def get_layer(self,layer_idx):
        if layer_idx < 0:
            raise RuntimeError
        if layer_idx == 0:
            return self.reward_adapter,self.logit_scale
        elif layer_idx == 1:
            return (self.temporal_encoder,)
        elif layer_idx == 2:
            return self.image_encoder.ln_post,self.image_encoder.projection,self.text_encoder.ln_final,self.text_encoder.projection,
        elif (layer_idx-3) < self.image_encoder._layers:
            return self.image_encoder.blocks[-(layer_idx-2)],self.text_encoder.blocks[-(layer_idx-2)]
        else:
            return self.image_encoder.pos_embed,self.image_encoder.cls_token,self.image_encoder.conv1,self.image_encoder.ln_pre,\
                   self.text_encoder.pos_embed,self.text_encoder.token_embedding

    @torch.no_grad()
    def clamp_logit_scale(self, value=100):
        """
        Follow OpenAI CLIP paper's trick to prevent training instability (sec 2.5)
        """
        self.logit_scale.data.clamp_(-np.log(value), np.log(value))


def build_pretrain_model(
    image_config: dict,
    text_config: dict,
    temporal_config: dict,
    adapter_config: dict,
    state_dict: Optional[dict] = None,
    **kwargs: Any,
) -> SegVideoCLIP:
    image_encoder = VisionTransformer(
        resolution=image_config['resolution'],
        patch_size=image_config['patch_size'],
        width=image_config['width'],
        layers=image_config['layers'],
        heads=image_config['heads'],
        output_dim=image_config['output_dim']
    )
    
    text_encoder = GPT(
        embed_dim=text_config['embed_dim'],
        context_length=text_config['context_length'],
        vocab_size=text_config['vocab_size'],
        layers=text_config['layers'],
        width=text_config['width'],
        heads=text_config['heads']
    )
    
    temporal_encoder = TemporalTransformer(
        input_dim=temporal_config['input_dim'],
        depth=temporal_config['depth'],
        num_heads=temporal_config['num_heads'],
        max_seq_len=temporal_config['video_seq_len'],
        ff_glu=True,
        ff_swish=True
    )
    
    reward_adapter = AdapterHead(
        adapter_config['video_layers'],
        adapter_config['text_layers'],
        adapter_config['feature_dim']
    )

    logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    model = SegVideoCLIP(
        image_encoder,
        text_encoder, 
        temporal_encoder, 
        reward_adapter, 
        logit_scale,
        **kwargs,
    )  
    
    if not state_dict is None :
        state_dict_back = model.state_dict()
        state_dict_back.update(state_dict)
        model.load_state_dict(state_dict_back)
    
    return model


MEAN = torch.tensor([0.3331, 0.3245, 0.3051])
STD = torch.tensor([0.2439, 0.2493, 0.2873])

def preprocess(img: np.ndarray) -> torch.Tensor:
    # if img.dtype == np.uint8:
    if "int" in str(img.dtype):
        img = img.astype(np.float32) / 255.0
    img = torch.as_tensor(img)
    img.sub_(MEAN[:, None, None]).div_(STD[:, None, None])
    return img


def load(
    path: Optional[str] = None,
    device: Union[str, torch.device] = "cuda",
    **kwargs: Any,
) -> Tuple[SegVideoCLIP, Callable[[np.ndarray], torch.Tensor]]:
    if path is None:
        conf_path = "src/others/mineclip_official/config.yml"
        model_path = "src/others/mineclip_official/adjust.pth"
    else:
        conf_path = os.path.join(path, "config.yml")
        model_path = os.path.join(path, "adjust.pth")
    
    clip_config = get_yaml_data(conf_path)
    model = build_pretrain_model(
        image_config = clip_config['image_config'],
        text_config = clip_config['text_config'],
        temporal_config = clip_config['temporal_config'],
        adapter_config = clip_config['adaptor_config'],
        state_dict = torch.load(model_path),
        **kwargs,
    ).to(device)
    model.eval()
    return model, preprocess


if __name__ == "__main__":
    model, process = load()
    print("Model loaded successfully!")
