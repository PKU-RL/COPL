from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from skimage import measure
from torch import nn

from ..others.mineclip_official.tokenizer import tokenize_batch


def prompt_denoising(
    features: torch.Tensor, 
    scale: nn.Parameter, 
    threshold: float
) -> torch.Tensor:
    # Ignore class whose maximum probability over the whole image is less than threshold
    b, h, w, _ = features.shape
    features = features.reshape(b, h * w, -1)
    probs = F.softmax(features * scale.exp(), dim=-1)
    max_probs_per_image = probs.max(dim=1, keepdim=True)[0]
    ignored = (max_probs_per_image < threshold).repeat(1, features.shape[1], 1)
    features[ignored] = -50.
    
    return features.reshape(b, h, w, -1)
    

def key_smoothing(
    features: torch.Tensor, 
    key: torch.Tensor, 
    scale: nn.Parameter, 
    threshold: float
) -> torch.Tensor:
    b, h, w, _ = features.shape
    features = features.reshape(b, h * w, -1)
    key = key.reshape(b, h * w, -1)
    
    probs = F.softmax(features * scale.exp(), dim=-1)
    key = F.normalize(key, p=2, dim=-1) # NLC
    weight = key @ key.transpose(1, 2)  # NLL
    weighted_probs = weight @ probs
    weighted_probs /= weighted_probs.sum(dim=-1, keepdim=True)
    selected = (probs.max(dim=-1, keepdim=True)[0] < threshold).repeat(1, 1, probs.shape[-1])
    probs[selected] = weighted_probs[selected]
    
    return probs.reshape(b, h, w, -1)


def encode_text_with_prompt_ensemble(
    model: nn.Module, 
    texts: List[str],
    device: Union[str, torch.device] = 'cpu',
    prompt_templates: Optional[List[str]] = None,
) -> torch.Tensor:
    # using default prompt templates for ImageNet
    if prompt_templates == None:
        prompt_templates = ['a bad photo of a {}.', 'a photo of many {}.', 'a sculpture of a {}.', 'a photo of the hard to see {}.', 'a low resolution photo of the {}.', 'a rendering of a {}.', 'graffiti of a {}.', 'a bad photo of the {}.', 'a cropped photo of the {}.', 'a tattoo of a {}.', 'the embroidered {}.', 'a photo of a hard to see {}.', 'a bright photo of a {}.', 'a photo of a clean {}.', 'a photo of a dirty {}.', 'a dark photo of the {}.', 'a drawing of a {}.', 'a photo of my {}.', 'the plastic {}.', 'a photo of the cool {}.', 'a close-up photo of a {}.', 'a black and white photo of the {}.', 'a painting of the {}.', 'a painting of a {}.', 'a pixelated photo of the {}.', 'a sculpture of the {}.', 'a bright photo of the {}.', 'a cropped photo of a {}.', 'a plastic {}.', 'a photo of the dirty {}.', 'a jpeg corrupted photo of a {}.', 'a blurry photo of the {}.', 'a photo of the {}.', 'a good photo of the {}.', 'a rendering of the {}.', 'a {} in a video game.', 'a photo of one {}.', 'a doodle of a {}.', 'a close-up photo of the {}.', 'a photo of a {}.', 'the origami {}.', 'the {} in a video game.', 'a sketch of a {}.', 'a doodle of the {}.', 'a origami {}.', 'a low resolution photo of a {}.', 'the toy {}.', 'a rendition of the {}.', 'a photo of the clean {}.', 'a photo of a large {}.', 'a rendition of a {}.', 'a photo of a nice {}.', 'a photo of a weird {}.', 'a blurry photo of a {}.', 'a cartoon {}.', 'art of a {}.', 'a sketch of the {}.', 'a embroidered {}.', 'a pixelated photo of a {}.', 'itap of the {}.', 'a jpeg corrupted photo of the {}.', 'a good photo of a {}.', 'a plushie {}.', 'a photo of the nice {}.', 'a photo of the small {}.', 'a photo of the weird {}.', 'the cartoon {}.', 'art of the {}.', 'a drawing of the {}.', 'a photo of the large {}.', 'a black and white photo of a {}.', 'the plushie {}.', 'a dark photo of a {}.', 'itap of a {}.', 'graffiti of the {}.', 'a toy {}.', 'itap of my {}.', 'a photo of a cool {}.', 'a photo of a small {}.', 'a tattoo of the {}.', 'there is a {} in the scene.', 'there is the {} in the scene.', 'this is a {} in the scene.', 'this is the {} in the scene.', 'this is one {} in the scene.']

    text_features = []
    for t in texts:
        prompted_t = [template.format(t) for template in prompt_templates]
        prompted_t = tokenize_batch(prompted_t).to(device)
        class_embeddings = model.encode_text(prompted_t)
        class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
        class_embedding = class_embeddings.mean(dim=0)
        class_embedding /= class_embedding.norm()
        text_features.append(class_embedding)
    text_features = torch.stack(text_features, dim=1).to(device).t()

    return text_features


def get_similarity_map(
    sm: torch.Tensor, 
    shape: Tuple[int], 
    norm: bool = True, 
    mode: Optional[str] = None,
) -> torch.Tensor:
    # min-max norm
    if norm:
        b, h, w, _ = sm.shape
        sm = sm.reshape(b, h * w, -1)
        sm = (sm - sm.min()) / (sm.max() - sm.min())
        sm = sm.reshape(b, h, w, -1)
    
    # interpolate
    if mode is not None:
        sm = sm.permute(0, 3, 1, 2)  # BHWC -> BCHW
        if mode == 'repeat':
            sm = sm.repeat_interleave(16, dim=2).repeat_interleave(16, dim=3)
        else:
            sm = F.interpolate(sm, shape, mode=mode)
        sm = sm.permute(0, 2, 3, 1) # BCHW -> BHWC
    
    return sm


def generate_bbox(
    sm_map: torch.Tensor,
    cls_logits: torch.Tensor,
    target_index: List[int],
    image_prob_or_logit: bool = False,
    patch_prob_or_logit: bool = False,
    image_threshold: float = 0.2,
    patch_threshold: float = 0.2,
    logit_scale: torch.Tensor = torch.ones([]),
) -> Tuple[np.ndarray, np.ndarray]:
    target_index = torch.as_tensor(target_index, device=sm_map.device).unsqueeze(-1)
    if image_prob_or_logit:
        # cls_logits = F.softmax(cls_logits * logit_scale.exp(), dim=-1)[..., target_index]
        cls_logits = torch.gather(F.softmax(cls_logits * logit_scale.exp(), dim=-1),
                                  dim=-1, index=target_index)
    else:
        # cls_logits = cls_logits[..., target_index]
        cls_logits = torch.gather(cls_logits, dim=-1, index=target_index)
    if patch_prob_or_logit:
        sm_map = F.softmax(sm_map * logit_scale.exp(), dim=-1)
    sm_map = sm_map.cpu().numpy()

    bboxes, masks = [], []
    for i, sm in enumerate(sm_map):
        seg = sm.argmax(axis=-1)

        ignored = (sm < patch_threshold).all(axis=-1)
        seg[ignored] = -1

        mask = (seg == target_index[i, 0].item()).astype(np.uint8)
        labels = measure.label(mask)
        properties = measure.regionprops(labels)

        max_area = 0
        bbox = None
        mask = np.zeros_like(seg)
        for prop in properties:
            min_row, min_col, max_row, max_col = prop.bbox
            # areas = (max_row - min_row) * (max_col - min_col)
            areas = (labels == prop.label).sum()
            if areas > max_area:
                max_area = areas
                bbox = prop.bbox
        
        if max_area == 0 or cls_logits[i] < image_threshold:
            # print(f"Target not found at frame {i}")
            bbox = np.array([0, 0, 0, 0])
        else:
            min_x, min_y, max_x, max_y = bbox
            bbox = np.array([min_x, min_y, max_x, max_y])
            mask[min_x: max_x, min_y: max_y] = 1.
        
        bboxes.append(bbox)
        masks.append(mask)

    bboxes = np.array(bboxes).astype(np.float32)
    # Scale bbox to 0~1
    bboxes[:, 0] /= sm_map.shape[1]
    bboxes[:, 1] /= sm_map.shape[2]
    bboxes[:, 2] /= sm_map.shape[1]
    bboxes[:, 3] /= sm_map.shape[2]
    
    return bboxes, np.array(masks)


def softmax(x: np.ndarray, axis: int = None):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)
