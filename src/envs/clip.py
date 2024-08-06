import numpy as np
import torch
from torch.nn import functional as F

from ..others.mineclip_official import tokenize_batch
from ..segment.segmineclip import preprocess


class CLIPReward:
    def __init__(self, clip_model, text, device, num_frames=16, neg_text_path=None):
        self.clip_model = clip_model
        self.text = text
        self.device = device
        self.num_frames = num_frames

        # load negative prompts
        if neg_text_path is None:
            neg_text_path = 'src/envs/negative_prompts.txt'
        with open(neg_text_path, 'r') as f:
            self.neg_text = f.read().splitlines()

        with torch.no_grad():
            self.text_emb = self.clip_model.text_encoder(tokenize_batch(self.text + self.neg_text).to(self.device))
            assert self.text_emb.shape[0] == 1 + len(self.neg_text)
        
        self._zero_img = preprocess(np.zeros([1, 1, 3, 160, 256])).to(self.device).float()
        with torch.no_grad():
            # ViT embedding of a ZERO image
            self._pad_emb = self.clip_model.image_encoder(self._zero_img)[0][0, 0]
        self._prev_embs = None  # ViT embedding of previous #num_frames frames
        
    def _reward(self, video_emb):
        v_emb = self.clip_model.temporal_encoder(video_emb)
        adapted_video, adapted_text = self.clip_model.reward_adapter(v_emb, self.text_emb)
        v_f = adapted_video / adapted_video.norm(dim=1, keepdim=True)   # [B, 512]
        v_f = self.clip_model.logit_scale.exp() * v_f
        t_f = adapted_text / adapted_text.norm(dim=1, keepdim=True)     # [32, 512]
        logits_per_video = v_f @ t_f.t()                                # [B, 32]
        assert logits_per_video.shape[-1] == 32
        prob = F.softmax(logits_per_video, dim=-1)[:, 0].detach().cpu().numpy() # P(video corresponds to the prompt)

        r_clip = prob - 1. / 32
        r_clip[r_clip < 0] = 0.
        
        return r_clip
    
    @torch.no_grad()
    def compute_reward(self, img_emb, next_img_emb, first):
        """ MineCLIP reward for frame t: previous #num_frames - 2 frames + current frame + next frame, 
        i.e. the MineCLIP output at frame t+1 is the reward for frame t.
        """
        T, B = img_emb.shape[:2]
        img_emb = torch.as_tensor(img_emb).to(self.device).float()
        next_img_emb = torch.as_tensor(next_img_emb).to(self.device).float()
        
        if self.num_frames == 1:
            video_emb = next_img_emb.reshape(T * B, -1).unsqueeze(dim=1)     # [T*B, 1, 512]
            r_clip = self._reward(video_emb).reshape(T, B)
        else:
            if self._prev_embs is None:
                self._prev_embs = self._pad_emb.repeat(B, self.num_frames - 1, 1)
            
            r_clip = []
            for t in range(T):
                if np.any(first[t]):
                    # if first frame, fill previous #num_frames frames with pad_emb and first frame
                    self._prev_embs[first[t], :-1] = self._pad_emb.repeat(np.sum(first[t]), self.num_frames - 2, 1)
                    self._prev_embs[first[t], -1] = img_emb[t, first[t]]
                # concatenate previous #num_frames frames and current frame
                video_emb = torch.cat([self._prev_embs, next_img_emb[t].unsqueeze(1)], dim=1)  # [B, #num_frames, 512]
                # print(",".join([str(hash(str(x.mean().item()))) for x in video_emb[2]]))
                r_clip.append(self._reward(video_emb))
                # add current frame to the end of previous #num_frames frames
                self._prev_embs = torch.roll(self._prev_embs, shifts=-1, dims=1)
                self._prev_embs[:, -1] = next_img_emb[t].clone()
        
        return np.array(r_clip)


class CLIPReward_MT:
    def __init__(self, clip_model, texts, device, num_frames=16, neg_text_path=None):
        self.clip_model = clip_model
        self.texts = texts
        self.device = device
        self.num_frames = num_frames

        # load negative prompts
        if neg_text_path is None:
            neg_text_path = 'src/envs/negative_prompts.txt'
        with open(neg_text_path, 'r') as f:
            self.neg_text = f.read().splitlines()
        
        with torch.no_grad():
            self.text_emb = self.clip_model.text_encoder(tokenize_batch(self.texts + self.neg_text).to(self.device))
            assert self.text_emb.shape[0] == len(self.texts) + len(self.neg_text)
        self._n_target = len(self.texts)

        self._zero_img = preprocess(np.zeros([1, 1, 3, 160, 256])).to(self.device).float()
        with torch.no_grad():
            # ViT embedding of a ZERO image
            self._pad_emb = self.clip_model.image_encoder(self._zero_img)[0][0, 0]
        self._prev_embs = None  # ViT embedding of previous #num_frames frames
    
    def _reward(self, video_emb, indices):
        v_emb = self.clip_model.temporal_encoder(video_emb)
        adapted_video, adapted_text = self.clip_model.reward_adapter(v_emb, self.text_emb)
        v_f = adapted_video / adapted_video.norm(dim=1, keepdim=True)   # [B, 512]
        v_f = self.clip_model.logit_scale.exp() * v_f
        t_f = adapted_text / adapted_text.norm(dim=1, keepdim=True)     # [n+31, 512]
        logits_per_video = v_f @ t_f.t()                                # [B, n+31]
        assert logits_per_video.shape[-1] == self._n_target + len(self.neg_text)

        neg_text_logits = logits_per_video[:, self._n_target:]
        indices = torch.as_tensor(indices).to(self.device)
        pos_text_logits = logits_per_video[torch.arange(len(indices)), indices]
        logits_per_video = torch.cat([pos_text_logits.unsqueeze(1), neg_text_logits], dim=1)
        prob = F.softmax(logits_per_video, dim=-1)[:, 0].detach().cpu().numpy() # P(video corresponds to the prompt)

        r_clip = prob - 1. / 32
        r_clip[r_clip < 0] = 0.

        return r_clip
    
    @torch.no_grad()
    def compute_reward(self, img_emb, next_img_emb, first, indices):
        """ MineCLIP reward for frame t: previous #num_frames - 2 frames + current frame + next frame, 
        i.e. the MineCLIP output at frame t+1 is the reward for frame t.
        """
        T, B = img_emb.shape[:2]
        img_emb = torch.as_tensor(img_emb).to(self.device).float()
        next_img_emb = torch.as_tensor(next_img_emb).to(self.device).float()
        
        if self.num_frames == 1:
            video_emb = next_img_emb.reshape(T * B, -1).unsqueeze(dim=1)     # [T*B, 1, 512]
            r_clip = self._reward(video_emb).reshape(T, B)
        else:
            if self._prev_embs is None:
                self._prev_embs = self._pad_emb.repeat(B, self.num_frames - 1, 1)
            
            r_clip = []
            for t in range(T):
                if np.any(first[t]):
                    # if first frame, fill previous #num_frames frames with pad_emb and first frame
                    self._prev_embs[first[t], :-1] = self._pad_emb.repeat(np.sum(first[t]), self.num_frames - 2, 1)
                    self._prev_embs[first[t], -1] = img_emb[t, first[t]]
                # concatenate previous #num_frames frames and current frame
                video_emb = torch.cat([self._prev_embs, next_img_emb[t].unsqueeze(1)], dim=1)  # [B, #num_frames, 512]
                # print(",".join([str(hash(str(x.mean().item()))) for x in video_emb[2]]))
                r_clip.append(self._reward(video_emb, indices[t]))
                # add current frame to the end of previous #num_frames frames
                self._prev_embs = torch.roll(self._prev_embs, shifts=-1, dims=1)
                self._prev_embs[:, -1] = next_img_emb[t].clone()
        
        return np.array(r_clip)


if __name__ == "__main__":
    from ..others.mineclip_official import build_pretrain_model
    from ..utils.utils import get_yaml_data
    
    clip_config = get_yaml_data("src/others/mineclip_official/config.yml")
    model_clip = build_pretrain_model(
        image_config = clip_config['image_config'],
        text_config = clip_config['text_config'],
        temporal_config = clip_config['temporal_config'],
        adapter_config = clip_config['adaptor_config'],
        state_dict = torch.load("src/others/mineclip_official/adjust.pth")
    ).to("cuda:0")
    model_clip.eval()
    print('MineCLIP model loaded.')

    cr = CLIPReward(model_clip, ["obtain milk from a cow with an empty bucket and a cow nearby"], device="cuda:0")
    img_emb = np.random.randn(30, 3, 512)
    first = np.zeros((30, 3), dtype=bool)
    first[0] = True
    first[10, 0] = True
    first[20, 2] = True
    cr.compute_reward(img_emb, first)
