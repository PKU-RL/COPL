from typing import Any, Sequence

import numpy as np
import torch
from minedojo.sim.inventory import InventoryItem

# from ..others.mineagent.features.voxel import VOXEL_BLOCK_NAME_MAP
VOXEL_BLOCK_NAME_MAP = {
    "obsidian": 1,
    "portal": 1,
    "tnt": 2,
    "melon": 3,
    "sugar cane": 4,
    "crops": 5,
    "carrot": 6,
    "fence": 7,
    "water": 8,
    "air": 9,
    "grass block": 10,
    "farmland": 11,
    "oak fence": 12,
    "lava": 13,
    "wood": 14,
    "pumpkin": 15,
    "oak sapling": 16,
    "carpet": 17,
    # ------ below are placeholders, so we can have an embedding layer with consistent size ------
    "placeholder_7": 18,
    "placeholder_8": 19,
    "placeholder_9": 20,
    "placeholder_10": 21,
    "placeholder_11": 22,
    "placeholder_12": 23,
    "placeholder_13": 24,
    "placeholder_14": 25,
    "placeholder_15": 26,
    "placeholder_16": 27,
    "placeholder_17": 28,
    "placeholder_18": 29,
    "placeholder_19": 30,
    "placeholder_20": 31,
}
MAX_IDX = max(list(VOXEL_BLOCK_NAME_MAP.values()))
NUM_TYPES = MAX_IDX + 1


def preprocess_obs(obs):
    """
    Here you preprocess the raw env obs to pass to the agent.
    Preprocessing includes, for example, use MineCLIP to extract image feature and prompt feature,
    flatten and embed voxel names, mask unused obs, etc.
    """
    B = 1

    def cvt_voxels(vox):
        ret = np.zeros(3 * 3 * 3, dtype=np.long)
        for i, v in enumerate(vox.reshape(3 * 3 * 3)):
            if v in VOXEL_BLOCK_NAME_MAP:
                ret[i] = VOXEL_BLOCK_NAME_MAP[v]
        return ret

    # I consider the move and functional action only, because the camera space is too large?
    # construct a 3*3*4*3 action embedding
    def cvt_action(act):
        if act[5] <= 1:
            return act[0] + 3*act[1] + 9*act[2] + 36*act[5]
        elif act[5] == 3:
            return act[0] + 3*act[1] + 9*act[2] + 72
        else:
            raise Exception('Action[5] should be 0, 1, 3')

    yaw_ = np.deg2rad(obs["location_stats"]["yaw"])
    pitch_ = np.deg2rad(obs["location_stats"]["pitch"])
    gps_scaled = obs["location_stats"]["pos"]
    gps_scaled[0] /= 1000
    gps_scaled[1] /= 100
    gps_scaled[2] /= 1000
    
    processed_obs = {
        "compass": np.concatenate([np.cos(yaw_), np.sin(yaw_), np.cos(pitch_), np.sin(pitch_)]),
        "gps": gps_scaled,
        "voxels": cvt_voxels(obs["voxels"]["block_name"]),
        "biome_id": np.array([int(obs["location_stats"]["biome_id"])], dtype=np.int64),
    }
    
    if "prev_action" in obs:
        processed_obs["prev_action"] = np.array([cvt_action(obs["prev_action"])], dtype=np.int64)
    
    if "obs_emb" in obs:
        processed_obs["obs_emb"] = obs["obs_emb"]
        if "target_emb" in obs:
            processed_obs["target_emb"] = obs["target_emb"]
            processed_obs["target_bbox"] = obs["target_bbox"]
    else:
        # raw image
        processed_obs["obs_rgb"] = obs["rgb"]
    
    return processed_obs


def transform_action(act, camera_interval=2):
    assert len(act) == 2 # (1, 2)
    # act = act[0]
    # act = act.cpu().numpy()
    act1, act2 = act[0], act[1]
    
    action = [0, 0, 0, 12, 12, 0, 0, 0] #self.base_env.action_space.no_op()
    assert act1 < 12
    if act1 == 0: # no op
        action = action
    elif act1 < 3: # forward backward
        action[0] = act1
    elif act1 < 5: # left right
        action[1] = act1 - 2
    elif act1 < 8: # jump sneak sprint
        action[2] = act1 - 4
    elif act1 == 8: # camera pitch 10 (default)
        action[3] = 12 - camera_interval
    elif act1 == 9: # camera pitch 14
        action[3] = 12 + camera_interval
    elif act1 == 10: # camera yaw 10
        action[4] = 12 - camera_interval
    elif act1 == 11: # camera yaw 14
        action[4] = 12 + camera_interval

    assert act2 < 3
    if act2 == 1: # use
        action[5] = 1
        # pass # deactivate use
    elif act2 == 2: #attack
        action[5] = 3
    return action #(8)


def allsame(xs):
    """
    Returns whether all elements of sequence are the same
    """
    assert len(xs) > 0
    return all(x == xs[0] for x in xs[1:])


def _chunk_seq(x: Sequence, sizes: Sequence[int]) -> Sequence[Any]:
    result = []
    assert len(x) == sum(sizes), f"x has incorrect length {len(x)} != {sum(sizes)}"
    start = 0
    for size in sizes:
        end = start + size
        result.append(x[start:end])
        start = end
    return result


def _parse_inventory_dict(inv_dict: dict[str, dict]) -> list[InventoryItem]:
    return [InventoryItem(slot=k, **v) for k, v in inv_dict.items()]
