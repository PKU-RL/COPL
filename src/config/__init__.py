import json

from torch import nn

from ..utils.utils import multimap

fn_mapping = {
    "ELU": nn.ELU,
    "ReLU": nn.ReLU,
    "Tanh": nn.Tanh,
}


def _convert(_config):
    for key, value in _config.items():
        if isinstance(value, dict):
            _convert(value)
            continue
        if isinstance(value, list):
            _config[key] = tuple(value)
        if "activation" in key:
            _config[key] = fn_mapping[value]


def load_config(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    
    _convert(config)
    
    return config
 