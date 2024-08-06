import importlib


def get_callable(name: str):
    module_name, class_name = name.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), class_name)


def get_n_actions(ac_space):
    if ac_space.__class__.__name__ == "Discrete":
        return ac_space.n
    if ac_space.__class__.__name__ == "Box":
        return ac_space.shape[0]
    if ac_space.__class__.__name__ == "MultiDiscrete":
        return ac_space.nvec.tolist()
    raise ValueError("Unknown action space: {}".format(ac_space))
