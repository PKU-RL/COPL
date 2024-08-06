from typing import Any, Callable, Sequence

import numpy as np
import yaml


def multimap(f: Callable, *xs: Any) -> Any:
    """
    Apply f at each leaf of the list of trees

    A tree is:
        * a (possibly nested) dict
        * a (possibly nested) DictType
        * any other object (a leaf value)

    `{"a": 1}`, `{"a": {"b": 2}}`, and `3` are all valid trees, where the leaf values
    are the integers

    :param f: function to call at each leaf, must take len(xs) arguments
    :param xs: a list of trees, all with the same structure

    :returns: A tree of the same structure, where each leaf contains f's return value.
    """
    first = xs[0]
    if isinstance(first, dict) or first.__class__.__name__ == "DictType":
        assert all(isinstance(x, dict) or first.__class__.__name__ == "DictType" for x in xs)
        assert all(sorted(x.keys()) == sorted(first.keys()) for x in xs)
        return {k: multimap(f, *(x[k] for x in xs)) for k in sorted(first.keys())}
    else:
        return f(*xs)


def concat(xs: Sequence[Any], axis: int = 0) -> Any:
    return multimap(lambda *xs: np.concatenate(xs, axis=axis), *xs)


def stack(xs: Sequence[Any], axis: int = 0) -> Any:
    return multimap(lambda *xs: np.stack(xs, axis=axis), *xs)


def split(x: Any, sections: Sequence[int]) -> Sequence[Any]:
    result = []
    start = 0
    for end in sections:
        select_tree = multimap(lambda arr: arr[start:end], x)
        start = end
        result.append(select_tree)
    return result


def assign(x: dict, y: dict, index: int):
    # x[index] = y
    def _func(a, b):
        a[index] = b
    multimap(_func, x, y)


def get_yaml_data(yaml_file):
    file = open(yaml_file, 'r', encoding="utf-8")
    file_data = file.read()
    file.close()
    
    data = yaml.load(file_data, Loader=yaml.FullLoader)
    return data


def split_episode(x, first):
    assert x.shape[0] == first.shape[0]
    assert x.shape[1] + 1 == first.shape[1]
    
    x_eps = []
    for xb, fb in zip(x, first):
        indices = np.where(fb)[0]
        for b, e in zip(indices[:-1], indices[1:]):
            x_eps.append(xb[b:e])
    
    return x_eps
