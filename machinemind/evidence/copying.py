"""Detached internal snapshots; no evidence admission, authority or cache."""
from copy import deepcopy as _generic_deepcopy, _keep_alive as _copy_keep_alive


def deepcopy(value, memo=None):
    """Match stdlib alias/memo semantics, with a bounded-JSON scalar fast path.

    All containers are detached. Unknown classes retain copy.deepcopy and their
    hooks. In particular, caller-provided memo entries override scalar values,
    including components of a numeric vector. No type is admitted by copying.
    """
    if memo is not None and type(memo) is not dict:
        return _generic_deepcopy(value, memo)
    typ = type(value)
    if value is None or typ in (str, bool, int, float, bytes):
        if memo is not None and id(value) in memo:
            return _generic_deepcopy(value, memo)
        return value
    if memo is None:
        memo = {}
    if id(value) in memo:
        return memo[id(value)]
    if typ is dict:
        out = {}
        memo[id(value)] = out
        for key, item in value.items():
            out[deepcopy(key, memo)] = deepcopy(item, memo)
        _copy_keep_alive(value, memo)
        return out
    if typ is list:
        if (value and all(type(x) is float for x in value)
                and not any(id(x) in memo for x in value)):
            out = value.copy()
            memo[id(value)] = out
            _copy_keep_alive(value, memo)
            return out
        out = []
        memo[id(value)] = out
        out.extend(deepcopy(item, memo) for item in value)
        _copy_keep_alive(value, memo)
        return out
    return _generic_deepcopy(value, memo)
