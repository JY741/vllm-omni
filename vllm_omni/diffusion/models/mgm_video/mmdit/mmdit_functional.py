# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Minimal port of MGM-Video-Ascend functional.py.

Only includes utilities needed by the MMDiT inference code.
"""


def _ntuple(n):
    from functools import reduce
    from operator import mul

    def parse(x):
        if isinstance(x, (list, tuple)):
            return x
        return tuple(repeat(x, n))

    def repeat(elem, n):
        return (elem,) * n

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)