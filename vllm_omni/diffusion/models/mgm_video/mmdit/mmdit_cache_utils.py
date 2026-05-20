# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Cache algorithm utilities for TDM (Token-level Diffusion Model) cache.

Ported from MGM-Video-Ascend mimogpt/models/dit/cache_utils.py.
"""

import numpy as np


def read_2d_array_from_file_int(file_path):
    """Read a 2D array of integers from a text file.

    Each line should contain exactly 8 space-separated integers.
    Used by MMDiTInference to load the cache scheme that determines
    which blocks to compute vs skip at each denoising timestep.
    """
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                try:
                    numbers = [int(num) for num in line.split()]
                    if len(numbers) == 8:
                        data.append(numbers)
                    else:
                        print(f"Warning: Line '{line}' does not contain exactly 8 numbers.")
                except ValueError:
                    print(f"Warning: Could not convert line '{line}' to numbers.")
    return np.array(data)