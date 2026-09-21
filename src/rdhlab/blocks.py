from __future__ import annotations
import numpy as np


def iter_blocks(image: np.ndarray, block_size: int = 32):
    h,w=image.shape
    idx=0
    for y in range(0,h-block_size+1,block_size):
        for x in range(0,w-block_size+1,block_size):
            yield idx, y, x, image[y:y+block_size,x:x+block_size]
            idx+=1
