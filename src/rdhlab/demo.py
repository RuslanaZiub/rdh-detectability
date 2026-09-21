from __future__ import annotations
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw


def generate_demo_images(out_dir: str | Path, n: int = 24, size: int = 256, seed: int = 20260916) -> list[Path]:
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(seed); paths=[]
    yy,xx=np.mgrid[:size,:size]
    for i in range(n):
        base=(70 + 80*(xx/size) + 35*np.sin((yy+i*7)/17.0)).astype(float)
        noise=rng.normal(0, 5 + (i%4)*3, size=(size,size))
        arr=np.clip(base+noise,0,255).astype(np.uint8)
        im=Image.fromarray(arr,'L')
        draw=ImageDraw.Draw(im)
        if i%3==0: draw.rectangle((40,40,180,180), outline=180, width=3)
        if i%3==1: draw.ellipse((55,55,205,205), outline=35, width=4)
        p=out/f'demo_{i:03d}.png'; im.save(p); paths.append(p)
    return paths
