import numpy as np
from rdhlab.codec import (
    embed, extract, capacity_bits, sideinfo_bits,
    choose_peak_zero, location_map_storage_bits,
)
from rdhlab.io import write_gray_png, read_gray


def synthetic():
    rng=np.random.default_rng(1)
    # narrow distribution guarantees empty histogram bins and a useful peak
    return np.clip(rng.normal(120,18,size=(256,256)),20,220).astype(np.uint8)


def full_support():
    # Every 8-bit value occurs exactly 256 times: no empty histogram bin.
    return np.tile(np.arange(256,dtype=np.uint8),(256,1))


def test_roundtrip_memory():
    x=synthetic(); rng=np.random.default_rng(2)
    n=min(1000,capacity_bits(x))
    bits=rng.integers(0,2,size=n,dtype=np.uint8)
    y,m=embed(x,bits); xr,br=extract(y,m)
    assert np.array_equal(x,xr)
    assert np.array_equal(bits,br)
    assert m.location_map_b64 is None
    assert sideinfo_bits(m)==50


def test_roundtrip_png(tmp_path):
    x=synthetic(); bits=np.array([0,1,1,0,1]*100,dtype=np.uint8)
    y,m=embed(x,bits)
    p=tmp_path/'stego.png'; write_gray_png(p,y); y2=read_gray(p)
    xr,br=extract(y2,m)
    assert np.array_equal(x,xr)
    assert np.array_equal(bits,br)


def test_full_support_histogram_uses_location_map():
    x=full_support(); rng=np.random.default_rng(42)
    p,z,d=choose_peak_zero(x)
    assert np.bincount(x.ravel(),minlength=256)[z] > 0
    n=min(200,capacity_bits(x))
    bits=rng.integers(0,2,size=n,dtype=np.uint8)
    y,m=embed(x,bits)
    xr,br=extract(y,m)
    assert m.location_map_b64 is not None
    assert m.location_map_count > 0
    assert location_map_storage_bits(m) > 0
    assert sideinfo_bits(m) > 50
    assert np.array_equal(x,xr)
    assert np.array_equal(bits,br)


def test_full_support_roundtrip_png(tmp_path):
    x=full_support(); rng=np.random.default_rng(43)
    n=min(128,capacity_bits(x))
    bits=rng.integers(0,2,size=n,dtype=np.uint8)
    y,m=embed(x,bits)
    p=tmp_path/'full_support.png'; write_gray_png(p,y); y2=read_gray(p)
    xr,br=extract(y2,m)
    assert np.array_equal(x,xr)
    assert np.array_equal(bits,br)


def test_too_large_payload():
    x=synthetic(); n=capacity_bits(x)
    try:
        embed(x,np.zeros(n+1,dtype=np.uint8))
    except ValueError:
        return
    assert False
