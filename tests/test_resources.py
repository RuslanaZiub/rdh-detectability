import numpy as np
import pandas as pd
from rdhlab.resources import build_strategy_order, sideinfo_ratios, summarize_exact_recovery

class DummyRisk:
    def score_image_blocks(self,image,image_key=""):
        out=[]
        bs=4; bid=0
        for y in range(0,image.shape[0],bs):
            for x in range(0,image.shape[1],bs):
                out.append({"block_id":bid,"predictability":float(bid),"detectability_risk":float(10-bid)})
                bid+=1
        return out

def test_strategy_orders_basic_and_deterministic():
    x=np.arange(64,dtype=np.uint8).reshape(8,8)
    risk=DummyRisk()
    r,_,_=build_strategy_order(x,"a","raster",risk,.25,4,7)
    assert r.tolist()==[0,1,2,3]
    a,_,_=build_strategy_order(x,"a","random",risk,.25,4,7)
    b,_,_=build_strategy_order(x,"a","random",risk,.25,4,7)
    assert a.tolist()==b.tolist()
    d,_,_=build_strategy_order(x,"a","detectability",risk,.25,4,7)
    assert d.tolist()==[3,2,1,0]
    j,_,_=build_strategy_order(x,"a","joint",risk,.25,4,7)
    assert sorted(j.tolist())==[0,1,2,3]

def test_sideinfo_ratios():
    r=sideinfo_ratios(120,20,100)
    assert np.isclose(r["sideinfo_fraction_gross"],1/6)
    assert np.isclose(r["sideinfo_per_net"],.2)

def test_exact_summary():
    df=pd.DataFrame([
        {"feasible":True,"exact_image":True,"exact_message":True,"ber":0.0,"file_io_checked":True},
        {"feasible":False,"exact_image":None,"exact_message":None,"ber":None,"file_io_checked":True},
    ])
    s=summarize_exact_recovery(df)
    assert s["feasible_cases"]==1
    assert s["file_io_feasible_cases"]==1
    assert s["file_io_exact_image"]==1
