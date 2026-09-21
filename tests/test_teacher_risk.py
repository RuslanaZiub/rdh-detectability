import numpy as np
from rdhlab.detectors import fit_rich_detector
from rdhlab.teacher_risk import SRMTeacherLocalRisk, rich_detector_logit


def test_srm_teacher_local_risk_smoke():
    rng=np.random.default_rng(12)
    covers=[rng.integers(0,256,size=(64,64),dtype=np.uint8) for _ in range(8)]
    stegos=[]
    for x in covers:
        y=x.copy(); y[10:20,10:20] = np.clip(y[10:20,10:20].astype(np.int16)+1,0,255).astype(np.uint8)
        stegos.append(y)
    teacher=fit_rich_detector(covers,stegos,seed=7)
    m=SRMTeacherLocalRisk(teacher=teacher,block_size=32,probe_fraction=.25,probe_max_bits=32,seed=9)
    rows=m.score_image_blocks(covers[0],"a")
    assert len(rows)==4
    assert all(np.isfinite(r['detectability_risk']) for r in rows)
    assert all(r['probe_bits'] > 0 for r in rows)
    assert np.isfinite(rich_detector_logit(teacher,covers[0]))
