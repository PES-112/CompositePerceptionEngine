"""Which SANPO object classes does v3 miss? Descriptive, over the frames cached by detector_recall_sweep.py
(172 train + 80 test). Objects and the 50%-cover rule as in methodology §4.21; "right class" uses a
per-class match list, "any box" any v3 box at conf 0.30. YOLOE names (if open_vocab_recall.py ran) show
what the open-vocabulary model calls each one (conf >= 0.2). Methodology §4.22, "Other classes".

    python evaluation/missed_objects_sanpo.py
"""
import sys, json
from collections import Counter, defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, cv2, gzip
from evaluation.detector_recall_sweep import CACHE, MIN_AREA, MAX_DEPTH_M
from src.perception_stack.depth_loader import decode_sanpo_depth
C = {12:"pedestrian",13:"rider",14:"animal",21:"vehicle",20:"obstacle",22:"traffic sign",23:"traffic light",
     24:"pole",25:"bus stop",26:"bike rack",15:"stairs",10:"opening-door",11:"opening-gate",4:"guard rail",9:"hand rail",28:"tree"}
M = {"pedestrian":{"person"},"rider":{"person","bicycle","motorcycle"},"animal":{"dog"},
     "vehicle":{"car","truck","bus","motorcycle","bicycle"},"obstacle":set(),"traffic sign":{"stop sign","pole"},
     "traffic light":{"traffic light","pole"},"pole":{"pole","bollard","traffic light","stop sign"},"bus stop":{"bench","pole"},
     "bike rack":{"bicycle","bollard","pole","bench"},"stairs":{"stairs"},"opening-door":set(),"opening-gate":set(),
     "guard rail":set(),"hand rail":set(),"tree":set()}
tot=Counter(); found=Counter(); anyv3=Counter(); ye=defaultdict(Counter); frames=0
def cov(x,y,bw,bh,b): return max(0,min(x+bw,b[4])-max(x,b[2]))*max(0,min(y+bh,b[5])-max(y,b[3]))/(bw*bh)
for split in ("train","test"):
    for d in sorted((CACHE/split).iterdir()):
        if not (d/"boxes.json").exists(): continue
        frames+=1
        B=json.loads((d/"boxes.json").read_text())
        v3=[b for b in B["640"] if b[1]>=0.30]; yo=[b for b in B.get("yoloe11s_pf",[]) if b[1]>=0.20]
        img=cv2.imread(str(d/"frame.png"),cv2.IMREAD_REDUCED_COLOR_2); h,w=img.shape[0]*2,img.shape[1]*2
        m=cv2.imread(str(d/"mask.png"),cv2.IMREAD_UNCHANGED); ids=m if m.ndim==2 else m[...,2]
        dep=decode_sanpo_depth(np.frombuffer(gzip.decompress((d/"depth.float16.gz").read_bytes()),np.float16))
        ids=cv2.resize(ids,(w,h),interpolation=cv2.INTER_NEAREST); dep=cv2.resize(dep,(w,h),interpolation=cv2.INTER_NEAREST)
        for cid,name in C.items():
            n,lab,st,_=cv2.connectedComponentsWithStats((ids==cid).astype(np.uint8),8)
            for k in range(1,n):
                x,y,bw,bh,a=st[k]
                if a<MIN_AREA*h*w: continue
                dd=float(np.median(dep[lab==k]))
                if not (0.3<dd<=MAX_DEPTH_M): continue
                tot[name]+=1
                found[name]+=any(b[0] in M[name] and cov(x,y,bw,bh,b)>=0.5 for b in v3)
                anyv3[name]+=any(cov(x,y,bw,bh,b)>=0.5 for b in v3)
                best=max(((cov(x,y,bw,bh,b),b[0]) for b in yo), default=(0,None))
                if best[0]>=0.5: ye[name][best[1]]+=1
print(frames,"frames (172 train + 80 test), objects within 10 m, >=0.3% of image")
print(f"{'SANPO class':14s} {'n':>4s} {'v3 right class':>15s} {'v3 any box':>11s}   what YOLOE (conf>=0.2) calls them")
for name,_ in tot.most_common():
    n=tot[name]
    print(f"{name:14s} {n:4d} {found[name]:6d} ({found[name]/n:.2f}) {anyv3[name]:5d} ({anyv3[name]/n:.2f})   {dict(ye[name].most_common(6))}")
