"""Pack feature arrays into auditable NPZ shards."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from configs.config import OUTPUT_DIR, PROJECT_ROOT
FEATURES_ROOT=PROJECT_ROOT/"features"; MANIFEST_ROOT=OUTPUT_DIR/"features"; SPLITS=("train","validation","test"); FEATURE_NAMES=("mfcc","mel","spectral","chroma","zcr","energy")

def load_feature(path, name):
    p=Path(path)
    if not p.is_file(): raise FileNotFoundError(f"Missing {name}: {p}")
    a=np.asarray(np.load(p,allow_pickle=False),dtype=np.float32)
    if a.ndim!=2 or 0 in a.shape or not np.isfinite(a).all(): raise ValueError(f"Invalid {name} array: {p} shape={a.shape}")
    return np.ascontiguousarray(a)

def feature_path_from_manifest(value, segment_id, feature_name):
    p=Path(value)
    candidates=[p if p.is_absolute() else (PROJECT_ROOT/p).resolve(), FEATURES_ROOT/feature_name/f"{segment_id.replace(':','_')}.npy"]
    for c in candidates:
        if c.is_file(): return c.resolve()
    raise FileNotFoundError(f"Could not find {feature_name} for {segment_id}")

def pack_split(split,shard_size=256):
    if split not in SPLITS: raise ValueError(split)
    mp=MANIFEST_ROOT/f"{split}_feature_manifest.csv"
    if not mp.is_file(): raise FileNotFoundError(mp)
    df=pd.read_csv(mp,dtype=str,keep_default_na=False)
    req={"split","segment_id","binary_label","source_sha256","recording_group_id",*(f"{n}_path" for n in FEATURE_NAMES)}
    missing=req-set(df.columns)
    if missing: raise ValueError(f"{mp} missing: {', '.join(sorted(missing))}")
    if "status" in df.columns and (df.status.astype(str)=="failed").any(): raise ValueError(f"{mp} contains failed feature rows")
    df=df[df.get("status",pd.Series(["success"]*len(df))).isin(["success","skipped"])].copy().reset_index(drop=True)
    if df.empty: raise ValueError(f"No usable rows: {mp}")
    if set(df.split.astype(str))!={split}: raise ValueError(f"{mp} contains wrong split values")
    if df.segment_id.astype(str).duplicated().any(): raise ValueError(f"{mp} contains duplicate segment_id")
    out=FEATURES_ROOT/"shards"/split; out.mkdir(parents=True,exist_ok=True)
    for old in out.glob("shard_*.npz"): old.unlink()
    rows=[]
    for sn,start in enumerate(range(0,len(df),shard_size)):
        batch=df.iloc[start:start+shard_size]
        fb={n:[] for n in FEATURE_NAMES}; labels=[]; ids=[]; shas=[]; groups=[]
        for _,r in batch.iterrows():
            sid=str(r.segment_id)
            for n in FEATURE_NAMES: fb[n].append(load_feature(feature_path_from_manifest(r[f"{n}_path"],sid,n),n))
            labels.append(int(r.binary_label)); ids.append(sid); shas.append(str(r.source_sha256)); groups.append(str(r.recording_group_id))
        arrays={n:np.stack(v).astype(np.float32,copy=False) for n,v in fb.items()}
        sp=out/f"shard_{sn:04d}.npz"
        np.savez_compressed(sp,**arrays,labels=np.asarray(labels,dtype=np.int64),segment_ids=np.asarray(ids,dtype=np.str_),source_sha256=np.asarray(shas,dtype=np.str_),recording_group_ids=np.asarray(groups,dtype=np.str_))
        for i,sid in enumerate(ids): rows.append({"split":split,"segment_id":sid,"binary_label":labels[i],"source_sha256":shas[i],"recording_group_id":groups[i],"source_dataset":str(batch.iloc[i].get("source_dataset","")),"source_relative_path":str(batch.iloc[i].get("source_relative_path","")),"shard_path":str(sp),"shard_index":i})
    outm=MANIFEST_ROOT/f"{split}_shard_manifest.csv"; pd.DataFrame(rows).to_csv(outm,index=False); print(f"{split}: {len(rows):,} rows -> {outm}")

def main():
    p=argparse.ArgumentParser(); p.add_argument("--shard-size",type=int,default=256); a=p.parse_args()
    if a.shard_size<1: raise SystemExit("--shard-size must be >0")
    for s in SPLITS: pack_split(s,a.shard_size)
if __name__=="__main__": main()
