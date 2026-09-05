"""Fail-closed integrity checks for the acoustic-drone dataset.

Canonical source manifests are authoritative. Derived segment and shard manifests
must exactly inherit source SHA, recording group, label, and split membership.
"""
from __future__ import annotations
from collections.abc import Mapping
from pathlib import Path
import pandas as pd

SPLIT_NAMES=("train","validation","test")

class SplitLeakageError(ValueError):
    pass

def _read(path: Path)->pd.DataFrame:
    path=Path(path).resolve()
    if not path.is_file(): raise SplitLeakageError(f"Manifest not found: {path}")
    df=pd.read_csv(path,dtype=str,keep_default_na=False)
    if df.empty: raise SplitLeakageError(f"Manifest is empty: {path}")
    return df

def _require(df, cols, origin):
    missing=set(cols)-set(df.columns)
    if missing: raise SplitLeakageError(f"{origin} missing columns: {', '.join(sorted(missing))}")

def _nonempty(df,col,origin):
    s=df[col].astype(str).str.strip()
    if s.eq("").any() or s.str.lower().eq("nan").any():
        raise SplitLeakageError(f"{origin} has empty '{col}' values")

def _overlap(frames:Mapping[str,pd.DataFrame], col:str):
    vals={k:set(v[col].astype(str)) for k,v in frames.items()}
    out={}
    names=list(frames)
    for i,a in enumerate(names):
        for b in names[i+1:]:
            x=vals[a]&vals[b]
            if x: out[f"{a}&{b}"]=x
    return out

def _fail_overlap(frames,col,name):
    x=_overlap(frames,col)
    if x:
        pair,vals=next(iter(x.items()))
        raise SplitLeakageError(f"{name} != 0: {', '.join(f'{k}={len(v)}' for k,v in x.items())}; example {pair}={next(iter(vals))}")

def load_split_manifest(path:Path, expected_split:str|None=None, *, kind:str="derived"):
    df=_read(path)
    required={"split","binary_label","recording_group_id"}
    if kind=="source": required|={"sha256"}
    elif kind=="segment": required|={"segment_id","source_sha256","segment_index"}
    elif kind=="shard": required|={"segment_id","source_sha256","shard_path","shard_index"}
    else: required|={"segment_id"}
    _require(df,required,str(path))
    labels=pd.to_numeric(df["binary_label"],errors="raise").astype(int)
    if not labels.isin([0,1]).all(): raise SplitLeakageError(f"{path}: binary_label must be 0/1")
    df["binary_label"]=labels
    if expected_split is not None and set(df["split"].astype(str))!={expected_split}:
        raise SplitLeakageError(f"{path}: expected only split '{expected_split}'")
    for c in ("recording_group_id",): _nonempty(df,c,str(path))
    return df

def assert_disjoint_splits(frames:Mapping[str,pd.DataFrame], *, check_source_sha=True):
    if set(frames)-set(SPLIT_NAMES): raise SplitLeakageError("Unknown split name")
    for name,df in frames.items():
        _require(df,{"split","recording_group_id","binary_label"},f"split '{name}'")
        if df.empty: raise SplitLeakageError(f"split '{name}' is empty")
        if set(df["split"].astype(str))!={name}: raise SplitLeakageError(f"split '{name}' contains other split values")
        groups=df.groupby(df.recording_group_id.astype(str)).binary_label.nunique()
        if (groups>1).any(): raise SplitLeakageError(f"split '{name}' contains mixed-label recording groups")
    if all("segment_id" in d.columns for d in frames.values()):
        for n,d in frames.items():
            if d.segment_id.astype(str).duplicated().any(): raise SplitLeakageError(f"split '{n}' has duplicate segment_id values")
        _fail_overlap(frames,"segment_id","cross_split_duplicate_segment_overlap")
    if check_source_sha and all("source_sha256" in d.columns for d in frames.values()):
        _fail_overlap(frames,"source_sha256","cross_split_source_sha256_overlap")
    if check_source_sha and all("sha256" in d.columns for d in frames.values()):
        _fail_overlap(frames,"sha256","cross_split_source_sha256_overlap")
    _fail_overlap(frames,"recording_group_id","cross_split_recording_overlap")

def assert_source_partition(combined_path:Path, split_paths:Mapping[str,Path]):
    combined=load_split_manifest(combined_path,kind="source")
    if combined.sha256.astype(str).duplicated().any(): raise SplitLeakageError("combined_manifest.csv contains duplicate sha256 values")
    frames={s:load_split_manifest(split_paths[s],expected_split=s,kind="source") for s in SPLIT_NAMES}
    assert_disjoint_splits(frames,check_source_sha=True)
    combined_keys=set(combined.sha256.astype(str))
    split_keys=set().union(*(set(d.sha256.astype(str)) for d in frames.values()))
    if combined_keys!=split_keys:
        raise SplitLeakageError(f"Canonical source partition mismatch: missing={len(combined_keys-split_keys)}, extra={len(split_keys-combined_keys)}")
    return frames

def assert_segments_match_sources(source_frames, segment_paths):
    out={}
    for split in SPLIT_NAMES:
        seg=load_split_manifest(segment_paths[split],expected_split=split,kind="segment")
        src=source_frames[split]
        if seg.segment_id.astype(str).duplicated().any(): raise SplitLeakageError(f"{split}: duplicate segment_id")
        by_sha=src.set_index(src.sha256.astype(str),drop=False)
        sh=seg.source_sha256.astype(str)
        missing=~sh.isin(by_sha.index)
        if missing.any(): raise SplitLeakageError(f"{split}: {int(missing.sum())} segments point outside canonical source split")
        for col in ("recording_group_id","binary_label"):
            expected=sh.map(by_sha[col].astype(str))
            actual=seg[col].astype(str)
            bad=expected!=actual
            if bad.any(): raise SplitLeakageError(f"{split}: {int(bad.sum())} {col} mismatches source manifest")
        expected_id=sh+":"+seg.segment_index.astype(str)
        if not expected_id.equals(seg.segment_id.astype(str)): raise SplitLeakageError(f"{split}: segment_id is not source_sha256:segment_index")
        out[split]=seg
    assert_disjoint_splits(out,check_source_sha=True)
    return out

def assert_shards_match_segments(segment_frames, shard_paths):
    out={}
    for split in SPLIT_NAMES:
        sh=load_split_manifest(shard_paths[split],expected_split=split,kind="shard")
        seg=segment_frames[split]
        if len(sh)!=len(seg): raise SplitLeakageError(f"{split}: shard rows {len(sh)} != segment rows {len(seg)}")
        cols=["segment_id","binary_label","source_sha256","recording_group_id"]
        a=sh[cols].sort_values("segment_id").reset_index(drop=True)
        b=seg[cols].sort_values("segment_id").reset_index(drop=True)
        if not a.equals(b): raise SplitLeakageError(f"{split}: shard manifest does not exactly match segment manifest")
        out[split]=sh
    assert_disjoint_splits(out,check_source_sha=True)
    return out

def assert_training_manifests_are_safe(train_path:Path,validation_path:Path,test_path:Path|None=None):
    frames={"train":load_split_manifest(train_path,expected_split="train",kind="shard"),"validation":load_split_manifest(validation_path,expected_split="validation",kind="shard")}
    if test_path is not None:
        frames["test"]=load_split_manifest(test_path,expected_split="test",kind="shard")
    assert_disjoint_splits(frames,check_source_sha=True)
    return frames
