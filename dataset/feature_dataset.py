"""NPZ feature dataset with identity validation."""
from __future__ import annotations
from collections import OrderedDict
from pathlib import Path
from typing import Any
import numpy as np,pandas as pd,torch
from torch.utils.data import Dataset
FEATURE_NAMES=("mfcc","mel","spectral","chroma","zcr","energy")
class FeatureDataset(Dataset):
    def __init__(self,manifest_path,*,validate_features=True,project_root=None,max_cached_shards=2):
        self.manifest_path=Path(manifest_path).resolve(); self.project_root=Path(project_root).resolve() if project_root else self.manifest_path.parents[2]
        if not self.manifest_path.is_file(): raise FileNotFoundError(self.manifest_path)
        self.dataframe=pd.read_csv(self.manifest_path,dtype=str,keep_default_na=False)
        req={"split","segment_id","binary_label","source_sha256","recording_group_id","shard_path","shard_index"}; missing=req-set(self.dataframe.columns)
        if missing: raise ValueError(f"Manifest missing columns: {', '.join(sorted(missing))}")
        if self.dataframe.empty: raise ValueError("Manifest is empty")
        self.dataframe.binary_label=pd.to_numeric(self.dataframe.binary_label,errors="raise").astype(int); self.dataframe.shard_index=pd.to_numeric(self.dataframe.shard_index,errors="raise").astype(int)
        if not self.dataframe.binary_label.isin([0,1]).all(): raise ValueError("binary_label must be 0/1")
        if self.dataframe.segment_id.astype(str).duplicated().any(): raise ValueError("Duplicate segment_id in shard manifest")
        for c in ("segment_id","source_sha256","recording_group_id"):
            if self.dataframe[c].astype(str).str.strip().eq("").any(): raise ValueError(f"Empty identity in {c}")
        self.validate_features=validate_features; self._cache=OrderedDict(); self.max_cached_shards=max(1,int(max_cached_shards))
        if validate_features: self._validate_shards()
    def __len__(self): return len(self.dataframe)
    def _resolve(self,v):
        p=Path(v); candidates=[p if p.is_absolute() else (self.project_root/p).resolve()]
        if p.is_absolute() and "features" in p.parts: candidates.append(self.project_root/Path(*p.parts[p.parts.index("features"):]))
        for c in candidates:
            if c.is_file(): return c.resolve()
        raise FileNotFoundError(v)
    def _load(self,p):
        p=self._resolve(p)
        if p in self._cache: x=self._cache.pop(p); self._cache[p]=x; return x
        x=np.load(p,allow_pickle=False); self._cache[p]=x
        while len(self._cache)>self.max_cached_shards:
            _,old=self._cache.popitem(last=False); old.close()
        return x
    def _validate_shards(self):
        for idx,r in self.dataframe.iterrows():
            sh=self._load(r.shard_path); i=int(r.shard_index)
            if i<0 or i>=len(sh["labels"]): raise IndexError(f"Invalid shard_index at row {idx}")
            checks={"segment_id":str(sh["segment_ids"][i]),"binary_label":int(sh["labels"][i]),"source_sha256":str(sh["source_sha256"][i]),"recording_group_id":str(sh["recording_group_ids"][i])}
            for c,v in checks.items():
                expected=str(r[c]); actual=str(v)
                if expected!=actual: raise ValueError(f"Shard/manifest {c} mismatch at row {idx}: {expected} != {actual}")
            for n in FEATURE_NAMES:
                a=np.asarray(sh[n]);
                if a.ndim!=3 or a.shape[0]!=len(sh["labels"]): raise ValueError(f"Invalid {n} shape in {r.shard_path}: {a.shape}")
                if not np.isfinite(a).all(): raise ValueError(f"NaN/Inf in {n}: {r.shard_path}")
    def __getitem__(self,index):
        r=self.dataframe.iloc[index]; sh=self._load(r.shard_path); i=int(r.shard_index)
        out={n:torch.from_numpy(np.asarray(sh[n][i],dtype=np.float32)) for n in FEATURE_NAMES}; out["label"]=torch.tensor(int(r.binary_label),dtype=torch.long); out["segment_id"]=str(r.segment_id); out["source_sha256"]=str(r.source_sha256); out["recording_group_id"]=str(r.recording_group_id); return out
