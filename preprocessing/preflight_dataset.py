"""Fail-closed dataset preflight. No files are modified."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from configs.config import METADATA_DIR, PROCESSED_DATASET_DIR, OUTPUT_DIR
from utils.split_integrity import assert_source_partition, assert_segments_match_sources, assert_shards_match_segments, SplitLeakageError
SPLITS=("train","validation","test")
def main():
    p=argparse.ArgumentParser(); p.add_argument("--write-report",action="store_true"); a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    source_paths={s:PROCESSED_DATASET_DIR/"manifests"/f"{s}.csv" for s in SPLITS}
    segment_paths={s:PROCESSED_DATASET_DIR/"manifests"/f"{s}_segments.csv" for s in SPLITS}
    shard_paths={s:OUTPUT_DIR/"features"/f"{s}_shard_manifest.csv" for s in SPLITS}
    try:
        print("="*78); print("ACOUSTIC DRONE DATASET PREFLIGHT"); print("="*78)
        sources=assert_source_partition(METADATA_DIR/"combined_manifest.csv",source_paths)
        print("\nSOURCE PARTITION: PASS")
        for s,d in sources.items(): print(f"  {s:12s} recordings={d.recording_group_id.nunique():,} rows={len(d):,} class0={(d.binary_label==0).sum():,} class1={(d.binary_label==1).sum():,}")
        segments=assert_segments_match_sources(sources,segment_paths)
        print("SEGMENT IDENTITY: PASS")
        for s,d in segments.items(): print(f"  {s:12s} recordings={d.recording_group_id.nunique():,} segments={len(d):,}")
        shards=assert_shards_match_segments(segments,shard_paths)
        print("SHARD IDENTITY: PASS")
        print("\nCROSS-SPLIT RECORDING OVERLAP: 0")
        print("CROSS-SPLIT SOURCE-SHA OVERLAP: 0")
        print("CROSS-SPLIT SEGMENT OVERLAP: 0")
        print("\nRESULT: PASS")
        if a.write_report:
            rows=[]
            for s,d in shards.items(): rows.append({"split":s,"rows":len(d),"recordings":d.recording_group_id.nunique(),"source_sha256":d.source_sha256.nunique(),"class_0":int((d.binary_label==0).sum()),"class_1":int((d.binary_label==1).sum())})
            out=OUTPUT_DIR/"diagnostics"/"dataset_preflight.csv"; out.parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(out,index=False); print(f"Report: {out}")
    except Exception as e:
        print(f"\nRESULT: FAIL\n{e}")
        raise SystemExit(2)
if __name__=="__main__": main()
