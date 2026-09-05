"""Rebuild segments -> features -> shards from the existing canonical splits.

This command NEVER regenerates or changes combined_manifest.csv or the canonical
train/validation/test source manifests. It backs up derived manifests and
removes only derived segments/features/shards before rebuilding them.
"""
from __future__ import annotations
import argparse, shutil
from datetime import datetime, timezone
from pathlib import Path
from configs.config import METADATA_DIR, PROCESSED_DATASET_DIR, RAW_DATASET_DIR, OUTPUT_DIR
from preprocessing.preprocess_training_audio import preprocess_all_splits
from feature_extraction.build_features import FeatureExtractionSettings, build_features_for_splits
from pack_features import pack_split
from utils.split_integrity import assert_source_partition, assert_segments_match_sources, assert_shards_match_segments
SPLITS=("train","validation","test")
def main():
    p=argparse.ArgumentParser(); p.add_argument("--workers",type=int,default=1); p.add_argument("--shard-size",type=int,default=256); a=p.parse_args()
    if a.workers<1 or a.shard_size<1: raise SystemExit("workers and shard-size must be positive")
    print("CANONICAL SPLITS WILL NOT BE REGENERATED.")
    source_paths={s:PROCESSED_DATASET_DIR/"manifests"/f"{s}.csv" for s in SPLITS}
    segment_paths={s:PROCESSED_DATASET_DIR/"manifests"/f"{s}_segments.csv" for s in SPLITS}
    shard_paths={s:OUTPUT_DIR/"features"/f"{s}_shard_manifest.csv" for s in SPLITS}
    sources=assert_source_partition(METADATA_DIR/"combined_manifest.csv",source_paths)
    print("[1/5] Canonical source partition: PASS")
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup=OUTPUT_DIR/"diagnostics"/f"pre_rebuild_backup_{stamp}"; backup.mkdir(parents=True,exist_ok=True)
    for path in [*segment_paths.values(),*shard_paths.values(),*[(OUTPUT_DIR/"features"/f"{s}_feature_manifest.csv") for s in SPLITS]]:
        if path.is_file(): shutil.copy2(path,backup/path.name)
    print(f"Backup: {backup}")
    # Derived audio segments
    for s in SPLITS:
        d=PROCESSED_DATASET_DIR/"segments"/s
        if d.exists(): shutil.rmtree(d)
        m=segment_paths[s]
        if m.exists(): m.unlink()
    # Derived feature arrays and shards
    for name in ("mfcc","mel","spectral","chroma","zcr","energy"):
        d=Path(__file__).resolve().parents[1]/"features"/name
        if d.exists(): shutil.rmtree(d)
    sd=Path(__file__).resolve().parents[1]/"features"/"shards"
    if sd.exists(): shutil.rmtree(sd)
    for s in SPLITS:
        fm=OUTPUT_DIR/"features"/f"{s}_feature_manifest.csv"
        sm=shard_paths[s]
        if fm.exists(): fm.unlink()
        if sm.exists(): sm.unlink()
    print("[2/5] Derived artifacts cleared; raw audio untouched.")
    preprocess_all_splits(source_manifest_directory=PROCESSED_DATASET_DIR/"manifests",raw_datasets_directory=RAW_DATASET_DIR,processed_datasets_directory=PROCESSED_DATASET_DIR,workers=a.workers,overwrite=True)
    segments=assert_segments_match_sources(sources,segment_paths); print("[3/5] Segments: PASS")
    summaries=build_features_for_splits(SPLITS,FeatureExtractionSettings(skip_existing=False))
    failed=sum(x.failed_segments for x in summaries)
    if failed: raise RuntimeError(f"Feature extraction failed for {failed:,} segments")
    print("[4/5] Features: PASS")
    for s in SPLITS: pack_split(s,a.shard_size)
    assert_shards_match_segments(segments,shard_paths); print("[5/5] Shards: PASS")
    print("RESULT: PASS — safe to train")
if __name__=="__main__": main()
