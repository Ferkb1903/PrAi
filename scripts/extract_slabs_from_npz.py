from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np


def extract_slabs(
    input_npz: np.ndarray,
    slab_depth: int = 8,
    slab_lateral: int = 64,
    overlap: int = 2,
) -> list[dict]:
    """
    Extrae slabs (bloques 3D) a lo largo del eje del haz.
    
    Cada slab es:
    - Eje del haz (profundidad): `slab_depth` voxeles
    - Transversal: `slab_lateral x slab_lateral` píxeles
    
    Returns: lista de dicts con posiciones y datos de cada slab
    """
    d_low = input_npz["d_low"]
    d_high = input_npz["d_high"]
    
    if d_low.shape != d_high.shape:
        raise ValueError(f"Shape mismatch low={d_low.shape} high={d_high.shape}")
    
    zlen, ylen, xlen = d_low.shape
    
    # Centro transversal
    y_center = ylen // 2
    x_center = xlen // 2
    y_start = max(0, y_center - slab_lateral // 2)
    y_end = min(ylen, y_start + slab_lateral)
    x_start = max(0, x_center - slab_lateral // 2)
    x_end = min(xlen, x_start + slab_lateral)
    
    slabs = []
    z = 0
    slab_idx = 0
    
    while z + slab_depth <= zlen:
        z_end = z + slab_depth
        
        slab_low = d_low[z:z_end, y_start:y_end, x_start:x_end].astype(np.float32)
        slab_high = d_high[z:z_end, y_start:y_end, x_start:x_end].astype(np.float32)
        
        # Validar que el slab tenga dosis no trivial
        if np.max(slab_high) < 0.1:
            z += slab_depth - overlap
            continue
        
        slabs.append({
            "slab_idx": slab_idx,
            "z_start": int(z),
            "z_end": int(z_end),
            "y_start": int(y_start),
            "y_end": int(y_end),
            "x_start": int(x_start),
            "x_end": int(x_end),
            "d_low": slab_low,
            "d_high": slab_high,
            "shape": slab_low.shape,
        })
        
        slab_idx += 1
        z += slab_depth - overlap
    
    return slabs


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract 3D slabs from NPZ for curriculum learning")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory with NPZ files (e.g. spot_campaign_v2_low2k)")
    parser.add_argument("--slab-depth", type=int, default=8, help="Slab depth along beam axis (voxels)")
    parser.add_argument("--slab-lateral", type=int, default=64, help="Slab transversal size (pixels)")
    parser.add_argument("--overlap", type=int, default=2, help="Overlap between slabs")
    parser.add_argument("--out-dir", type=Path, default=Path("data/training_npz/slabs_2k_1M"))
    parser.add_argument("--max-slabs-per-case", type=int, default=10, help="Limit slabs extracted per case")
    args = parser.parse_args()
    
    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input dir no existe: {args.input_dir}")
    
    args.out_dir.mkdir(parents=True, exist_ok=True)
    
    npz_files = sorted(args.input_dir.glob("*.npz"))
    print(f"Encontrados {len(npz_files)} NPZ en {args.input_dir}")
    
    slab_metadata = []
    total_slabs = 0
    
    for npz_file in npz_files:
        case_name = npz_file.stem
        print(f"Procesando: {case_name}...", end=" ")
        
        try:
            d = np.load(npz_file)
            slabs = extract_slabs(
                d,
                slab_depth=args.slab_depth,
                slab_lateral=args.slab_lateral,
                overlap=args.overlap,
            )
        except Exception as e:
            print(f"ERROR: {e}")
            continue
        
        slabs = slabs[:args.max_slabs_per_case]
        print(f"{len(slabs)} slabs")
        
        for slab in slabs:
            slab_name = f"{case_name}_z{slab['z_start']:03d}_{slab['z_end']:03d}"
            slab_npz = args.out_dir / f"{slab_name}.npz"
            
            np.savez_compressed(
                slab_npz,
                d_low=slab["d_low"],
                d_high=slab["d_high"],
                z_start=slab["z_start"],
                z_end=slab["z_end"],
                beam_axis=np.array([2], dtype=np.int32),  # asume eje Z es el del haz
                source_case=case_name,
            )
            
            slab_metadata.append({
                "slab_npz": str(slab_npz),
                "source_case": case_name,
                "slab_idx": slab["slab_idx"],
                "z_start": slab["z_start"],
                "z_end": slab["z_end"],
                "shape": slab["shape"],
            })
            
            total_slabs += 1
    
    # Guardar manifest de slabs
    manifest_path = args.out_dir / "manifest_all_slabs.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(slab_metadata[0].keys()) if slab_metadata else [])
        writer.writeheader()
        writer.writerows(slab_metadata)
    
    print(f"\n✓ Total slabs extraidos: {total_slabs}")
    print(f"✓ Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
