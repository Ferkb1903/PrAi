from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import tqdm


class SlabDataset(Dataset):
    """Dataset de slabs 3D para curriculum learning"""
    
    def __init__(self, manifest_csv: Path, low_physical_factor: float = 1.0, normalize: bool = False):
        self.manifest_csv = manifest_csv
        self.low_physical_factor = low_physical_factor
        self.normalize = normalize
        
        self.rows = list(csv.DictReader(manifest_csv.open()))
        print(f"[SlabDataset] Cargado manifest: {len(self.rows)} slabs")
        
        # Estadísticas para normalización (opcional)
        self.d_low_stats = None
        self.d_high_stats = None
        if self.normalize:
            self._compute_stats()
    
    def _compute_stats(self) -> None:
        """Calcula media/std sobre todos los slabs"""
        d_low_vals = []
        d_high_vals = []
        for row in self.rows[:min(100, len(self.rows))]:  # muestra para rapidez
            npz_path = Path(row["slab_npz"])
            if npz_path.exists():
                d = np.load(npz_path)
                d_low_vals.extend(d["d_low"].ravel())
                d_high_vals.extend(d["d_high"].ravel())
        
        d_low_vals = np.array(d_low_vals)
        d_high_vals = np.array(d_high_vals)
        
        self.d_low_stats = {
            "mean": float(np.mean(d_low_vals[d_low_vals > 0])) if np.any(d_low_vals > 0) else 0.0,
            "std": float(np.std(d_low_vals[d_low_vals > 0])) if np.any(d_low_vals > 0) else 1.0,
        }
        self.d_high_stats = {
            "mean": float(np.mean(d_high_vals[d_high_vals > 0])) if np.any(d_high_vals > 0) else 0.0,
            "std": float(np.std(d_high_vals[d_high_vals > 0])) if np.any(d_high_vals > 0) else 1.0,
        }
        print(f"[Stats] d_low: mean={self.d_low_stats['mean']:.4f} std={self.d_low_stats['std']:.4f}")
        print(f"[Stats] d_high: mean={self.d_high_stats['mean']:.4f} std={self.d_high_stats['std']:.4f}")
    
    def __len__(self) -> int:
        return len(self.rows)
    
    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        npz_path = Path(row["slab_npz"])
        d = np.load(npz_path)
        
        d_low = d["d_low"].astype(np.float32) * self.low_physical_factor
        d_high = d["d_high"].astype(np.float32)
        
        if self.normalize and self.d_low_stats is not None:
            d_low = (d_low - self.d_low_stats["mean"]) / max(self.d_low_stats["std"], 1e-8)
            d_high = (d_high - self.d_high_stats["mean"]) / max(self.d_high_stats["std"], 1e-8)
        
        return {
            "d_low": torch.from_numpy(d_low).unsqueeze(0),  # (1, Z, Y, X)
            "d_high": torch.from_numpy(d_high).unsqueeze(0),
            "source_case": row["source_case"],
        }


class SimpleUNet3D(nn.Module):
    """UNet 3D simple para slabs"""
    
    def __init__(self, in_ch: int = 1, out_ch: int = 1):
        super().__init__()
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv3d(in_ch, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 16, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool3d(2)
        
        self.enc2 = nn.Sequential(
            nn.Conv3d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool3d(2)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv3d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # Decoder
        self.upconv2 = nn.ConvTranspose3d(64, 32, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv3d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        self.upconv1 = nn.ConvTranspose3d(32, 16, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv3d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 16, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        self.final = nn.Conv3d(16, out_ch, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        # Bottleneck
        b = self.bottleneck(p2)
        
        # Decoder
        u2 = self.upconv2(b)
        u2 = torch.cat([u2, e2], dim=1)
        d2 = self.dec2(u2)
        
        u1 = self.upconv1(d2)
        u1 = torch.cat([u1, e1], dim=1)
        d1 = self.dec1(u1)
        
        out = self.final(d1)
        return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Train on 3D slabs (curriculum learning)")
    parser.add_argument("--manifest-train", type=Path, required=True, help="Train manifest CSV")
    parser.add_argument("--manifest-val", type=Path, default=None)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--low-physical-factor", type=float, default=1.0)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=Path("checkpoints/slab_training"))
    args = parser.parse_args()
    
    args.out_dir.mkdir(parents=True, exist_ok=True)
    
    # Dataset
    train_ds = SlabDataset(args.manifest_train, args.low_physical_factor, args.normalize)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    
    val_dl = None
    if args.manifest_val:
        val_ds = SlabDataset(args.manifest_val, args.low_physical_factor, args.normalize)
        val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Model
    model = SimpleUNet3D(in_ch=1, out_ch=1).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.L1Loss()
    
    writer = SummaryWriter(f"{args.out_dir}/logs")
    
    # Training loop
    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss = 0.0
        n_batches = 0
        
        for batch in tqdm.tqdm(train_dl, desc=f"Epoch {epoch+1}/{args.epochs}"):
            d_low = batch["d_low"].to(args.device)
            d_high = batch["d_high"].to(args.device)
            
            pred = model(d_low)
            loss = criterion(pred, d_high)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += float(loss.item())
            n_batches += 1
        
        train_loss /= max(1, n_batches)
        writer.add_scalar("loss/train", train_loss, epoch)
        
        # Validation
        if val_dl:
            model.eval()
            val_loss = 0.0
            val_batches = 0
            with torch.no_grad():
                for batch in val_dl:
                    d_low = batch["d_low"].to(args.device)
                    d_high = batch["d_high"].to(args.device)
                    pred = model(d_low)
                    loss = criterion(pred, d_high)
                    val_loss += float(loss.item())
                    val_batches += 1
            
            val_loss /= max(1, val_batches)
            writer.add_scalar("loss/val", val_loss, epoch)
            print(f"Epoch {epoch+1}: train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
        else:
            print(f"Epoch {epoch+1}: train_loss={train_loss:.6f}")
        
        # Save checkpoint
        ckpt_path = args.out_dir / f"epoch_{epoch+1:03d}.pt"
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "loss": train_loss,
        }, ckpt_path)
    
    writer.close()
    print(f"\n✓ Training completo. Checkpoints en: {args.out_dir}")


if __name__ == "__main__":
    main()
