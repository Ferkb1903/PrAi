from __future__ import annotations

import argparse
from pathlib import Path

import SimpleITK as sitk


def load_image(input_path: Path, is_dicom_series: bool) -> sitk.Image:
    if is_dicom_series:
        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(str(input_path))
        if not series_ids:
            raise ValueError(f"No DICOM series found in {input_path}")
        files = reader.GetGDCMSeriesFileNames(str(input_path), series_ids[0])
        reader.SetFileNames(files)
        return reader.Execute()
    return sitk.ReadImage(str(input_path))


def resample_isotropic(img: sitk.Image, spacing_mm: float) -> sitk.Image:
    old_spacing = img.GetSpacing()
    old_size = img.GetSize()

    new_spacing = (spacing_mm, spacing_mm, spacing_mm)
    new_size = [
        int(round(old_size[0] * old_spacing[0] / new_spacing[0])),
        int(round(old_size[1] * old_spacing[1] / new_spacing[1])),
        int(round(old_size[2] * old_spacing[2] / new_spacing[2])),
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(-1024)
    resampler.SetInterpolator(sitk.sitkLinear)
    return resampler.Execute(img)


def apply_bev_convention(img: sitk.Image, flip_z: bool) -> sitk.Image:
    # Standardize orientation first.
    oriented = sitk.DICOMOrient(img, "LPS")
    if flip_z:
        flip = sitk.FlipImageFilter()
        flip.SetFlipAxes([False, False, True])
        oriented = flip.Execute(oriented)
    return oriented


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess CT for OpenGATE: isotropic spacing + BEV orientation")
    parser.add_argument("--input", type=Path, required=True, help="Input CT path (.mhd) or DICOM series directory")
    parser.add_argument("--input-type", choices=["mhd", "dicom_series"], required=True)
    parser.add_argument("--output-mhd", type=Path, required=True)
    parser.add_argument("--spacing-mm", type=float, default=2.0)
    parser.add_argument("--flip-z", action="store_true", help="Flip z axis so beam can be treated as entering from -Z")
    args = parser.parse_args()

    img = load_image(args.input, is_dicom_series=(args.input_type == "dicom_series"))
    img = apply_bev_convention(img, flip_z=args.flip_z)
    img = resample_isotropic(img, spacing_mm=float(args.spacing_mm))

    args.output_mhd.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(args.output_mhd), useCompression=False)

    print(f"Saved preprocessed CT: {args.output_mhd}")
    print(f"Size: {img.GetSize()}")
    print(f"Spacing: {img.GetSpacing()}")


if __name__ == "__main__":
    main()
