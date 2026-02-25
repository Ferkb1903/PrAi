from pathlib import Path
import argparse

import SimpleITK as sitk


def convert_dicom_series_to_mhd(dicom_dir: Path, output_mhd: Path) -> None:
    """Convierte una serie DICOM CT a volumen MetaImage (.mhd + .raw)."""
    dicom_dir = Path(dicom_dir)
    output_mhd = Path(output_mhd)
    output_mhd.parent.mkdir(parents=True, exist_ok=True)

    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(dicom_dir))
    if not series_ids:
        raise ValueError(f"No se encontraron series DICOM en {dicom_dir}")

    # Para este caso usamos la primera serie encontrada en la carpeta.
    # Si hubiera múltiples series, puedes filtrar por SeriesInstanceUID.
    file_names = reader.GetGDCMSeriesFileNames(str(dicom_dir), series_ids[0])
    reader.SetFileNames(file_names)

    image = reader.Execute()
    sitk.WriteImage(image, str(output_mhd), useCompression=False)

    print(f"Serie convertida: {dicom_dir}")
    print(f"Salida MHD: {output_mhd}")
    print(f"Size voxels: {image.GetSize()}")
    print(f"Spacing mm: {image.GetSpacing()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convertir DICOM CT a MHD para OpenGATE")
    parser.add_argument(
        "--dicom-dir",
        type=Path,
        required=True,
        help="Directorio que contiene los .dcm de una serie CT",
    )
    parser.add_argument(
        "--output-mhd",
        type=Path,
        required=True,
        help="Ruta de salida del archivo .mhd",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert_dicom_series_to_mhd(args.dicom_dir, args.output_mhd)
