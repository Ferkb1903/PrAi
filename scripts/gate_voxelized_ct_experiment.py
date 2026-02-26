from pathlib import Path
import argparse
import json

import opengate as gate
import SimpleITK as sitk


# Mapeo HU -> material para arrancar el experimento.
# Debes calibrarlo con tu protocolo clínico HU->SPR/material.
DEFAULT_HU_MATERIALS = [
    (-2000, -950, "G4_AIR"),
    (-949, -300, "G4_LUNG_ICRP"),
    (-299, 200, "G4_WATER"),
    (201, 2000, "G4_BONE_COMPACT_ICRU"),
]


def _load_hu_material_map(hu_map_json: Path | None) -> list[tuple[int, int, str]]:
    if hu_map_json is None:
        return DEFAULT_HU_MATERIALS

    hu_map_json = Path(hu_map_json)
    if not hu_map_json.exists():
        raise FileNotFoundError(f"No existe hu-map-json: {hu_map_json}")

    with open(hu_map_json, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if "ranges" not in payload or not isinstance(payload["ranges"], list):
        raise ValueError("El JSON de materiales debe contener 'ranges' como lista")

    ranges = []
    for i, r in enumerate(payload["ranges"]):
        try:
            hu_min = int(r["hu_min"])
            hu_max = int(r["hu_max"])
            material = str(r["material"])
        except KeyError as exc:
            raise ValueError(f"Falta campo en ranges[{i}]: {exc}") from exc

        if hu_min > hu_max:
            raise ValueError(f"Rango inválido en ranges[{i}]: hu_min > hu_max")
        ranges.append((hu_min, hu_max, material))

    ranges.sort(key=lambda t: t[0])

    # Verifica continuidad sin huecos entre rangos (convención de enteros HU).
    for i in range(1, len(ranges)):
        prev_max = ranges[i - 1][1]
        curr_min = ranges[i][0]
        if curr_min > prev_max + 1:
            raise ValueError(
                f"Hueco en mapeo HU entre {prev_max} y {curr_min} en {hu_map_json}"
            )

    return ranges


def _add_point_proton_source(
    sim: gate.Simulation,
    name: str,
    proton_energy_mev: float,
    n_events: int,
    x_mm: float,
    y_mm: float,
    z_cm: float,
) -> None:
    mm = gate.g4_units.mm
    cm = gate.g4_units.cm
    MeV = gate.g4_units.MeV

    source = sim.add_source("GenericSource", name)
    source.particle = "proton"
    source.energy.mono = proton_energy_mev * MeV
    source.position.type = "point"
    source.position.translation = [x_mm * mm, y_mm * mm, z_cm * cm]
    source.direction.type = "momentum"
    source.direction.momentum = [0, 0, 1]
    source.n = int(n_events)


def _make_beamlet_grid(nx: int, ny: int, pitch_mm: float) -> list[tuple[float, float]]:
    x0 = -0.5 * (nx - 1) * pitch_mm
    y0 = -0.5 * (ny - 1) * pitch_mm
    return [(x0 + ix * pitch_mm, y0 + iy * pitch_mm) for iy in range(ny) for ix in range(nx)]


def build_simulation(
    ct_mhd: Path,
    output_dir: Path,
    proton_energy_mev: float,
    n_events: int,
    seed: int,
    source_mode: str,
    beamlet_nx: int,
    beamlet_ny: int,
    beamlet_pitch_mm: float,
    source_z_cm: float,
    hu_map_json: Path | None,
    event_modulo: int,
) -> gate.Simulation:
    sim = gate.Simulation()

    mm = gate.g4_units.mm
    m = gate.g4_units.m

    sim.output_dir = str(output_dir)
    sim.visu = False
    sim.random_seed = int(seed)
    if int(event_modulo) > 0:
        sim.g4_commands_after_init.append(f"/run/eventModulo {int(event_modulo)} 1")

    world = sim.world
    world.size = [2.0 * m, 2.0 * m, 2.0 * m]
    world.material = "G4_AIR"

    # Geometría voxelizada desde CT.
    patient = sim.add_volume("Image", "patient")
    patient.image = str(ct_mhd)
    patient.material = "G4_AIR"
    hu_materials = _load_hu_material_map(hu_map_json)
    patient.voxel_materials = hu_materials
    print(f"HU->material bins: {len(hu_materials)}")

    # Tomamos size/spacing reales del CT para que el scoring cubra toda la imagen.
    ct_img = sitk.ReadImage(str(ct_mhd))
    ct_size = list(ct_img.GetSize())
    ct_spacing_mm = list(ct_img.GetSpacing())

    if source_mode == "point":
        _add_point_proton_source(
            sim=sim,
            name="proton_source_000",
            proton_energy_mev=proton_energy_mev,
            n_events=int(n_events),
            x_mm=0.0,
            y_mm=0.0,
            z_cm=float(source_z_cm),
        )
        print(f"Modo fuente: point (n={int(n_events)})")
    elif source_mode == "beamlet":
        if beamlet_nx < 1 or beamlet_ny < 1:
            raise ValueError("beamlet-nx y beamlet-ny deben ser >= 1")

        spots = _make_beamlet_grid(beamlet_nx, beamlet_ny, beamlet_pitch_mm)
        n_spots = len(spots)
        base = int(n_events) // n_spots
        rem = int(n_events) % n_spots

        for i, (sx, sy) in enumerate(spots):
            n_i = base + (1 if i < rem else 0)
            if n_i <= 0:
                continue
            _add_point_proton_source(
                sim=sim,
                name=f"proton_spot_{i:03d}",
                proton_energy_mev=proton_energy_mev,
                n_events=n_i,
                x_mm=float(sx),
                y_mm=float(sy),
                z_cm=float(source_z_cm),
            )

        print(
            "Modo fuente: beamlet "
            f"({beamlet_nx}x{beamlet_ny}, pitch={beamlet_pitch_mm} mm, spots={n_spots}, n_total={int(n_events)})"
        )
    else:
        raise ValueError("source-mode debe ser 'point' o 'beamlet'")

    # Tally de dosis básico para verificar la geometría.
    dose = sim.add_actor("DoseActor", "dose")
    dose.attached_to = "patient"
    dose.size = [int(ct_size[0]), int(ct_size[1]), int(ct_size[2])]
    dose.spacing = [ct_spacing_mm[0] * mm, ct_spacing_mm[1] * mm, ct_spacing_mm[2] * mm]
    dose.output_filename = "dose_voxelized_ct.mhd"
    dose.write_to_disk = True

    return sim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenGATE: experimento con CT voxelizada")
    parser.add_argument("--ct-mhd", type=Path, required=True, help="CT en formato .mhd")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/gate_ct_test"))
    parser.add_argument("--energy-mev", type=float, default=150.0)
    parser.add_argument("--n-events", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source-mode", type=str, default="point", choices=["point", "beamlet"])
    parser.add_argument("--beamlet-nx", type=int, default=5)
    parser.add_argument("--beamlet-ny", type=int, default=5)
    parser.add_argument("--beamlet-pitch-mm", type=float, default=6.0)
    parser.add_argument("--source-z-cm", type=float, default=-30.0)
    parser.add_argument("--hu-map-json", type=Path, default=None, help="Archivo JSON con mapeo HU->material")
    parser.add_argument(
        "--event-modulo",
        type=int,
        default=100000,
        help="Imprime progreso Geant4 cada N eventos (0 desactiva)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Solo construye la simulación, no la ejecuta")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sim = build_simulation(
        ct_mhd=args.ct_mhd,
        output_dir=args.output_dir,
        proton_energy_mev=args.energy_mev,
        n_events=args.n_events,
        seed=args.seed,
        source_mode=args.source_mode,
        beamlet_nx=args.beamlet_nx,
        beamlet_ny=args.beamlet_ny,
        beamlet_pitch_mm=args.beamlet_pitch_mm,
        source_z_cm=args.source_z_cm,
        hu_map_json=args.hu_map_json,
        event_modulo=args.event_modulo,
    )

    if args.dry_run:
        print("Simulación construida correctamente (dry-run).")
    else:
        sim.run()
        print(f"Simulación finalizada. Resultados en: {args.output_dir}")
