#!/usr/bin/env python3
"""Generate the MHE-IK optimal control problem ahead of time, for either backend.

Both backends bake the human's forward kinematics into generated code, so both
used to regenerate whenever a new person was calibrated -- 20-40 s, in the middle
of a live session, inside whichever process happened to need it. The OCP is now
parameterized by the subject's geometry (segment lengths and marker offsets), so
one generated artefact serves everybody and this script produces it once:

    python3 scripts/python/core/run_ocp_codegen.py                # both backends
    python3 scripts/python/core/run_ocp_codegen.py --backend acados
    python3 scripts/python/core/run_ocp_codegen.py --check        # verify only

Artefacts land under ``<repo>/ocp/<backend>/`` (override with RTCOSMIK_OCP_DIR),
each beside an ``ocp_manifest.json`` recording exactly what it was generated
from. The pipeline refuses to load an artefact whose manifest does not match the
running configuration, because the alternative is a solver that runs happily on
the wrong skeleton.

Run this after changing anything the OCP bakes in: the tracked marker set, N,
the URDF, or the marker-to-joint mapping -- plus dt for acados, which bakes it
into the discrete dynamics (fatrop takes dt as a runtime input, so its artefact
survives a framerate change). ``--check`` tells you whether you need to, and is
what CI should call.
"""
import argparse
import os
import sys
import time
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[3] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rtcosmik.config_loader import settings
from rtcosmik.ik import ocp_model
from rtcosmik.ik.ik import RT_SWIKA_ACADOS, RT_SWIKA_FATROP


def structural_model():
    """The model to generate from: right structure, placeholder numbers."""
    model = ocp_model.build_structural_model(settings)
    keys = list(settings.keys_to_track_list)
    missing = [k for k in keys if not model.existFrame(k)]
    if missing:
        raise SystemExit(
            f"markers {missing} are in settings.keys_to_track_list but are not "
            "registered frames. The generated OCP has a fixed residual size, so "
            "the tracked set cannot vary at runtime -- fix the marker mapping or "
            "the tracked list before generating.")
    return model, keys


def current_description(model, keys, backend, profile):
    """What this backend bakes in.

    fatrop takes dt as a runtime input, so its artefact stays valid across a
    framerate change; acados bakes dt into the discrete dynamics and does not.
    """
    _, _, _, joint_ids, frame_ids = ocp_model.parameterize(model, keys)
    dt = settings.dt if backend == "acados" else None
    cls = RT_SWIKA_ACADOS if backend == "acados" else RT_SWIKA_FATROP
    options = {**cls.DEFAULT_SOLVER_OPTIONS,
               **ocp_model.profile_options(backend, profile)}
    return ocp_model.describe(model, keys, settings.N, dt, True,
                              joint_ids, frame_ids, solver_options=options)


def do_check(backends, profiles, model, keys):
    """Report whether each backend/profile artefact matches the configuration."""
    stale = []
    for backend in backends:
        for profile in profiles:
            directory = ocp_model.backend_dir(backend, settings, profile)
            try:
                ocp_model.check_manifest(
                    directory, current_description(model, keys, backend, profile),
                    backend)
                print(f"  {backend}/{profile}: up to date")
            except RuntimeError as exc:
                stale.append(f"{backend}/{profile}")
                print(f"  {backend}/{profile}: STALE")
                for line in str(exc).splitlines()[1:]:
                    print(f"  {line}")
    return stale


def generate_fatrop(model, keys, profile):
    directory = ocp_model.backend_dir("fatrop", settings, profile)
    options = {**RT_SWIKA_FATROP.DEFAULT_SOLVER_OPTIONS,
               **ocp_model.profile_options("fatrop", profile)}
    os.makedirs(directory, exist_ok=True)
    print(f"  building the OCP ({model.nq} dof, {len(keys)} markers, N={settings.N}) ...")
    started = time.time()
    solver = RT_SWIKA_FATROP(model, keys, settings.N,
                             export_dir=directory,
                             solver_options=options)
    print(f"    {solver.n_params} geometry parameters, {time.time()-started:.1f} s")
    print("  compiling ...")
    started = time.time()
    library = solver.compile_Ccode(directory)
    print(f"    {library} ({os.path.getsize(library)/1e6:.1f} MB, "
          f"{time.time()-started:.1f} s)")
    ocp_model.write_manifest(directory, solver.describe(), "fatrop",
                             extra={"library": os.path.basename(library)})
    return directory


def generate_acados(model, keys, profile):
    directory = ocp_model.backend_dir("acados", settings, profile)
    options = {**RT_SWIKA_ACADOS.DEFAULT_SOLVER_OPTIONS,
               **ocp_model.profile_options("acados", profile)}
    os.makedirs(directory, exist_ok=True)
    print(f"  generating and compiling ({model.nq} dof, {len(keys)} markers, "
          f"N={settings.N}, dt={settings.dt}) ...")
    started = time.time()
    solver = RT_SWIKA_ACADOS(model, keys, settings.N, settings.dt, build=True,
                             export_dir=directory,
                             acados_source_dir=settings.acados_source_dir,
                             solver_options=options)
    print(f"    {solver.n_params} geometry parameters, {time.time()-started:.1f} s")
    return directory


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=["fatrop", "acados", "both"],
                        default="both", help="which OCP to generate (default both)")
    parser.add_argument("--profile", choices=["realtime", "accurate", "both"],
                        default="both",
                        help="which speed/accuracy profile to generate "
                             "(default both; each is a separate artefact)")
    parser.add_argument("--check", action="store_true",
                        help="only report whether the artefacts are up to date; "
                             "exits non-zero if any is stale")
    args = parser.parse_args()

    backends = ["fatrop", "acados"] if args.backend == "both" else [args.backend]
    profiles = (["realtime", "accurate"] if args.profile == "both"
                else [args.profile])

    print("Structural model: topology only, no subject data needed")
    model, keys = structural_model()
    for backend in backends:
        for profile in profiles:
            digest = ocp_model.fingerprint(
                current_description(model, keys, backend, profile))
            print(f"  {backend}/{profile}: {digest[:16]}... "
                  f"{ocp_model.profile_options(backend, profile)}")
    print(f"  artefacts under {ocp_model.artifact_root(settings)}\n")

    if args.check:
        print("Checking:")
        stale = do_check(backends, profiles, model, keys)
        if stale:
            print(f"\nStale: {', '.join(stale)}. Regenerate before running.")
            return 1
        print("\nAll generated OCPs match the current configuration.")
        return 0

    for backend in backends:
        for profile in profiles:
            print(f"{backend}/{profile}:")
            try:
                directory = (generate_fatrop(model, keys, profile)
                             if backend == "fatrop"
                             else generate_acados(model, keys, profile))
            except ImportError as exc:
                print(f"  skipped: {exc}")
                continue
            print(f"  manifest written to {directory}\n")

    print("Done. The pipeline will reuse these; it refuses to load one whose "
          "manifest\nno longer matches the configuration, so re-run this after "
          "changing N, dt,\nthe tracked marker set, or the model.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
