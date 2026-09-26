#!/usr/bin/env python3
"""Assemble the final submission zip in the exact structure required by the challenge.

  python build_package.py --team <team_name> --outputs output/sub_v1 [--validate]

Result: <team_name>_submission.zip
  output/matching_results.tsv, output/candidate_pairs.tsv
  code/business_entity_resolution/{src/, README.md, requirements.txt}
  Documentation_template.md
src/ contains the library (src/ber), the stage files (modal_app), run_pipeline.py and configs.
Stdlib only.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", required=True)
    ap.add_argument("--outputs", required=True, help="folder holding matching_results.tsv and candidate_pairs.tsv")
    ap.add_argument("--validate", action="store_true", help="run the official validator first")
    args = ap.parse_args()
    m = os.path.join(args.outputs, "matching_results.tsv")
    c = os.path.join(args.outputs, "candidate_pairs.tsv")
    for p in (m, c):
        if not os.path.isfile(p):
            print(f"missing {p}")
            return 1
    prov = os.path.join(args.outputs, "predict_report.json")
    if not os.path.isfile(prov):
        print(f"WARNING: no {prov}: cannot confirm these outputs were produced by the packaged code/config")
    else:
        with open(prov) as fh:
            pr = json.load(fh)
        print("provenance:", {k: pr.get(k) for k in ("model", "feat", "rule", "validator_exit")})
    if args.validate:
        r = subprocess.run([sys.executable, os.path.join(HERE, "student_resource", "utils", "validate_submission.py"),
                            "--matching", m, "--candidate", c, "--test-dir",
                            os.path.join(HERE, "student_resource", "dataset", "test")])
        if r.returncode != 0:
            print("validator failed; not packaging")
            return 1
    stage = os.path.join(HERE, "package_build", f"{args.team}_submission")
    shutil.rmtree(os.path.dirname(stage), ignore_errors=True)
    code = os.path.join(stage, "code", "business_entity_resolution")
    os.makedirs(os.path.join(stage, "output"))
    os.makedirs(os.path.join(code, "src"))
    shutil.copy(m, os.path.join(stage, "output", "matching_results.tsv"))
    shutil.copy(c, os.path.join(stage, "output", "candidate_pairs.tsv"))
    ign = shutil.ignore_patterns("__pycache__", "*.pyc")
    shutil.copytree(os.path.join(HERE, "src", "ber"), os.path.join(code, "src", "ber"), ignore=ign)
    shutil.copytree(os.path.join(HERE, "modal_app"), os.path.join(code, "src", "modal_app"), ignore=ign)
    shutil.copytree(os.path.join(HERE, "configs"), os.path.join(code, "src", "configs"))
    shutil.copytree(os.path.join(HERE, "tests"), os.path.join(code, "src", "tests"), ignore=ign)
    shutil.copy(os.path.join(HERE, "run_pipeline.py"), os.path.join(code, "src", "run_pipeline.py"))
    shutil.copy(os.path.join(HERE, "README.md"), os.path.join(code, "README.md"))
    shutil.copy(os.path.join(HERE, "requirements.txt"), os.path.join(code, "requirements.txt"))
    shutil.copy(os.path.join(HERE, "Documentation_template.md"), os.path.join(stage, "Documentation_template.md"))
    zpath = os.path.join(HERE, f"{args.team}_submission.zip")
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for dp, _, files in os.walk(stage):
            for f in files:
                full = os.path.join(dp, f)
                z.write(full, os.path.relpath(full, stage))  # spec: output/, code/, Documentation_template.md at root
    print(f"wrote {zpath} ({os.path.getsize(zpath) / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
