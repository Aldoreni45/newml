"""Model I/O helpers: resolve the feature list a saved LightGBM model expects."""
from __future__ import annotations

import json
import os
import re
from typing import List


def model_features(model, models_dir: str, tag: str) -> List[str]:
    """Feature names for a saved booster.

    Models trained from bare numpy arrays carry names Column_0..N. Their true feature order
    is stored in <models_dir>/<base_tag>_features.json (base tag = tag without _f<k> suffix).
    """
    names = model.feature_name()
    if names and not all(n.startswith("Column_") for n in names):
        return names
    for cand in (tag, re.sub(r"_f\d+$", "", tag)):
        p = os.path.join(models_dir, f"{cand}_features.json")
        if os.path.exists(p):
            with open(p) as fh:
                feats = json.load(fh)
            if len(feats) != len(names):
                raise ValueError(f"feature list length {len(feats)} != model {len(names)} for {tag}")
            return feats
    raise FileNotFoundError(f"no feature list for {tag}")
