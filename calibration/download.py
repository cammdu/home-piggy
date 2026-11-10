"""Download every registered series (cached) and write calibration/output/manifest.json."""
from __future__ import annotations

import json
import sys
import time

from .sources import REGISTRY, ROOT, fetch


def main(refresh: bool = False) -> None:
    manifest = {}
    for name in REGISTRY:
        t0 = time.time()
        body, meta = fetch(name, refresh=refresh)
        manifest[name] = {k: meta[k] for k in ("provider", "key", "url", "description", "units",
                                               "retrieved_utc", "sha256", "bytes", "licence_note")}
        print(f"{name:14s} {meta['bytes']:>10,d} bytes  {time.time() - t0:5.1f}s  {meta['retrieved_utc']}", flush=True)
        time.sleep(2)   # be polite to the providers
    out = ROOT / "calibration" / "output" / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2))
    print("manifest written:", out)


if __name__ == "__main__":
    main(refresh="--refresh" in sys.argv)
