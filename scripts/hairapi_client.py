"""Desktop test client for the hairstyle API (server/hairapi). Image in, image out.

The API listens on 127.0.0.1:8090 on <host>; open a tunnel first:   ssh -N -L 8090:127.0.0.1:8090 <host>
The key is read from the environment (HAIRAPI_KEY) or --key; it is never printed.

  python scripts/hairapi_client.py catalog [--gender female]
  python scripts/hairapi_client.py edit --photo data/hairstyle_set/photo_a.jpg --style f_short_french_bob [--out results/api]
"""
import argparse
import os
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["health", "catalog", "edit"])
    ap.add_argument("--url", default="http://127.0.0.1:8090")
    ap.add_argument("--key", default=os.environ.get("HAIRAPI_KEY", ""))
    ap.add_argument("--gender")
    ap.add_argument("--photo")
    ap.add_argument("--style")
    ap.add_argument("--out", default=str(ROOT / "results" / "api"))
    ap.add_argument("--timeout", type=float, default=120)
    a = ap.parse_args()
    headers = {"X-API-Key": a.key}

    if a.action == "health":
        r = requests.get(f"{a.url}/healthz", timeout=10)
        print(r.status_code, r.text)
    elif a.action == "catalog":
        r = requests.get(f"{a.url}/v1/catalog", params={"gender": a.gender} if a.gender else None, headers=headers, timeout=10)
        print(r.status_code, r.text)
    else:
        if not (a.photo and a.style):
            sys.exit("edit needs --photo and --style")
        t0 = time.time()
        with open(a.photo, "rb") as f:
            r = requests.post(f"{a.url}/v1/hairstyle", headers=headers, data={"style_id": a.style},
                              files={"photo": (Path(a.photo).name, f, "image/jpeg")}, timeout=a.timeout)
        wall = time.time() - t0
        if r.status_code != 200:
            print(r.status_code, r.text)
            sys.exit(1)
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        dst = out / f"{Path(a.photo).stem}__{a.style}.jpg"
        dst.write_bytes(r.content)
        print(f"{dst}  wall {wall:.1f}s  server {r.headers.get('X-Timings')}  warnings {r.headers.get('X-Warnings', '-')}")


if __name__ == "__main__":
    main()
