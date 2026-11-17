"""Write a catalog.json for a folder of candidate reference images, so the hairstyle API can run them without touching the live catalogue.

  python scripts/make_test_catalog.py nano          # data/hairapi_refs/incoming/nano/*.jpg -> incoming/nano/catalog.json

File names are <style_id>.jpg or <style_id>__<n>.jpg (several candidates per style); the style id is the part before "__".
Each file becomes one catalogue entry (id = file stem) with the gender/length/name/hair_prompt of its style (styles.json).
Then:  scripts\\hs push ; scripts\\hs api-test-start -g 1 --refs-dir refs_nano
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REFS = ROOT / "data" / "hairapi_refs"


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    folder = REFS / "incoming" / sys.argv[1]
    styles = {s["id"]: s for s in json.loads((REFS / "styles.json").read_text(encoding="utf-8"))["styles"]}
    entries = []
    for p in sorted(folder.glob("*.jpg")):
        base = p.stem.split("__")[0]
        if base not in styles:
            sys.exit(f"{p.name}: unknown style id {base!r} (see styles.json)")
        s = styles[base]
        entries.append({"id": p.stem, "gender": s["gender"], "length": s["length"], "name": s["name"],
                        "hair_prompt": s["hair_prompt"], "reference": p.name})
    if not entries:
        sys.exit(f"no .jpg files in {folder}")
    out = folder / "catalog.json"
    out.write_text(json.dumps({"version": 1, "_note": f"test catalogue from {folder.name}", "styles": entries}, indent=2) + "\n",
                   encoding="utf-8")
    print(f"{out}: {len(entries)} styles: " + ", ".join(e["id"] for e in entries))


if __name__ == "__main__":
    main()
