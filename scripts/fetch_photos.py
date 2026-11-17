"""Fetch openly licensed portrait photos from Wikimedia Commons for the Phase 0 test set.

Photos go to data/photos/ (gitignored). data/manifest.csv keeps author, license and source URL
so attribution survives even though the images are not committed.
"""
import csv
import re
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
PHOTO_DIR = ROOT / "data" / "photos_wiki"
MANIFEST = ROOT / "data" / "manifest_wiki.csv"
API = "https://commons.wikimedia.org/w/api.php"
HEADERS = {"User-Agent": "virtual-try-on-spike/0.1 (personal portfolio project; research use)"}

# (category tag, search text). Tag ends up in the filename and manifest for later grouping.
# Gap-fill only: the Kaggle selfie set already covers straight/wavy frontal hair indoors.
QUERIES = [
    ("curly", "woman portrait curly hair"),
    ("curly", "man portrait curly hair"),
    ("curly", "headshot curly hair 2018"),
    ("coily", "portrait afro hair"),
    ("coily", "portrait natural coily hair"),
    ("braids", "portrait braids hair"),
    ("braids", "portrait locs dreadlocks"),
    ("dyed", "portrait dyed hair colorful"),
    ("dyed", "portrait pink blue hair"),
    ("glasses", "portrait woman glasses hair"),
    ("glasses", "portrait man glasses hair"),
    ("hat", "portrait hat hair"),
    ("hat", "portrait beanie cap woman"),
    ("busy_bg", "portrait street crowd hair"),
]
PER_QUERY = 5
# Titles that signal paintings, drawings, objects, animals or derivative crops rather than real portrait photos.
BLOCK_TITLE = re.compile(
    r"seal|painting|drawing|engrav|illustrat|anime|cat\b|cats\b|mummy|graffiti|nude|google art|wellcome|"
    r"d.rer|WDL|MSS|FindID|sketch|statue|bust|cartoon|comic|poster|advert|cropped|\(page|dpla|"
    r"museum|schaeferin|after |copy of|curtis|ca\.\s?1[0-8]\d\d",
    re.I,
)
OK_LICENSE = re.compile(r"^(cc0|cc[- ]by(-sa)?[- ]\d|public domain|pd)", re.I)
MIN_SIDE = 600


def clean(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text or "").strip()


def search(term: str, limit: int = 30) -> list[dict]:
    params = {
        "action": "query", "format": "json", "generator": "search", "gsrsearch": f"{term} filetype:bitmap",
        "gsrnamespace": 6, "gsrlimit": limit, "prop": "imageinfo",
        "iiprop": "url|size|extmetadata|mime", "iiurlwidth": 1024,
    }
    r = requests.get(API, params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return list(r.json().get("query", {}).get("pages", {}).values())


def main() -> None:
    PHOTO_DIR.mkdir(parents=True, exist_ok=True)
    rows, seen = [], set()
    for tag, term in QUERIES:
        got = 0
        for page in search(term):
            if got >= PER_QUERY:
                break
            info = (page.get("imageinfo") or [{}])[0]
            meta = info.get("extmetadata", {})
            lic = clean(meta.get("LicenseShortName", {}).get("value", ""))
            w, h = info.get("width", 0), info.get("height", 0)
            title = page["title"]
            stem = re.sub(r"[\s_]*\(?\d+\)?$", "", re.sub(r"\.\w+$", "", title))  # "... 01" / "... 02" -> same stem
            if title in seen or stem in seen or info.get("mime") not in ("image/jpeg", "image/png"):
                continue
            if BLOCK_TITLE.search(title):
                continue
            if not OK_LICENSE.match(lic) or min(w, h) < MIN_SIDE or h < w * 0.9:
                continue  # wrong license, too small, or landscape (we want portrait-ish)
            url = info.get("thumburl") or info["url"]
            name = f"{tag}_{len(rows):03d}.jpg"
            img = requests.get(url, headers=HEADERS, timeout=60)
            if img.status_code != 200:
                continue
            (PHOTO_DIR / name).write_bytes(img.content)
            seen.update((title, stem))
            got += 1
            rows.append({
                "file": name, "tag": tag, "title": title, "license": lic,
                "author": clean(meta.get("Artist", {}).get("value", "")),
                "source": info.get("descriptionurl", ""),
            })
            print(f"{name}  {lic}  {title}")
            time.sleep(0.3)  # be polite to Commons
    with MANIFEST.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "tag", "title", "license", "author", "source"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n{len(rows)} photos -> {PHOTO_DIR}\nmanifest -> {MANIFEST}")


if __name__ == "__main__":
    main()
