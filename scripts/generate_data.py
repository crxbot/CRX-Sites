#!/usr/bin/env python3
"""Alle DWD-Einzelstandorte: laden -> uint8-dBZ -> gzip -> R2 (nur das jeweils neueste Bild).

R2-Layout:
    sites/radar/metadata.json            (alle Radare: status, timestamp, grid)
    sites/radar/dbz_<code>_latest        (gzip, wird pro Lauf ueberschrieben)

status:  ok           Datei geladen, gewandelt, hochgeladen
         unavailable  DWD hat die Datei nicht geliefert (404 bis Timeout) -> altes Bild + alter timestamp bleiben
         invalid      Datei kaputt / nicht lesbar / fast nur "kein Datum" -> altes Bild + alter timestamp bleiben

Aufruf:  python scripts/sites_radar.py            (alle Standorte aus scripts/sites.json)
         python scripts/sites_radar.py umd mem    (nur diese; die anderen Eintraege bleiben unveraendert)
"""
import gzip
import json
import os
import sys
import tempfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import boto3
import h5py
from botocore.exceptions import ClientError
from pyproj import Transformer

import site_native as sn  # liegt im selben Ordner (scripts/)

BASE_URL = "https://opendata.dwd.de/weather/radar/sites/lmax"
R2_PREFIX = os.environ.get("SITES_R2_PREFIX", "sites/radar")
WAIT_SECONDS = int(os.environ.get("SITES_WAIT_SECONDS", "180"))          # wartet auf die Datei
RETRY_WAIT_SECONDS = int(os.environ.get("SITES_RETRY_WAIT_SECONDS", "20"))  # bei Radaren, die zuletzt nicht ok waren
MAX_NODATA = float(os.environ.get("SITES_MAX_NODATA", "0.97"))           # Anteil Code 255, ab dem das Bild "invalid" ist
SITES_FILE = Path(__file__).with_name("sites.json")
CACHE = "public, max-age=30"


def convert(path: Path):
    """HDF5 -> (uint8-Codes, grid-dict). Gleiche Logik wie main() in site_native.py."""
    with h5py.File(path, "r") as f:
        ds = sn.find_dbz_dataset(f)
        dbz = sn.read_dbz(ds)
        polar = sn.find_polar_info(f, ds)
        if polar:
            dbz, projdef = sn.polar_to_grid(dbz, polar)
            h, w = dbz.shape
            xs = ys = sn.GRID_PS
            ll_x = ll_y = -sn.GRID_N * sn.GRID_PS / 2
        else:
            wh = ds.parent.parent.get("where")
            if wh is None or "projdef" not in wh.attrs:
                wh = sn.find_where(f)
            projdef = sn._s(wh.attrs["projdef"])
            h, w = dbz.shape
            xs = float(sn._x(wh.attrs.get("xscale", sn.DEFAULT_SCALE_M)))
            ys = float(sn._x(wh.attrs.get("yscale", sn.DEFAULT_SCALE_M)))
            if "LL_lon" in wh.attrs and "LL_lat" in wh.attrs:
                to_proj = Transformer.from_crs("EPSG:4326", projdef, always_xy=True)
                ll_x, ll_y = to_proj.transform(
                    float(sn._x(wh.attrs["LL_lon"])), float(sn._x(wh.attrs["LL_lat"]))
                )
            else:
                ll_x, ll_y = -w * xs / 2, -h * ys / 2

    codes = sn.encode_dbz(dbz)
    ll_x, ll_y = round(ll_x, 3), round(ll_y, 3)
    grid = {
        "format": "raw-uint8",
        "width": w,
        "height": h,
        "row_order": "north-to-south",
        "projdef": projdef,
        "pixel_size_m": xs,
        "ll_x": ll_x,
        "ll_y": ll_y,
        "extent_native": [ll_x, ll_y, round(ll_x + w * xs, 3), round(ll_y + h * ys, 3)],
        "nodata": sn.NODATA_CODE,
    }
    return codes, grid


def download(site: dict, slot: datetime, dest: Path, wait: int) -> None:
    """Laedt die Slot-Datei; 404 -> alle 5 s erneut, bis `wait` Sekunden um sind (dann TimeoutError)."""
    stamp = slot.strftime("%Y%m%d%H%M%S")
    name = f"rab02-lm_{site['wmo']}-{stamp}-de{site['code']}-hd5"
    url = f"{BASE_URL}/{site['code']}/{name}"
    deadline = time.time() + wait
    attempt = 0
    while True:
        attempt += 1
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                data = r.read()
            if data:
                dest.write_bytes(data)
                print(f"[{site['code']}] geladen (Versuch {attempt}): {name}")
                return
        except urllib.error.HTTPError as e:
            if e.code != 404:
                print(f"[{site['code']}] HTTP {e.code}")
        except OSError as e:
            print(f"[{site['code']}] Netzfehler: {e}")
        if time.time() > deadline:
            raise TimeoutError(f"{name} nicht verfuegbar")
        time.sleep(5)


def get_json(s3, bucket: str, key: str) -> dict:
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        return json.loads(obj["Body"].read())
    except ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return {}
        raise
    except json.JSONDecodeError:
        return {}


def process_site(s3, bucket: str, site: dict, slot: datetime, wait: int) -> dict:
    """Gibt das Update fuer den Meta-Eintrag zurueck: {"status": ..., (nur bei ok:) "timestamp", "grid"}."""
    code = site["code"]
    ts = slot.strftime("%Y%m%d_%H%M")

    try:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "in.hd5"
            download(site, slot, src, wait)
            codes, grid = convert(src)
    except TimeoutError as e:
        print(f"[{code}] unavailable: {e}")
        return {"status": "unavailable"}
    except Exception as e:
        print(f"[{code}] invalid (Lesefehler): {e}")
        return {"status": "invalid"}

    nodata_share = float((codes == sn.NODATA_CODE).mean())
    print(f"[{code}] Anteil nodata: {nodata_share:.1%}")
    if nodata_share > MAX_NODATA:
        print(f"[{code}] invalid: {nodata_share:.1%} > {MAX_NODATA:.0%} nodata")
        return {"status": "invalid"}

    s3.put_object(
        Bucket=bucket,
        Key=f"{R2_PREFIX}/dbz_{code}_latest",
        Body=gzip.compress(codes.tobytes(), 9, mtime=0),
        ContentType="application/gzip",
        CacheControl=CACHE,
    )
    print(f"[{code}] OK {ts}")
    return {"status": "ok", "timestamp": ts, "grid": grid}


def main() -> None:
    sites = json.loads(SITES_FILE.read_text(encoding="utf-8"))
    only = set(sys.argv[1:])
    run_sites = [s for s in sites if not only or s["code"] in only]

    slot = datetime.fromtimestamp(int(time.time()) // 300 * 300, tz=timezone.utc)
    print(f"Slot {slot:%Y-%m-%d %H:%M} UTC, {len(run_sites)} Standorte")

    s3 = boto3.client(
        "s3",
        endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )
    bucket = os.environ["R2_BUCKET"]
    meta_key = f"{R2_PREFIX}/metadata.json"

    old_radars = get_json(s3, bucket, meta_key).get("radars", {})
    radars = dict(old_radars)

    errors = []
    with ThreadPoolExecutor(max_workers=len(run_sites) or 1) as pool:
        futs = {}
        for s in run_sites:
            prev = old_radars.get(s["code"], {})
            # Radar war zuletzt schon gestoert -> nicht wieder 3 Minuten warten
            wait = WAIT_SECONDS if prev.get("status", "ok") == "ok" else RETRY_WAIT_SECONDS
            futs[pool.submit(process_site, s3, bucket, s, slot, wait)] = s

        for fut in as_completed(futs):
            s = futs[fut]
            code = s["code"]
            try:
                update = fut.result()
            except Exception as e:  # z.B. R2-Upload fehlgeschlagen: Eintrag unveraendert lassen
                print(f"[{code}] FEHLER: {e}")
                errors.append(code)
                continue
            prev = old_radars.get(code, {})
            radars[code] = {
                "wmo": s["wmo"],
                "status": update["status"],
                "timestamp": update.get("timestamp", prev.get("timestamp")),
                "file": f"dbz_{code}_latest",
                "grid": update.get("grid", prev.get("grid")),
            }

    # Reihenfolge wie in sites.json
    order = [s["code"] for s in sites]
    radars = {c: radars[c] for c in order if c in radars} | {c: v for c, v in radars.items() if c not in order}

    meta = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
        "radars": radars,
    }
    s3.put_object(
        Bucket=bucket, Key=meta_key,
        Body=json.dumps(meta, indent=2, ensure_ascii=False).encode("utf-8"),
        ContentType="application/json",
        CacheControl=CACHE,
    )

    summary = ", ".join(f"{c}={v['status']}" for c, v in radars.items())
    print(f"metadata.json geschrieben: {summary}")
    bad = [c for c, v in radars.items() if v["status"] != "ok"]
    if bad:
        print(f"::warning::Radare nicht ok: {', '.join(bad)}")
    if errors:
        print(f"::error::Upload-Fehler: {', '.join(errors)}")
        sys.exit(1)


if __name__ == "__main__":
    main()