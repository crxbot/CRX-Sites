#!/usr/bin/env python3
"""Alle DWD-Einzelstandorte: laden -> nach uint8-dBZ wandeln -> gzip -> R2 + metadata.json je Standort.

R2-Layout:
    regenradar-sites/<code>/metadata.json
    regenradar-sites/<code>/dbz_<code>_<YYYYMMDD>_<HHMM>      (gzip, max. MAX_KEEP Stueck, UTC)

Aufruf:  python scripts/sites_radar.py            (alle Standorte aus scripts/sites.json)
         python scripts/sites_radar.py umd mem    (nur diese)
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
MAX_KEEP = int(os.environ.get("SITES_MAX_KEEP", "6"))
WAIT_SECONDS = int(os.environ.get("SITES_WAIT_SECONDS", "360"))
SITES_FILE = Path(__file__).with_name("sites.json")


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
        "encoding": "gzip",
        "unit": "dBZ",
        "width": w,
        "height": h,
        "row_order": "north-to-south",
        "dbz_min": sn.DBZ_MIN,
        "dbz_step": sn.DBZ_STEP,
        "nodata": [0, sn.NODATA_CODE],
        "projdef": projdef,
        "pixel_size_m": xs,
        "ll_x": ll_x,
        "ll_y": ll_y,
        "extent_native": [ll_x, ll_y, round(ll_x + w * xs, 3), round(ll_y + h * ys, 3)],
    }
    return codes, grid


def download(site: dict, slot: datetime, dest: Path, deadline: float) -> None:
    stamp = slot.strftime("%Y%m%d%H%M%S")
    name = f"rab02-lm_{site['wmo']}-{stamp}-de{site['code']}-hd5"
    url = f"{BASE_URL}/{site['code']}/{name}"
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


def get_existing(s3, bucket: str, key: str) -> dict:
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        return json.loads(obj["Body"].read())
    except ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return {}
        raise
    except json.JSONDecodeError:
        return {}


def process_site(s3, bucket: str, site: dict, slot: datetime, deadline: float) -> str:
    code = site["code"]
    ts = slot.strftime("%Y%m%d_%H%M")
    prefix = f"{R2_PREFIX}/{code}"

    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "in.hd5"
        download(site, slot, src, deadline)
        codes, grid = convert(src)

    blob = gzip.compress(codes.tobytes(), 9, mtime=0)
    fname = f"dbz_{code}_{ts}"

    existing = get_existing(s3, bucket, f"{prefix}/metadata.json")
    all_ts = sorted(set(existing.get("timesteps", [])) | {ts})
    kept, dropped = all_ts[-MAX_KEEP:], all_ts[:-MAX_KEEP]

    meta = {
        "site": code,
        "wmo": site["wmo"],
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "latest": kept[-1],
        "num_steps": len(kept),
        "timesteps": kept,
        "file_pattern": f"dbz_{code}_{{timestep}}",
        "grid": grid,
    }

    # Reihenfolge: erst Binaerdatei, dann metadata (sonst zeigt sie auf Nicht-Existentes), dann aufraeumen
    s3.put_object(
        Bucket=bucket, Key=f"{prefix}/{fname}", Body=blob,
        ContentType="application/gzip",
        CacheControl="public, max-age=31536000, immutable",
    )
    s3.put_object(
        Bucket=bucket, Key=f"{prefix}/metadata.json",
        Body=json.dumps(meta, indent=2).encode("utf-8"),
        ContentType="application/json",
        CacheControl="public, max-age=30",
    )
    for old in dropped:
        s3.delete_object(Bucket=bucket, Key=f"{prefix}/dbz_{code}_{old}")

    return f"[{code}] OK {fname} ({len(blob)} B gz), {len(kept)} Steps, {len(dropped)} geloescht"


def main() -> None:
    sites = json.loads(SITES_FILE.read_text(encoding="utf-8"))
    only = set(sys.argv[1:])
    if only:
        sites = [s for s in sites if s["code"] in only]

    now = int(time.time())
    slot = datetime.fromtimestamp(now // 300 * 300, tz=timezone.utc)
    deadline = time.time() + WAIT_SECONDS
    print(f"Slot {slot:%Y-%m-%d %H:%M} UTC, {len(sites)} Standorte")

    s3 = boto3.client(
        "s3",
        endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )
    bucket = os.environ["R2_BUCKET"]

    failed = []
    with ThreadPoolExecutor(max_workers=len(sites) or 1) as pool:
        futs = {pool.submit(process_site, s3, bucket, s, slot, deadline): s["code"] for s in sites}
        for fut in as_completed(futs):
            code = futs[fut]
            try:
                print(fut.result())
            except Exception as e:  # ein Standort darf die anderen nicht killen
                print(f"[{code}] FEHLER: {e}")
                failed.append(code)

    if failed:
        print(f"::error::Fehlgeschlagen: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()