#!/usr/bin/env python3
"""Alle DWD-Einzelstandorte: laden -> uint8-dBZ -> gzip -> R2 (nur das jeweils neueste Bild).

R2-Layout:
    sites/radar/metadata.json            (alle Radare: status, timestamp, lat/lon, projdef; Grid einmal)
    sites/radar/<code>/latest            (dBZ, gzip, wird pro Lauf ueberschrieben)
    sites/radial/metadata.json           (Radialgeschwindigkeit: status, timestamp, Decodier-Info, Grid wie radar)
    sites/radial/<code>/latest           (km/h, gleiches Raster wie dBZ, gzip)

status:  ok           Datei geladen, gewandelt, hochgeladen
         unavailable  DWD hat die Datei nicht geliefert (404 bis Timeout) -> altes Bild + alter timestamp bleiben
         invalid      Datei kaputt / nicht lesbar / fast nur "kein Datum" -> altes Bild + alter timestamp bleiben

Quelle:  DWD Polar-Sweeps, 0.5 Grad, DBZH:
         https://opendata.dwd.de/weather/radar/sites/sweep_vol_z/<code>/hdf5/filter_simple/
         Alle 5 min eine Datei pro Standort; Minute/Sekunde sind je Standort verschieden (z.B. boo 20:38:01, mem 20:57:52)
         -> es wird die NEUESTE Datei aus der Verzeichnisliste genommen, der Zeitstempel kommt aus dem Dateinamen (UTC).

Aufruf:  python scripts/generate_data.py            (alle Standorte aus scripts/sites.json)
         python scripts/generate_data.py umd mem    (nur diese; die anderen Eintraege bleiben unveraendert)

sites.json:  [{"code": "umd", "wmo": "10356", "name": "Ummendorf", "lat": 52.16, "lon": 11.17}, ...]
             name, lat, lon sind optional; lat/lon dienen nur als Fallback, falls sie nicht in der HDF5-Datei stehen.
"""
import gzip
import json
import os
import re
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

BASE = "https://opendata.dwd.de/weather/radar/sites"   # + /<dir>/<code>/hdf5/<filter>/

# Produkte: Reflektivitaet (wie bisher) und Radialgeschwindigkeit.
#   prefix     = R2-Ordner (siehe R2_PREFIX / RADIAL_PREFIX), file = Dateiname je Standort
#   nodata_chk = Anteil-nodata-Pruefung (bei Geschwindigkeit aus: bei Klarluft ist fast alles "kein Echo")
PRODUCTS = {
    "dbz":  {"dir": "sweep_vol_z", "name": "dbzh",  "qty": "DBZ",  "unit": "dBZ",
             "encode": sn.encode_dbz,  "file": "latest", "prefix": "radar",  "nodata_chk": True},
    "vrad": {"dir": "sweep_vol_v", "name": "vradh", "qty": "VRAD", "unit": "m/s",
             "encode": sn.encode_vrad, "file": "latest", "prefix": "radial", "nodata_chk": False},
}
ELEV = os.environ.get("SITES_ELEV", "05")       # Elevations-Kennung im Dateinamen: 05 = 0.5 Grad
FILTER = os.environ.get("SITES_FILTER", "filter_polarimetric")   # oder "filter_simple"
MAX_AGE_MIN = int(os.environ.get("SITES_MAX_AGE_MIN", "4"))   # aelter als das gilt als "nicht verfuegbar"
GRACE_SECONDS = int(os.environ.get("SITES_GRACE_SECONDS", "30"))   # Nachfrist fuer Nachzuegler, sobald ein Standort fertig geladen hat
POLL_SECONDS = 10                               # Abstand der Verzeichnis-Abfragen beim Warten
R2_PREFIX = os.environ.get("SITES_R2_PREFIX", "sites/radar")
RADIAL_PREFIX = os.environ.get("SITES_RADIAL_PREFIX", "sites/radial")
PREFIXES = {"radar": R2_PREFIX, "radial": RADIAL_PREFIX}
WAIT_SECONDS = int(os.environ.get("SITES_WAIT_SECONDS", "120"))             # wartet auf die Datei
RETRY_WAIT_SECONDS = int(os.environ.get("SITES_RETRY_WAIT_SECONDS", "20"))  # bei Radaren, die zuletzt nicht ok waren
MAX_NODATA = float(os.environ.get("SITES_MAX_NODATA", "0.97"))              # Anteil Code 255, ab dem das Bild "invalid" ist
SITES_FILE = Path(__file__).with_name("sites.json")
CACHE = "public, max-age=30"

# Diese Grid-Felder stehen einmal oben in der metadata.json (inkl. gemeinsamer projdef).
# Pro Radar stehen nur ll_x, ll_y und extent_native (in dieser gemeinsamen Projektion).
SHARED_KEYS = ("format", "width", "height", "row_order", "projdef", "pixel_size_m", "nodata")

FALSE_ORIGIN = re.compile(r"\s\+(x_0|y_0)=(-?[\d.]+(?:[eE][-+]?\d+)?)")


def normalize(grid: dict) -> dict:
    """Entfernt +x_0/+y_0 aus der projdef und rechnet sie in ll_x/ll_y/extent_native ein.

    Die DWD-Dateien nutzen dieselbe stereographische Projektion, verschieben aber den Ursprung
    (false easting/northing) pro Standort. Ohne diese Verschiebung ist die projdef bei allen gleich;
    ein Punkt hat dann x_true = x_site - x_0, y_true = y_site - y_0.
    """
    x0 = y0 = 0.0
    for k, v in FALSE_ORIGIN.findall(grid["projdef"]):
        if k.lower() == "x_0":
            x0 = float(v)
        else:
            y0 = float(v)
    ex = grid["extent_native"]
    g = dict(grid)
    g["projdef"] = FALSE_ORIGIN.sub("", grid["projdef"]).strip()
    g["ll_x"] = round(grid["ll_x"] - x0, 3)
    g["ll_y"] = round(grid["ll_y"] - y0, 3)
    g["extent_native"] = [round(ex[0] - x0, 3), round(ex[1] - y0, 3),
                          round(ex[2] - x0, 3), round(ex[3] - y0, 3)]
    return g


def expand(top: dict, r: dict):
    """Volles, normalisiertes Grid eines gespeicherten Radar-Eintrags (neues UND altes Meta-Format) oder None."""
    g = {**top, **(r.get("grid") or {})}
    for k in ("ll_x", "ll_y", "extent_native", "projdef"):
        if r.get(k) is not None:
            g[k] = r[k]
    if not all(k in g for k in ("projdef", "ll_x", "ll_y", "extent_native", "width", "height")):
        return None
    return normalize(g)


def find_location(f: h5py.File):
    """lat/lon des Radarstandorts aus der HDF5-Datei, sonst (None, None)."""
    root = f.get("where")
    grp = root if (root is not None and "lat" in root.attrs and "lon" in root.attrs) else None
    if grp is None:
        found = []

        def visitor(name, obj):
            if not found and isinstance(obj, h5py.Group) and "lat" in obj.attrs and "lon" in obj.attrs:
                found.append(obj)

        f.visititems(visitor)
        grp = found[0] if found else None
    if grp is None:
        return None, None
    return round(float(sn._x(grp.attrs["lat"])), 5), round(float(sn._x(grp.attrs["lon"])), 5)


def convert(path: Path, prod: dict):
    """HDF5 -> (uint8-Codes, volles Grid-dict inkl. projdef, (lat, lon)). Logik wie main() in site_native.py."""
    with h5py.File(path, "r") as f:
        ds = sn.find_dbz_dataset(f, prod["qty"])
        dbz = sn.read_dbz(ds, prod["unit"])
        loc = find_location(f)
        polar = sn.find_polar_info(f, ds)
        if polar:
            dbz, projdef = sn.polar_to_grid(dbz, polar)
            h, w = dbz.shape
            xs = ys = sn.GRID_PS
            ll_x = ll_y = -sn.GRID_N * sn.GRID_PS / 2
            loc = (round(polar["lat"], 5), round(polar["lon"], 5))
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

    codes = prod["encode"](dbz)
    ll_x, ll_y = round(ll_x, 3), round(ll_y, 3)
    grid = {
        "format": "raw-uint8",
        "width": w,
        "height": h,
        "row_order": "north-to-south",
        "pixel_size_m": xs,
        "ll_x": ll_x,
        "ll_y": ll_y,
        "extent_native": [ll_x, ll_y, round(ll_x + w * xs, 3), round(ll_y + h * ys, 3)],
        "nodata": sn.NODATA_CODE,
        "projdef": projdef,
    }
    return codes, normalize(grid), loc


_first_done: dict[str, float] = {}   # je Produkt: Zeitpunkt, an dem der erste Standort seine Datei hatte (gilt fuer alle Threads)


def list_files(site: dict, prod: dict) -> list[str]:
    """Dateinamen aus dem DWD-Verzeichnis des Standorts (Apache-Index)."""
    url = f"{BASE}/{prod['dir']}/{site['code']}/hdf5/{FILTER}/"
    with urllib.request.urlopen(url, timeout=30) as r:
        html = r.read().decode("utf-8", errors="ignore")
    return re.findall(r'href="([^"/]+-hd5)"', html)


def find_latest_file(site: dict, prod: dict) -> tuple[str, datetime] | None:
    """Neueste Datei des Standorts (Name, Zeit aus dem Dateinamen in UTC) oder None."""
    pat = re.compile(
        rf"ras\d+(?:-stqual)?-vol5minng01_sweeph5onem_{prod['name']}_{ELEV}-(\d{{14}})\d{{2}}"
        rf"-{re.escape(site['code'])}-{re.escape(site['wmo'])}-hd5$",
        re.IGNORECASE,
    )
    best = None
    for n in list_files(site, prod):
        m = pat.match(n)
        if m and (best is None or m.group(1) > best[0]):
            best = (m.group(1), n)
    if best is None:
        return None
    return best[1], datetime.strptime(best[0], "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)


def download(site: dict, prod: dict, pkey: str, newer_than: str | None, dest: Path, wait: int) -> tuple[str, datetime]:
    """Laedt die neueste Datei, sofern sie juenger als `newer_than` ("%Y%m%d_%H%M") und nicht aelter als MAX_AGE_MIN ist.

    Gibt es noch nichts Neues, wird alle POLL_SECONDS erneut gefragt, bis `wait` s um sind (dann TimeoutError).
    Sobald irgendein Standort seine Datei hat, endet das Warten der anderen spaetestens GRACE_SECONDS danach.
    """
    start = time.time()
    attempt = 0
    while True:
        attempt += 1
        try:
            found = find_latest_file(site, prod)
            if found:
                name, ft = found
                stamp = ft.strftime("%Y%m%d_%H%M")
                age_min = (datetime.now(timezone.utc) - ft).total_seconds() / 60
                if newer_than and stamp <= newer_than:
                    pass                                    # schon veroeffentlicht, auf die naechste warten
                elif age_min > MAX_AGE_MIN:
                    print(f"[{site['code']}/{pkey}] neueste Datei {stamp} ist {age_min:.0f} min alt (> {MAX_AGE_MIN})")
                else:
                    url = f"{BASE}/{prod['dir']}/{site['code']}/hdf5/{FILTER}/{name}"
                    with urllib.request.urlopen(url, timeout=30) as r:
                        data = r.read()
                    if data:
                        dest.write_bytes(data)
                        _first_done.setdefault(pkey, time.time())
                        print(f"[{site['code']}/{pkey}] geladen (Versuch {attempt}): {name}")
                        return name, ft
        except urllib.error.HTTPError as e:
            if e.code != 404:
                print(f"[{site['code']}/{pkey}] HTTP {e.code}")
        except OSError as e:
            print(f"[{site['code']}/{pkey}] Netzfehler: {e}")
        # Hat schon ein anderer Standort geliefert, bekommen Nachzuegler nur noch GRACE_SECONDS
        deadline = start + wait
        fd = _first_done.get(pkey)
        if fd is not None:
            deadline = min(deadline, max(fd + GRACE_SECONDS, start + POLL_SECONDS))
        if time.time() > deadline:
            raise TimeoutError(f"keine neue Datei fuer {site['code']}/{pkey}")
        time.sleep(POLL_SECONDS)


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


def process_site(s3, bucket: str, site: dict, pkey: str, prev_ts: str | None, wait: int) -> dict:
    """Update fuer den Meta-Eintrag: {"status": ...} bzw. bei ok zusaetzlich timestamp, lat, lon, grid."""
    code = site["code"]
    prod = PRODUCTS[pkey]
    tag = f"{code}/{pkey}"

    try:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "in.hd5"
            _name, file_time = download(site, prod, pkey, prev_ts, src, wait)
            ts = file_time.strftime("%Y%m%d_%H%M")       # Zeit aus dem Dateinamen
            codes, grid, (lat, lon) = convert(src, prod)
    except TimeoutError as e:
        print(f"[{tag}] unavailable: {e}")
        return {"status": "unavailable"}
    except Exception as e:
        print(f"[{tag}] invalid (Lesefehler): {e}")
        return {"status": "invalid"}

    nodata_share = float((codes == sn.NODATA_CODE).mean())
    print(f"[{tag}] Anteil nodata: {nodata_share:.1%}")
    if prod["nodata_chk"] and nodata_share > MAX_NODATA:
        print(f"[{tag}] invalid: {nodata_share:.1%} > {MAX_NODATA:.0%} nodata")
        return {"status": "invalid"}

    s3.put_object(
        Bucket=bucket,
        Key=f"{PREFIXES[prod['prefix']]}/{code}/{prod['file']}",
        Body=gzip.compress(codes.tobytes(), 9, mtime=0),
        ContentType="application/gzip",
        CacheControl=CACHE,
    )
    print(f"[{tag}] OK {ts}")
    return {
        "status": "ok",
        "timestamp": ts,
        "lat": lat if lat is not None else site.get("lat"),
        "lon": lon if lon is not None else site.get("lon"),
        "grid": grid,
    }


def first(*vals):
    return next((v for v in vals if v is not None), None)


def main() -> None:
    sites = json.loads(SITES_FILE.read_text(encoding="utf-8"))
    order = [s["code"] for s in sites]
    only = set(sys.argv[1:])
    run_sites = [s for s in sites if not only or s["code"] in only]

    print(f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC, {len(run_sites)} Standorte")

    s3 = boto3.client(
        "s3",
        endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )
    bucket = os.environ["R2_BUCKET"]
    meta_key = f"{R2_PREFIX}/metadata.json"

    old_meta = get_json(s3, bucket, meta_key)
    old_radars = old_meta.get("radars", {})
    old_shared = old_meta.get("grid") or {}
    radial_key = f"{RADIAL_PREFIX}/metadata.json"
    old_radial = get_json(s3, bucket, radial_key).get("radars", {})

    updates: dict[str, dict] = {}    # dbz
    vupdates: dict[str, dict] = {}   # vrad
    errors = []
    with ThreadPoolExecutor(max_workers=(len(run_sites) * len(PRODUCTS)) or 1) as pool:
        futs = {}
        for s in run_sites:
            for pkey in PRODUCTS:
                prev = (old_radars if pkey == "dbz" else old_radial).get(s["code"], {})
                # Radar war zuletzt schon gestoert -> nicht wieder 3 Minuten warten
                wait = WAIT_SECONDS if prev.get("status", "ok") == "ok" else RETRY_WAIT_SECONDS
                futs[pool.submit(process_site, s3, bucket, s, pkey, prev.get("timestamp"), wait)] = (s["code"], pkey)
        for fut in as_completed(futs):
            code, pkey = futs[fut]
            try:
                (updates if pkey == "dbz" else vupdates)[code] = fut.result()
            except Exception as e:  # z.B. R2-Upload fehlgeschlagen: Eintrag unveraendert lassen
                print(f"[{code}/{pkey}] FEHLER: {e}")
                errors.append(f"{code}/{pkey}")

    # --- Eintraege + volle Grids je Radar zusammenbauen (alt + neu) ---
    radars: dict[str, dict] = {}
    fulls: dict[str, dict] = {}
    for s in sites:
        code = s["code"]
        prev = old_radars.get(code)
        upd = updates.get(code)
        if upd is None:
            if prev is not None:
                radars[code] = dict(prev)
                g = expand(old_shared, prev)
                if g:
                    fulls[code] = g
            continue
        prev = prev or {}
        entry = {"wmo": s["wmo"]}
        if s.get("name"):
            entry["name"] = s["name"]
        entry["status"] = upd["status"]
        entry["timestamp"] = first(upd.get("timestamp"), prev.get("timestamp"))
        entry["lat"] = first(upd.get("lat"), prev.get("lat"), s.get("lat"))
        entry["lon"] = first(upd.get("lon"), prev.get("lon"), s.get("lon"))
        entry["file"] = f"{code}/latest"
        if upd["status"] == "ok":
            fulls[code] = upd["grid"]
        else:
            g = expand(old_shared, prev)
            if g:
                fulls[code] = g
        radars[code] = entry
    for code, prev in old_radars.items():       # Radare, die nicht (mehr) in sites.json stehen, unveraendert lassen
        if code not in radars:
            radars[code] = prev

    # --- Grid einmal oben; projdef pro Radar; Abweichler bekommen ein eigenes "grid" ---
    ref = None
    for code in order + [c for c in fulls if c not in order]:
        if code in fulls:
            ref = {k: fulls[code][k] for k in SHARED_KEYS}
            break
    for r in radars.values():
        r.pop("projdef", None)   # steht jetzt einmal oben im grid
        r.pop("grid", None)
    for code, full in fulls.items():
        r = radars[code]
        r["ll_x"], r["ll_y"], r["extent_native"] = full["ll_x"], full["ll_y"], full["extent_native"]
        diff = {k: full[k] for k in SHARED_KEYS if full[k] != ref[k]}
        if diff:
            level = "::notice::" if set(diff) <= {"projdef"} else "::warning::"
            print(f"{level}[{code}] weicht vom gemeinsamen Grid ab: {', '.join(diff)}")
            r["grid"] = diff
    for r in radars.values():
        for k in ("ll_x", "ll_y", "extent_native"):
            r.setdefault(k, None)

    meta = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
        "grid": ref or old_shared,
        "radars": radars,
    }
    s3.put_object(
        Bucket=bucket, Key=meta_key,
        Body=json.dumps(meta, indent=2, ensure_ascii=False).encode("utf-8"),
        ContentType="application/json",
        CacheControl=CACHE,
    )

    # --- Radialgeschwindigkeit: eigene metadata.json unter sites/radial (Grid-Angaben identisch zu radar) ---
    rradars: dict[str, dict] = {}
    for s in sites:
        code = s["code"]
        v = vupdates.get(code)
        old_v = old_radial.get(code)
        if v is None:
            if old_v is not None:
                rradars[code] = old_v
            continue
        old_v = old_v or {}
        base = radars.get(code, {})
        rradars[code] = {
            "wmo": s["wmo"],
            **({"name": s["name"]} if s.get("name") else {}),
            "status": v["status"],
            "timestamp": first(v.get("timestamp"), old_v.get("timestamp")),
            "lat": base.get("lat"), "lon": base.get("lon"),
            "file": f"{code}/latest",
            "ll_x": base.get("ll_x"), "ll_y": base.get("ll_y"), "extent_native": base.get("extent_native"),
        }
    for code, prev in old_radial.items():       # nicht mehr in sites.json: unveraendert lassen
        rradars.setdefault(code, prev)
    rmeta = {
        "generated_at": meta["generated_at"],
        "unit": "km/h",   # Decodieren im Client: kmh = (code - zero_code) * step_kmh; 0 und 255 transparent
        "step_kmh": sn.VRAD_STEP_KMH, "zero_code": sn.VRAD_ZERO_CODE,
        "min_code": 1, "max_code": sn.VRAD_ZERO_CODE + sn.VRAD_MAX_STEPS,
        "nodata": [0, sn.NODATA_CODE],
        "grid": meta["grid"],
        "radars": rradars,
    }
    s3.put_object(
        Bucket=bucket, Key=radial_key,
        Body=json.dumps(rmeta, indent=2, ensure_ascii=False).encode("utf-8"),
        ContentType="application/json",
        CacheControl=CACHE,
    )

    print("metadata.json geschrieben: " + ", ".join(f"{c}={v.get('status')}" for c, v in radars.items()))
    bad = [c for c, v in radars.items() if v.get("status") != "ok"]
    bad += [f"{c}/vrad" for c, v in rradars.items() if v.get("status") != "ok"]
    if bad:
        print(f"::warning::Radare nicht ok: {', '.join(bad)}")
    if errors:
        print(f"::error::Upload-Fehler: {', '.join(errors)}")
        sys.exit(1)


if __name__ == "__main__":
    main()