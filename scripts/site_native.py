#!/usr/bin/env python3
"""DWD-Einzelstandort (ODIM-HDF5, Polar-Sweep 0.5 Grad, DBZH, 360 Strahlen x 720 Bins x 250 m) -> natives uint8-Raster in dBZ (1440 x 1440 px, 250 m).

Dateiname: ras07-stqual-vol5minng01_sweeph5onem_dbzh_<elev>-<yyyymmddHHMMSS><cc>-<kuerzel>-<WMO-Nr>-hd5
           z.B. ras07-stqual-vol5minng01_sweeph5onem_dbzh_05-2026102220380100-boo-10132-hd5
           (elev 05 = 0.5 Grad, cc = Hundertstel-Sekunden, boo = Boostedt)

Ausgabe: rohe uint8-Bytes, keine Dateiendung, ysize Zeilen x xsize Spalten, Zeile 0 = Norden.
Die Werte BLEIBEN dBZ, nur in 0.5-dB-Schritten in ein Byte gepackt:

    0        kein Echo (undetect, < DBZ_MIN)   -> transparent
    1 .. 254 dBZ = DBZ_MIN + (code - 1) * DBZ_STEP     (-10.0 .. +116.5 dBZ)
    255      kein Daten / ausserhalb           -> transparent

Decodieren im Client:  dbz = -10 + (code - 1) * 0.5
Aufruf:   python3 site_native.py [datei]       (ohne Argument: neueste Datei in data/site)
Farben:   python3 site_native.py --colors
Struktur: python3 site_native.py <datei> --info      (Diagnose: zeigt alle Gruppen/Attribute)
"""
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
from pyproj import Transformer

SRC_DIR = Path("data/site")
OUT_DIR = Path("output/site")

FILENAME_RE = re.compile(
    r"ras\d+-stqual-vol5minng01_sweeph5onem_dbzh_(\d+)-(\d{14})\d{2}-([a-z0-9]+)-(\d+)-hd5",
    re.IGNORECASE,
)

DBZ_MIN = 0.0
DBZ_STEP = 1.0
NODATA_CODE = 255
MAX_CODE = 254

DEFAULT_NODATA = 65535
DEFAULT_SCALE_M = 1000.0

# Farbtabelle fuer den Client: (ab dBZ, (R, G, B)) - diskrete Stufen
COLOR_TABLE: list[tuple[float, tuple[int, int, int]]] = [
    (5,  (150, 150, 150)),
    (10, (0, 200, 255)),
    (15, (0, 120, 255)),
    (20, (0, 40, 246)),
    (25, (0, 200, 0)),
    (30, (0, 150, 0)),
    (35, (255, 255, 0)),
    (40, (255, 190, 0)),
    (45, (255, 120, 0)),
    (50, (255, 0, 0)),
    (55, (190, 0, 0)),
    (60, (255, 120, 255)),
    (65, (170, 0, 170)),
    (70, (255, 255, 255)),
]


def parse_filename(name: str) -> tuple[str, datetime, str]:
    m = FILENAME_RE.match(name)
    if not m:
        raise ValueError(f"Dateiname passt nicht zum Schema: {name}")
    _elev, ts, code, wmo = m.groups()
    t = datetime.strptime(ts, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    return wmo, t, code.lower()


def _s(v) -> str:
    return v.decode(errors="ignore") if isinstance(v, bytes) else str(v)


def _x(v):
    return v.item() if isinstance(v, np.ndarray) and v.size == 1 else v


def find_dbz_dataset(f: h5py.File, quantity: str = "DBZ") -> h5py.Dataset:
    """Waehlt das groesste 2D-Dataset der Groesse `quantity` ("DBZ" oder "VRAD"; Praefix-Vergleich, also DBZH/VRADH ok).

    Die Datei enthaelt mehrere Ansichten (z.B. 400x12 Seitenansichten und 400x400 Draufsicht).
    """
    cands: list[h5py.Dataset] = []

    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset) and name.endswith("/data") and obj.ndim == 2:
            what = obj.parent.get("what")
            q = _s(what.attrs["quantity"]).upper() if what is not None and "quantity" in what.attrs else quantity
            if q.startswith(quantity):
                cands.append(obj)

    f.visititems(visitor)
    if not cands:
        raise RuntimeError(f"Kein 2D-{quantity}-Datensatz gefunden.")

    def score(ds):
        top = 0
        w = ds.parent.parent.get("what")
        if w is not None and _s(w.attrs.get("product", "")).upper() == "MAX":
            top = 1
        return (top, ds.shape[0] * ds.shape[1])

    best = max(cands, key=score)
    print("Datasets:", ", ".join(f"{d.name}{d.shape}" for d in cands), "-> gewaehlt:", best.name)
    return best


def read_dbz(ds: h5py.Dataset, unit: str = "dBZ") -> np.ndarray:
    """Physikalischer Wert (dBZ bzw. m/s bei VRADH) als float64, NaN = kein Datum, -inf = undetect."""
    what = None
    for grp in (ds.parent, ds.parent.parent, ds.file):
        w = grp.get("what")
        if w is not None and "gain" in w.attrs:
            what = w
            break
    a = what.attrs if what is not None else {}
    gain = float(_x(a.get("gain", 0.5)))
    offset = float(_x(a.get("offset", -32.5)))
    nodata = _x(a.get("nodata", DEFAULT_NODATA))
    undetect = _x(a["undetect"]) if "undetect" in a else None

    raw = ds[()]
    dbz = raw.astype(np.float64) * gain + offset
    dbz[raw == nodata] = np.nan
    if undetect is not None and undetect != nodata:
        dbz[raw == undetect] = -np.inf

    # Werte < DBZ_MIN (Rauschen, bis -64 dBZ) sind gemessen -> encode_dbz macht daraus Code 0 (kein Echo), NICHT Nodata

    print(f"quantity='{_s(a.get('quantity', ''))}', gain={gain}, offset={offset}, "
          f"nodata={nodata}, undetect={undetect}")
    valid = dbz[np.isfinite(dbz)]
    if valid.size:
        print(f"Wertebereich: {valid.min():.1f} .. {valid.max():.1f} {unit}")
    return dbz


def find_where(f: h5py.File) -> h5py.Group:
    found: list[h5py.Group] = []

    def visitor(name, obj):
        if isinstance(obj, h5py.Group) and name.split("/")[-1] == "where" and "projdef" in obj.attrs:
            found.append(obj)

    root = f.get("where")
    if root is not None and "projdef" in root.attrs:
        return root
    f.visititems(visitor)
    if not found:
        raise RuntimeError("Keine 'where'-Gruppe mit projdef gefunden (evtl. Polar- statt Kartesisch-Daten).")
    return found[0]



GRID_N = 1440           # Zielraster bei Polar-Daten: 1440 x 1440 px
GRID_PS = 250.0         # 250 m / px (= Bin-Laenge der Sweeps) -> +-180 km um das Radar


def dump_structure(f: h5py.File) -> None:
    """Gibt Gruppen, Attribute und Dataset-Formen aus (zur Diagnose)."""
    def visitor(name, obj):
        kind = f"Dataset {obj.shape} {obj.dtype}" if isinstance(obj, h5py.Dataset) else "Group"
        print(f"/{name}  [{kind}]")
        for k, v in obj.attrs.items():
            print(f"     {k} = {v!r}")
    for k, v in f.attrs.items():
        print(f"(root) {k} = {v!r}")
    f.visititems(visitor)


def find_polar_info(f: h5py.File, ds: h5py.Dataset):
    """Polar-Infos (nrays/nbins/rscale/Azimute/...) zum Dataset, sonst None.

    ODIM-Polar: .../dataset1/data1/data  ->  nbins/nrays/rscale/a1gate in .../dataset1/where,
    Strahlwinkel in .../dataset1/how (startazA/stopazA), Standort lat/lon in /where.
    """
    w = None
    for grp in (ds.parent, ds.parent.parent):
        cand = grp.get("where")
        if cand is not None and "nbins" in cand.attrs and "nrays" in cand.attrs:
            w = cand
            break
    if w is None:
        return None
    root = f.get("where")
    src = root if (root is not None and "lat" in root.attrs) else w
    info = {
        "nbins": int(_x(w.attrs["nbins"])),
        "nrays": int(_x(w.attrs["nrays"])),
        "rscale": float(_x(w.attrs["rscale"])),
        "rstart_m": float(_x(w.attrs.get("rstart", 0.0))) * 1000.0,   # ODIM: rstart in km
        "a1gate": int(_x(w.attrs.get("a1gate", 0))),
        "elangle": float(_x(w.attrs["elangle"])) if "elangle" in w.attrs else None,
        "lat": float(_x(src.attrs["lat"])),
        "lon": float(_x(src.attrs["lon"])),
        "az_mid": None,
    }
    how = w.parent.get("how") if w.name != "/" else None
    if how is not None and "startazA" in how.attrs and "stopazA" in how.attrs:
        a0 = np.asarray(how.attrs["startazA"], dtype=np.float64).ravel()
        a1 = np.asarray(how.attrs["stopazA"], dtype=np.float64).ravel()
        if a0.size == info["nrays"] and a1.size == info["nrays"]:
            info["az_mid"] = (a0 + ((a1 - a0 + 180.0) % 360.0 - 180.0) / 2.0) % 360.0   # Strahlmitte, 0 = Nord
    return info


def polar_to_grid(dbz: np.ndarray, p: dict) -> tuple[np.ndarray, str]:
    """Polar (rays x bins) -> kartesisch GRID_N x GRID_N, Radar in der Mitte, Zeile 0 = Norden."""
    if dbz.shape != (p["nrays"], p["nbins"]):
        raise RuntimeError(f"Array {dbz.shape} passt nicht zu nrays/nbins {(p['nrays'], p['nbins'])}")
    half = GRID_N * GRID_PS / 2
    xs = ((np.arange(GRID_N) + 0.5) * GRID_PS - half).astype(np.float32)
    ys = (half - (np.arange(GRID_N) + 0.5) * GRID_PS).astype(np.float32)
    X, Y = np.meshgrid(xs, ys)
    rng = np.hypot(X, Y)
    az = np.degrees(np.arctan2(X, Y)) % 360.0                  # 0 = Nord, im Uhrzeigersinn
    del X, Y

    # Winkel -> Strahl ueber eine 0.1-Grad-Tabelle (spart Speicher gegenueber Pro-Pixel-Suche)
    centers = (np.arange(3600) + 0.5) / 10.0
    if p.get("az_mid") is not None:
        order = np.argsort(p["az_mid"])
        srt = p["az_mid"][order]
        i = np.searchsorted(srt, centers)
        lo, hi = srt[(i - 1) % srt.size], srt[i % srt.size]
        d_lo = np.abs((centers - lo + 180.0) % 360.0 - 180.0)
        d_hi = np.abs((centers - hi + 180.0) % 360.0 - 180.0)
        lut = order[np.where(d_lo <= d_hi, (i - 1) % srt.size, i % srt.size)]
    else:
        lut = (np.floor(centers / (360.0 / p["nrays"])).astype(int) + p["a1gate"]) % p["nrays"]
    ray = lut[np.minimum((az * 10).astype(np.int32), 3599)]
    del az

    b = np.floor((rng - p["rstart_m"]) / p["rscale"]).astype(np.int32)
    inside = (b >= 0) & (b < p["nbins"])
    out = np.full((GRID_N, GRID_N), np.nan)                    # ausserhalb Reichweite = nodata
    out[inside] = dbz[ray[inside], b[inside]]
    projdef = f"+proj=aeqd +lat_0={p['lat']} +lon_0={p['lon']} +x_0=0 +y_0=0 +ellps=sphere +units=m +no_defs"
    return out, projdef


def encode_dbz(dbz: np.ndarray) -> np.ndarray:
    codes = np.zeros(dbz.shape, dtype=np.uint8)           # 0 = kein Echo
    codes[np.isnan(dbz)] = NODATA_CODE
    ok = np.isfinite(dbz) & (dbz >= DBZ_MIN)
    k = np.round((dbz[ok] - DBZ_MIN) / DBZ_STEP) + 1
    codes[ok] = np.clip(k, 1, MAX_CODE).astype(np.uint8)
    return codes


# Radialgeschwindigkeit (VRADH): Quelle m/s, Ausgabe km/h in 3.6-km/h-Schritten (= 1 m/s), +-115.2 km/h
#   0        kein Echo (undetect)              -> transparent
#   1 .. 65  km/h = (code - 33) * 3.6          (-115.2 .. 0 .. +115.2)
#   255      kein Daten / ausserhalb           -> transparent
# Vorzeichen wie ODIM: negativ = auf das Radar zu, positiv = vom Radar weg.
VRAD_STEP_KMH = 3.6
VRAD_ZERO_CODE = 33
VRAD_MAX_STEPS = 32     # 32 * 3.6 = 115.2 km/h


def encode_vrad(v_ms: np.ndarray) -> np.ndarray:
    codes = np.zeros(v_ms.shape, dtype=np.uint8)          # 0 = kein Echo
    codes[np.isnan(v_ms)] = NODATA_CODE
    ok = np.isfinite(v_ms)
    steps = np.clip(np.round(v_ms[ok] * 3.6 / VRAD_STEP_KMH), -VRAD_MAX_STEPS, VRAD_MAX_STEPS)
    codes[ok] = (steps + VRAD_ZERO_CODE).astype(np.uint8)
    return codes


def print_client_colors() -> None:
    print("  dbz: {")
    print(f"    // dbz = {DBZ_MIN} + (code - 1) * {DBZ_STEP};  0 und {NODATA_CODE} transparent")
    print(f"    dbzMin: {DBZ_MIN}, dbzStep: {DBZ_STEP}, nodata: [0, {NODATA_CODE}],")
    print("    steps: [   // [ab Code, Farbe]  (ab dBZ)")
    for dbz, (r, g, b) in COLOR_TABLE:
        code = int(round((dbz - DBZ_MIN) / DBZ_STEP)) + 1
        print(f"      [{code}, '#{r:02x}{g:02x}{b:02x}'],   // ab {dbz} dBZ")
    print("    ],")
    print("  },")


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--colors" in sys.argv:
        print_client_colors()
        return

    if args:
        src = Path(args[0])
    else:
        files = sorted(p for p in SRC_DIR.glob("ras*-hd5") if FILENAME_RE.match(p.name))
        if not files:
            sys.exit(f"Keine Datei in {SRC_DIR} gefunden.")
        src = files[-1]
    wmo, ts, site = parse_filename(src.name)
    print(f"Quelle: {src.name}  Standort {site} ({wmo}), {ts:%Y-%m-%d %H:%M:%S} UTC")

    with h5py.File(src, "r") as f:
        if "--info" in sys.argv:
            dump_structure(f)
            return
        ds = find_dbz_dataset(f)
        print(f"Dataset: {ds.name}, Form {ds.shape}")
        dbz = read_dbz(ds)
        polar = find_polar_info(f, ds)
        if polar:
            print("POLAR-Daten erkannt:", {k: v for k, v in polar.items() if k != "az_mid"})
            dbz, projdef = polar_to_grid(dbz, polar)
            h, w = dbz.shape
            xscale = yscale = GRID_PS
            ll_x = ll_y = -GRID_N * GRID_PS / 2
        else:
            wh = ds.parent.parent.get("where")
            if wh is None or "projdef" not in wh.attrs:
                wh = find_where(f)
            projdef = _s(wh.attrs["projdef"])
            h, w = dbz.shape
            xscale = float(_x(wh.attrs.get("xscale", DEFAULT_SCALE_M)))
            yscale = float(_x(wh.attrs.get("yscale", DEFAULT_SCALE_M)))
            to_proj = Transformer.from_crs("EPSG:4326", projdef, always_xy=True)
            if "LL_lon" in wh.attrs and "LL_lat" in wh.attrs:
                ll_x, ll_y = to_proj.transform(float(_x(wh.attrs["LL_lon"])), float(_x(wh.attrs["LL_lat"])))
            else:
                ll_x, ll_y = -w * xscale / 2, -h * yscale / 2

    print(f"Raster: {w} x {h} px, {xscale} m / px, projdef: {projdef}")
    codes = encode_dbz(dbz)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"dbz_{site}_{ts:%Y%m%d_%H%M}"
    np.ascontiguousarray(codes).tofile(out)
    print(f"Gespeichert: {out} ({w}x{h}, {codes.nbytes} Bytes)")

    meta = {
        "format": "raw-uint8", "unit": "dBZ", "site": site, "wmo": wmo,
        "width": w, "height": h, "row_order": "north-to-south",
        "dbz_min": DBZ_MIN, "dbz_step": DBZ_STEP,
        "projdef": projdef, "pixel_size_m": xscale,
        "ll_x": ll_x, "ll_y": ll_y,
        "extent_native": [ll_x, ll_y, ll_x + w * xscale, ll_y + h * yscale],
    }
    print("Client-Metadaten:", json.dumps(meta))

if __name__ == "__main__":
    main()