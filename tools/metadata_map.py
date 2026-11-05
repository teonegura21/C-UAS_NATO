#!/usr/bin/env python3
"""Raw sensor class-metadata value -> exact submission dropdown string (the ONE place).

`map_value(field, raw)` returns a string from C.DROPDOWNS[field], or '' when the raw value has
no dropdown equivalent. 'Unknown' / 'Not Applicable' are NEVER returned here: whether to write
'Unknown' (cannot determine) or 'Not Applicable' (no data) is a fusion decision, not a mapping.

`field` is one of C.METADATA_COLUMNS: 'uav_model/manufacturer', 'protocol', 'serial_number',
'signal_frequency_band'.

Every rule below is backed by a raw value that is actually present in the sensor data or in
the evaluation Friendly(Blue)UAS.xlsx whitelists (the evidence is quoted next to the rule and
re-checked by tools/metadata_scan.py, which also writes every raw value it saw to
data/derived/metadata_values.csv). Nothing is guessed:

* protocol (case-insensitive, whitespace-trimmed whole-string match)
    'ocusync' | 'dji ocusync' | 'dji ocusyncccf' | 'sv-ocusync' -> 'DJI Ocusyinc'
        STUG protocol 'ocusync', OEPS technology 'DJI Ocusync', ORFH / YXAB id 'DJI OcuSyncCCF',
        IKJX / MMJJ protocol 'sv-ocusync'. (The dropdown spelling 'Ocusyinc' is the template's typo.)
    'wifi' -> 'Wifi'                          STUG protocol 'wifi'
    'autel skylink' -> 'Autel Skylink'        STUG protocol 'Autel Skylink'
    'video' -> 'Video'                        ORFH / YXAB id 'Video'
    'digital video link type 2' -> 'Digital video link type 2'   OEPS technology
    'diy crossfire' -> 'DIY Crossfire'        STUG model column 'DIY CROSSFIRE' (see protocol_from_model)
  Drone-type strings that sit in the protocol slot of ORFH / YXAB ('DJI Spark / Phantom 3 SE /
  Mavic Air / Mavic Mini', 'Eleron 3', 'Zala Aero') are not protocols -> ''.
* signal_frequency_band: raw = centre frequency in MHz. The band of C.DROPDOWNS whose
  [lo, hi] contains it; where bands overlap ('2200 – 2500' contains '2400 – 2483.5') the
  NARROWEST wins, so 2400-2483.5 MHz is '2400 – 2483.5' and only 2200-2400 / 2483.5-2500 MHz
  falls back to '2200 – 2500'. Outside every band -> ''.
* serial_number: whole-string match after strip() and upper() against the dropdown serials
  (CEQX writes its serials in lower case, e.g. '1748fev2hl7824051124'). Anything else -> ''.
  `serial_near` reports the near misses seen in the data (a vendor prefix / suffix around a
  dropdown serial: STUG '1581F45TB21BR1AE03FP' = '1581' + 'F45TB21BR1AE03FP', IKJX
  'F1ZNBJAB00C00MQ0' = 'F' + '1ZNBJAB00C00MQ' + '0'); it is NOT used by map_value.
* uav_model/manufacturer: whole-string, case-insensitive. Dropdown strings map to themselves;
  aliases seen in the data / whitelists:
    'mavic 3' -> 'DJI Mavic 3'                          STUG model 'Mavic 3'
    'mavic 3 e' -> 'DJI Mavic 3 Enterprise'             STUG model 'Mavic 3 E'
    'anafi se' -> 'Parrot Anafi'                        STUG model 'Anafi SE'
    'dji matrice 300r' -> 'MTRC 300R'                   whitelist 'Manufacturer' of all 3 evaluation iterations
    'autel evo 2' -> 'Autel Drone'                      whitelist (DII, DIII)
    'parrot' -> 'Parrot Anafi'                          GTYT objectInfo droneMaker 'Parrot' (maker only; the
        dropdown has a single Parrot model, so the template cannot express any other Parrot; rule
        name 'model:maker_only:parrot' flags it, see MAKER_ONLY. WEAKEST rule: the only supporting evidence is that the
        Parrot Anafi Dronetag flight is the best bearing candidate for 202 of 295 GTYT rows in
        testing S4-I4, all ambiguous.)
  Unmapped models seen: 'DJI Mavic 2 Pro', 'Mavic 2', 'Remote ID',
  'DIY DJI O3', 'OcuSync', 'DIY CROSSFIRE', 'DRI Drone' -> ''.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cuas_common as C  # noqa: E402

FIELDS = C.METADATA_COLUMNS
MODEL, PROTOCOL, SERIAL, BAND = FIELDS

_PROTOCOL_RULES = {
    "ocusync": "DJI Ocusyinc", "dji ocusync": "DJI Ocusyinc", "dji ocusyncccf": "DJI Ocusyinc",
    "sv-ocusync": "DJI Ocusyinc",
    "wifi": "Wifi",
    "autel skylink": "Autel Skylink",
    "video": "Video",
    "digital video link type 2": "Digital video link type 2",
    "diy crossfire": "DIY Crossfire",
}

# alias (lower case) -> dropdown model; the rule name documents the evidence.
_MODEL_RULES = {
    "mavic 3": "DJI Mavic 3",
    "mavic 3 e": "DJI Mavic 3 Enterprise",
    "anafi se": "Parrot Anafi",
    "dji matrice 300r": "MTRC 300R",
    "autel evo 2": "Autel Drone",
    "parrot": "Parrot Anafi",
}

# Aliases whose raw value names only a maker, not a model. The dropdown has exactly one entry for
# that maker, so the mapping is the only choice the template allows, but a different model of the
# same maker would be mislabelled. map_with_rule reports these as 'model:maker_only:<alias>' so the
# caller can discount them (e.g. do not treat them as model evidence when a model-bearing sensor
# disagrees).
MAKER_ONLY = frozenset({"parrot"})

# Serial -> dropdown model, i.e. the device identity behind a serial. Evidence (re-verified by
# tools/metadata_scan.py against every object that carries both a serial and a mapped model):
#   F45TB21BR1AE03FP      CEQX model 'DJI Mavic 3' and STUG model 'Mavic 3' on the same serial
#   F5FHD233M00DEYG6      CEQX model 'DJI Mavic 3 Enterprise' and STUG model 'Mavic 3 E'
#   1748FEV2HL7824051124  whitelist (DII, DIII): 'Autel Evo 2'
#   1ZNBJAB00C00MQ        whitelist (DI, DII, DIII): 'DJI Matrice 300R'
# 163DG1700154R3 is a DJI Mavic 2 Pro (CEQX) / 'Mavic 2' (STUG): not in the model dropdown.
SERIAL_MODEL = {
    "F45TB21BR1AE03FP": "DJI Mavic 3",
    "F5FHD233M00DEYG6": "DJI Mavic 3 Enterprise",
    "1748FEV2HL7824051124": "Autel Drone",
    "1ZNBJAB00C00MQ": "MTRC 300R",
}

_PLACEHOLDER = frozenset({"Unknown", "Not Applicable"})


def _norm(raw) -> str:
    return " ".join(str(raw).split()) if raw is not None else ""


def parse_bands() -> list[tuple[float, float, str]]:
    """[(lo, hi, dropdown string)] parsed from C.DROPDOWNS['signal_frequency_band']."""
    out = []
    for s in C.DROPDOWNS[BAND]:
        if s in _PLACEHOLDER:
            continue
        m = re.fullmatch(r"([\d.]+) – ([\d.]+)", s)
        if not m:
            raise ValueError(f"unparseable band {s!r}")
        out.append((float(m[1]), float(m[2]), s))
    return out


_BANDS = parse_bands()


def band_for(freq_mhz: float) -> str:
    """Narrowest dropdown band containing `freq_mhz` ('' if none)."""
    best = ""
    width = float("inf")
    for lo, hi, s in _BANDS:
        if lo <= freq_mhz <= hi and hi - lo < width:
            best, width = s, hi - lo
    return best


def map_with_rule(field: str, raw) -> tuple[str, str]:
    """(mapped dropdown string or '', rule name)."""
    if field not in FIELDS:
        raise KeyError(f"unknown metadata field {field!r}")
    s = _norm(raw)
    if not s:
        return "", "empty"
    low = s.lower()
    if field == PROTOCOL:
        hit = _PROTOCOL_RULES.get(low)
        return (hit, f"protocol:{low}") if hit else ("", "protocol:no_dropdown")
    if field == BAND:
        try:
            f = float(s)
        except ValueError:
            return "", "band:not_a_number"
        if f != f:
            return "", "band:not_a_number"
        hit = band_for(f)
        return (hit, "band:narrowest_containing") if hit else ("", "band:outside_all")
    if field == SERIAL:
        up = s.upper()
        if up in C.DROPDOWNS[SERIAL] and up not in _PLACEHOLDER:
            return up, "serial:exact"
        return "", "serial:no_dropdown"
    # model / manufacturer
    for d in C.DROPDOWNS[MODEL]:
        if d not in _PLACEHOLDER and d.lower() == low:
            return d, "model:dropdown"
    hit = _MODEL_RULES.get(low)
    if hit:
        return hit, (f"model:maker_only:{low}" if low in MAKER_ONLY else f"model:alias:{low}")
    return "", "model:no_dropdown"


def map_value(field: str, raw) -> str:
    """Dropdown string for a raw sensor / whitelist value; '' when unmappable."""
    return map_with_rule(field, raw)[0]


def serial_near(raw) -> str:
    """Dropdown serial that is a proper substring of `raw` (or contains it, len >= 10), '' if
    none and also '' when `raw` is an exact match. Reporting aid only."""
    s = _norm(raw).upper()
    if not s or s in C.DROPDOWNS[SERIAL]:
        return ""
    for d in C.DROPDOWNS[SERIAL]:
        if d in _PLACEHOLDER:
            continue
        if d in s or (len(s) >= 10 and s in d):
            return d
    return ""


def resolve_serial(raw) -> str:
    """Dropdown serial for a raw serial: exact (`map_value`), else the near match
    (`serial_near`), else ''. Used for device identity; `map_value` stays strict."""
    return map_value(SERIAL, raw) or serial_near(raw)


def model_for_serial(raw_serial) -> str:
    """Dropdown model implied by a serial (SERIAL_MODEL, near matches resolved), '' if unknown."""
    return SERIAL_MODEL.get(resolve_serial(raw_serial), "")


def protocol_from_model(raw_model) -> str:
    """STUG puts 'DIY CROSSFIRE' in its model slot; when a model string is itself a dropdown
    protocol name ('DIY Crossfire') return that protocol, else ''."""
    return "DIY Crossfire" if _norm(raw_model).lower() == "diy crossfire" else ""


if __name__ == "__main__":
    for f, v in ((PROTOCOL, "wifi"), (BAND, "2473.5"), (SERIAL, "f5fhd233m00deyg6"), (MODEL, "Mavic 3 E")):
        print(f, repr(v), "->", repr(map_value(f, v)))
