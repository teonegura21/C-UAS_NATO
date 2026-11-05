#!/usr/bin/env python3
"""Charts + glossary for the evaluation-only tier analysis.

Reads data/derived/eval/plot_data.json and field_ladder.csv (written by tools/eval_tiers.py)
and writes ONE self-contained page, data/derived/eval/EVAL_CLUSTERS.html: inline SVG, no
external scripts or fonts, so it opens offline and nothing leaves the machine.

Colour: the validated reference palette of the dataviz method. Scatter charts use at most
3 categorical colours (the all-pairs-safe slots); more groups become small multiples.
Every chart has a data table next to it.
"""

from __future__ import annotations

import csv
import html
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cuas_common as C  # noqa: E402
import kmeans as K  # noqa: E402

EVAL = C.DERIVED / "eval"
OUT = EVAL / "EVAL_CLUSTERS.html"
SERIES = ["s1", "s2", "s3", "s4", "s5", "s6", "s7"]      # CSS classes -> palette slots 1..7
W = 880                                                     # drawing width of a full-width chart

esc = html.escape


# --------------------------------------------------------------------------- #
# Small SVG helpers
# --------------------------------------------------------------------------- #


def nice_ticks(lo, hi, n=5):
    if hi <= lo:
        hi = lo + 1
    step = 10 ** math.floor(math.log10((hi - lo) / n))
    for m in (1, 2, 2.5, 5, 10):
        if (hi - lo) / (step * m) <= n:
            step *= m
            break
    start = math.ceil(lo / step) * step
    return [round(start + i * step, 10) for i in range(int((hi - start) / step) + 1)]


def fmt(v):
    if abs(v) >= 1000:
        return f"{v / 1000:g}k"
    return f"{v:g}"


class Plot:
    """Linear x/y plot area with hairline grid and axis labels."""

    def __init__(self, w, h, xlim, ylim, left=56, right=16, top=16, bottom=40, xlabel="", ylabel="",
                 xticks=None, yticks=None, xfmt=fmt, yfmt=fmt, equal=False):
        self.w, self.h, self.l, self.r, self.t, self.b = w, h, left, right, top, bottom
        self.x0, self.x1 = xlim
        self.y0, self.y1 = ylim
        if equal:                                   # same metres per pixel on both axes (maps)
            pw, ph = w - left - right, h - top - bottom
            sx, sy = (self.x1 - self.x0) / pw, (self.y1 - self.y0) / ph
            s = max(sx, sy)
            cx, cy = (self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2
            self.x0, self.x1 = cx - s * pw / 2, cx + s * pw / 2
            self.y0, self.y1 = cy - s * ph / 2, cy + s * ph / 2
        self.parts = []
        xt = xticks if xticks is not None else nice_ticks(self.x0, self.x1)
        yt = yticks if yticks is not None else nice_ticks(self.y0, self.y1)
        for v in yt:
            if self.y0 <= v <= self.y1:
                y = self.Y(v)
                self.parts.append(f'<line class="grid" x1="{left}" x2="{w - right}" y1="{y:.1f}" y2="{y:.1f}"/>'
                                  f'<text class="tick" x="{left - 6}" y="{y + 4:.1f}" text-anchor="end">{esc(yfmt(v))}</text>')
        for v in xt:
            if self.x0 <= v <= self.x1:
                x = self.X(v)
                self.parts.append(f'<line class="grid" x1="{x:.1f}" x2="{x:.1f}" y1="{top}" y2="{h - bottom}"/>'
                                  f'<text class="tick" x="{x:.1f}" y="{h - bottom + 16}" text-anchor="middle">{esc(xfmt(v))}</text>')
        self.parts.append(f'<line class="axis" x1="{left}" x2="{w - right}" y1="{h - bottom}" y2="{h - bottom}"/>')
        if xlabel:
            self.parts.append(f'<text class="lab" x="{(left + w - right) / 2:.1f}" y="{h - 6}" text-anchor="middle">{esc(xlabel)}</text>')
        if ylabel:
            self.parts.append(f'<text class="lab" transform="translate(14,{(top + h - bottom) / 2:.1f}) rotate(-90)" text-anchor="middle">{esc(ylabel)}</text>')

    def X(self, v):
        return self.l + (v - self.x0) / (self.x1 - self.x0) * (self.w - self.l - self.r)

    def Y(self, v):
        return self.h - self.b - (v - self.y0) / (self.y1 - self.y0) * (self.h - self.t - self.b)

    def add(self, s):
        self.parts.append(s)

    def svg(self, title):
        return (f'<svg viewBox="0 0 {self.w} {self.h}" role="img" aria-label="{esc(title)}">'
                f'<title>{esc(title)}</title>{"".join(self.parts)}</svg>')


def dot(x, y, cls, r=4, tip=None, ring=True):
    t = f' data-tip="{esc(tip)}" tabindex="0"' if tip else ""
    ringc = " ring" if ring else ""
    return f'<circle class="{cls}{ringc}" cx="{x:.1f}" cy="{y:.1f}" r="{r}"{t}/>'


def legend(items):
    """items: [(css class, label)] -> HTML legend row (identity never by colour alone: label beside swatch)."""
    return '<div class="legend">' + "".join(
        f'<span><i class="sw {c}"></i>{esc(lab)}</span>' for c, lab in items) + "</div>"


def table(header, rows, caption="Data table"):
    h = "".join(f"<th>{esc(str(x))}</th>" for x in header)
    b = "".join("<tr>" + "".join(f"<td>{esc(str(x))}</td>" for x in r) + "</tr>" for r in rows)
    return f'<details><summary>{esc(caption)}</summary><div class="tw"><table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div></details>'


def card(title, what, svg, extra="", how=""):
    how_html = f'<p class="how"><b>How to read it:</b> {how}</p>' if how else ""
    return f'<section class="card"><h3>{esc(title)}</h3><p class="what">{what}</p>{how_html}<div class="fig">{svg}</div>{extra}</section>'


# --------------------------------------------------------------------------- #
# Glossary illustrations (synthetic toy data, labelled as such)
# --------------------------------------------------------------------------- #


def toy_kmeans_svg():
    rng = np.random.default_rng(3)
    centers = np.array([[2.0, 2.0], [7.0, 3.0], [4.5, 7.5]])
    pts = np.vstack([c + rng.normal(0, 0.8, (25, 2)) for c in centers])
    res = K.kmeans(pts, 3, seed=0)
    p = Plot(420, 300, (0, 10), (0, 10), left=24, bottom=24, xticks=[], yticks=[], equal=True)
    for (x, y), lab in zip(pts, res.labels):
        p.add(dot(p.X(x), p.Y(y), SERIES[lab], r=4))
    for j, (cx, cy) in enumerate(res.centers):
        X, Y = p.X(cx), p.Y(cy)
        top = min(p.Y(y) for (x, y), lab in zip(pts, res.labels) if lab == j)
        p.add(f'<path class="cross" d="M{X - 8:.1f},{Y:.1f}H{X + 8:.1f}M{X:.1f},{Y - 8:.1f}V{Y + 8:.1f}"/>'
              f'<text class="ann strong" x="{X:.1f}" y="{top - 8:.1f}" text-anchor="middle">cluster {j} (+ = centroid)</text>')
    return p.svg("Toy example of K-means with K = 3")


def toy_silhouette_svg():
    rng = np.random.default_rng(5)
    a_pts = np.array([3.0, 5.0]) + rng.normal(0, 0.6, (12, 2))
    b_pts = np.array([7.5, 5.0]) + rng.normal(0, 0.6, (12, 2))
    me = a_pts[0]
    p = Plot(420, 300, (0, 10), (0.5, 9.5), left=24, bottom=24, xticks=[], yticks=[], equal=True)
    for q in a_pts[1:]:
        p.add(f'<line class="guide" x1="{p.X(me[0]):.1f}" y1="{p.Y(me[1]):.1f}" x2="{p.X(q[0]):.1f}" y2="{p.Y(q[1]):.1f}"/>')
    for q in a_pts[1:]:
        p.add(dot(p.X(q[0]), p.Y(q[1]), "s1", r=4))
    for q in b_pts:
        p.add(dot(p.X(q[0]), p.Y(q[1]), "s2", r=4))
    p.add(dot(p.X(me[0]), p.Y(me[1]), "s1", r=7))
    a = float(np.mean(np.hypot(*(a_pts[1:] - me).T)))
    b = float(np.mean(np.hypot(*(b_pts - me).T)))
    s = (b - a) / max(a, b)
    bx = p.X(b_pts[:, 0].mean())
    low = max(p.Y(q[1]) for q in np.vstack([a_pts, b_pts])) + 22
    high = min(p.Y(q[1]) for q in a_pts) - 12
    p.add(f'<line class="arrow" x1="{p.X(me[0]):.1f}" y1="{low - 12:.1f}" x2="{bx:.1f}" y2="{low - 12:.1f}"/>'
          f'<text class="ann" x="{(p.X(me[0]) + bx) / 2:.1f}" y="{low + 4:.1f}" text-anchor="middle">b = average distance to the nearest OTHER group = {b:.1f}</text>'
          f'<text class="ann" x="{p.X(a_pts[:, 0].mean()):.1f}" y="{high:.1f}" text-anchor="middle">a = average distance to its OWN group = {a:.1f}</text>')
    return p.svg("Toy example of the silhouette score"), a, b, s


# --------------------------------------------------------------------------- #
# Charts from the evaluation analysis
# --------------------------------------------------------------------------- #


def ladder_chart(ladder, n_doc):
    row_h, top, left = 18, 10, 270
    tiers = defaultdict(list)
    for r in ladder:
        tiers[int(r["tier"])].append(r)
    n_rows = len(ladder) + len(tiers)
    h = top + n_rows * row_h + 40
    p = Plot(W, h, (0, 100), (0, 1), left=left, right=70, top=top, bottom=40, xlabel=f"% of the {n_doc} PDF-documented evaluation sensors that have the field",
             xticks=[0, 25, 50, 75, 100], yticks=[], xfmt=lambda v: f"{v:g}%")
    y = top
    for t in sorted(tiers):
        rs = tiers[t]
        p.add(f'<text class="tier" x="8" y="{y + 13}">Tier {t} · {rs[0]["n_pdf"]}/{n_doc} sensors</text>')
        y += row_h
        for r in rs:
            pct = float(r["pct_pdf"])
            x0, x1 = p.X(0), p.X(pct)
            miss = r["missing"].split()
            who = "all sensors" if not miss else ("missing: " + ", ".join(miss) if len(miss) <= n_doc // 2 else "only: " + ", ".join(r["sensors"].split()))
            tip = f'{r["field"]}: {r["n_pdf"]}/{n_doc} sensors ({pct:.0f}%) — {who}'
            p.add(f'<text class="tick" x="{left - 8}" y="{y + 12}" text-anchor="end">{esc(r["field"])}</text>')
            if pct > 0:
                p.add(f'<path class="s1" d="{bar_path(x0, y + 3, x1 - x0, row_h - 6)}" data-tip="{esc(tip)}" tabindex="0"/>')
            p.add(f'<text class="val" x="{x1 + 6:.1f}" y="{y + 12}">{r["n_pdf"]}/{n_doc}</text>')
            y += row_h
    return p.svg("Field ladder")


def bar_path(x, y, w, h, r=4):
    """Horizontal bar, square at the baseline (left), 4px rounded data end (right)."""
    r = min(r, w / 2, h / 2)
    return (f"M{x:.1f},{y:.1f}H{x + w - r:.1f}Q{x + w:.1f},{y:.1f} {x + w:.1f},{y + r:.1f}"
            f"V{y + h - r:.1f}Q{x + w:.1f},{y + h:.1f} {x + w - r:.1f},{y + h:.1f}H{x:.1f}Z")


def scan_chart(rows, k_best, title):
    ks = [r["k"] for r in rows]
    sil = [r["silhouette"] for r in rows]
    p = Plot(280, 190, (min(ks) - 0.5, max(ks) + 0.5), (0, 1), left=40, right=10, top=14, bottom=36,
             xlabel="K (number of clusters)", yticks=[0, 0.25, 0.5, 0.75, 1], xticks=ks if len(ks) <= 8 else ks[::2])
    pts = " ".join(f"{p.X(k):.1f},{p.Y(s):.1f}" for k, s in zip(ks, sil))
    p.add(f'<polyline class="line1" points="{pts}"/>')
    for k, s in zip(ks, sil):
        best = k == k_best
        p.add(dot(p.X(k), p.Y(s), "s1" if best else "s1 faint", r=5 if best else 3, tip=f"K = {k}: silhouette {s:.3f}"))
        if best:
            lx = min(max(p.X(k), p.l + 60), p.w - p.r - 60)
            ly = p.Y(s) + 20 if s > 0.8 else p.Y(s) - 10
            p.add(f'<text class="ann strong" x="{lx:.1f}" y="{ly:.1f}" text-anchor="middle">chosen K = {k} ({s:.2f})</text>')
    return f'<div class="mini"><div class="minititle">{esc(title)}</div>{p.svg(title + " silhouette by K")}</div>'


def k1_charts(d):
    units = d["k1"]["units"]
    e = np.array([u["e"] for u in units])
    n = np.array([u["n"] for u in units])
    pad = 250
    p = Plot(W, 420, (e.min() - pad, e.max() + pad), (n.min() - pad, n.max() + pad), equal=True,
             xlabel="metres east of the common reference point", ylabel="metres north")
    seen = set()
    for u in units:
        cls = SERIES[u["area"]]
        p.add(dot(p.X(u["e"]), p.Y(u["n"]), cls, r=5, tip=f'{u["sensor"]} ({u["type"]}), {u["iter"]}: area {u["area"]}, site {u["site"]}'))
    for j, (cx, cy) in enumerate(d["k1"]["area_centers"]):
        X, Y = p.X(cx), p.Y(cy)
        members = sorted({u["sensor"] for u in units if u["area"] == j})
        p.add(f'<path class="cross" d="M{X - 9:.1f},{Y:.1f}H{X + 9:.1f}M{X:.1f},{Y - 9:.1f}V{Y + 9:.1f}"/>'
              f'<text class="ann" x="{X + 12:.1f}" y="{Y + 22:.1f}">area {j}: {len(members)} sensors</text>')
    overview = p.svg("Sensor positions coloured by the K1 area (K = 3)")
    # zoom on the busiest area with the 30 m sites
    busiest = Counter(u["area"] for u in units).most_common(1)[0][0]
    zu = [u for u in units if u["area"] == busiest]
    ze, zn = np.array([u["e"] for u in zu]), np.array([u["n"] for u in zu])
    radius = d["k1"]["site_radius_m"]
    span_e, span_n = np.ptp(ze) + 2 * radius + 40, np.ptp(zn) + 2 * radius + 40
    label_w = 330                                       # room on the right for the site labels
    plot_w = 880 - label_w
    zh = int(min(620, max(360, plot_w * span_n / span_e + 60)))
    z = Plot(plot_w, zh, (ze.min() - radius - 20, ze.max() + radius + 20), (zn.min() - radius - 20, zn.max() + radius + 20),
             equal=True, xlabel="metres east", ylabel="metres north", right=10)
    sites_here = sorted({u["site"] for u in zu})
    for s_id in sites_here:
        cx, cy = d["k1"]["site_centers"][s_id]
        rr = abs(z.X(cx + radius) - z.X(cx))
        z.add(f'<circle class="sitering" cx="{z.X(cx):.1f}" cy="{z.Y(cy):.1f}" r="{rr:.1f}"/>')
    for s_id in sites_here:
        mem = [u for u in zu if u["site"] == s_id]
        for sensor in sorted({u["sensor"] for u in mem}):
            us = [u for u in mem if u["sensor"] == sensor]
            x, y = float(np.mean([u["e"] for u in us])), float(np.mean([u["n"] for u in us]))
            z.add(dot(z.X(x), z.Y(y), SERIES[busiest], r=5, tip=f'{sensor} ({us[0]["type"]}) — site {s_id}, iterations {", ".join(sorted({u["iter"] for u in us}))}'))
    # one label per site in a column to the right, joined to its ring by a leader line
    labels = []
    for s_id in sites_here:
        cx, cy = d["k1"]["site_centers"][s_id]
        ids = sorted({u["sensor"] for u in zu if u["site"] == s_id})
        labels.append([z.Y(cy), s_id, cx, cy, ids])
    labels.sort()
    min_gap, ys = 44, []
    for lab in labels:
        y = max(lab[0], (ys[-1] + min_gap) if ys else z.t + 14)
        ys.append(y)
    overflow = ys[-1] - (zh - z.b - 10) if ys else 0
    if overflow > 0:
        ys = [y - overflow for y in ys]
    lx = plot_w + 24
    for (yc, s_id, cx, cy, ids), y in zip(labels, ys):
        rr = abs(z.X(cx + radius) - z.X(cx))
        z.add(f'<path class="guide" d="M{z.X(cx) + rr:.1f},{z.Y(cy):.1f}L{lx - 6:.1f},{y - 4:.1f}"/>'
              f'<text class="ann strong" x="{lx:.1f}" y="{y:.1f}">site {s_id} · {len(ids)} sensor{"s" if len(ids) > 1 else ""}</text>'
              f'<text class="ann" x="{lx:.1f}" y="{y + 16:.1f}">{esc(", ".join(ids))}</text>')
    z.w = 880                                           # widen the canvas to include the label column
    zoom = z.svg("Zoom on the busiest area with the 30 m sites")
    rows = [[u["iter"], u["sensor"], u["type"], u["e"], u["n"], u["area"], u["site"]] for u in units]
    return overview, zoom, rows, busiest


def k2_chart(d):
    groups = d["k2"]["groups"]
    units = sorted(d["k2"]["units"], key=lambda u: (u["cluster"], u["sensor_id"], u["iter_key"]))
    row_h, left = 20, 190
    n_clusters = len({u["cluster"] for u in units})
    h = 16 + (len(units) + n_clusters) * row_h + 40
    p = Plot(W, h, (0, 100), (0, 1), left=left, right=20, top=10, bottom=40, xticks=[0, 25, 50, 75, 100], yticks=[],
             xfmt=lambda v: f"{v:g}%", xlabel="share of the sensor's detections in each class group")
    y, cur = 10, None
    for u in units:
        if u["cluster"] != cur:
            cur = u["cluster"]
            members = sorted({x["sensor_id"] for x in units if x["cluster"] == cur})
            p.add(f'<text class="tier" x="8" y="{y + 14}">Group {cur} — {", ".join(members)}</text>')
            y += row_h
        p.add(f'<text class="tick" x="{left - 8}" y="{y + 14}" text-anchor="end">{esc(u["sensor_id"])} · {esc(u["iter_key"].split("/")[1])}</text>')
        x = p.X(0)
        for gi, g in enumerate(groups):
            v = float(u[g]) * 100
            if v <= 0:
                continue
            wpx = p.X(v) - p.X(0)
            gap = 2 if wpx > 3 else 0                     # 2px surface gap between stacked segments
            p.add(f'<rect class="{SERIES[gi]}" x="{x:.1f}" y="{y + 4}" width="{max(wpx - gap, 0.8):.1f}" height="{row_h - 8}" '
                  f'data-tip="{esc(u["sensor_id"])} {esc(u["iter_key"].split("/")[1])}: {g} {v:.0f}% of {u["rows"]} detections" tabindex="0"/>')
            x += wpx
        y += row_h
    leg = legend([(SERIES[i], g.replace("_", " ")) for i, g in enumerate(groups)])
    rows = [[u["cluster"], u["sensor_id"], u["iter_key"].split("/")[1], u["rows"]] + [f"{100 * float(u[g]):.0f}%" for g in groups] for u in units]
    return p.svg("Class mix per sensor, grouped by K2 cluster"), leg, rows, groups


def k3_charts(d):
    pts = d["k3"]["points"]
    x = np.array([p_[0] for p_ in pts]) / 1000
    y = np.array([p_[1] for p_ in pts]) / 1000
    lab = np.array([p_[2] for p_ in pts])
    sens = [p_[3] for p_ in pts]
    xl = (np.percentile(x, 0.5) - 0.3, np.percentile(x, 99.5) + 0.3)
    yl = (np.percentile(y, 0.5) - 0.3, np.percentile(y, 99.5) + 0.3)
    site_units = d["k1"]["units"]
    panels = []
    sizes = d["k3"]["sizes"]
    for j in range(len(d["k3"]["centers"])):
        p = Plot(430, 360, xl, yl, left=44, right=10, top=12, bottom=36, equal=True,
                 xlabel="km east", ylabel="km north", xfmt=lambda v: f"{v:g}", yfmt=lambda v: f"{v:g}")
        other = lab != j
        for xi, yi in zip(x[other], y[other]):
            if xl[0] <= xi <= xl[1] and yl[0] <= yi <= yl[1]:
                p.add(f'<circle class="bg" cx="{p.X(xi):.1f}" cy="{p.Y(yi):.1f}" r="1.6"/>')
        for xi, yi in zip(x[~other], y[~other]):
            if xl[0] <= xi <= xl[1] and yl[0] <= yi <= yl[1]:
                p.add(f'<circle class="s1 pt" cx="{p.X(xi):.1f}" cy="{p.Y(yi):.1f}" r="1.8"/>')
        for u in site_units:
            X, Y = p.X(u["e"] / 1000), p.Y(u["n"] / 1000)
            p.add(f'<path class="sensor" d="M{X:.1f},{Y - 5:.1f}L{X + 4.5:.1f},{Y + 3.5:.1f}L{X - 4.5:.1f},{Y + 3.5:.1f}Z"/>')
        cx, cy = d["k3"]["centers"][j]
        X, Y = p.X(cx / 1000), p.Y(cy / 1000)
        p.add(f'<path class="cross" d="M{X - 8:.1f},{Y:.1f}H{X + 8:.1f}M{X:.1f},{Y - 8:.1f}V{Y + 8:.1f}"/>')
        n_sens = len({s for s, l_ in zip(sens, lab) if l_ == j})
        panels.append(f'<div class="mini wide"><div class="minititle">Region {j} — {sizes[j]:,} samples, {n_sens} distinct sensors in the drawing sample</div>'
                      f'{p.svg(f"K3 region {j}")}</div>')
    return panels


def combos_chart(d):
    combos = d["same_time"]["combos"][:12]
    row_h, left = 24, 250
    h = 10 + len(combos) * row_h + 40
    mx = max(c["n_groups"] for c in combos)
    p = Plot(W, h, (0, mx * 1.12), (0, 1), left=left, right=40, top=10, bottom=40, yticks=[],
             xlabel=f"number of 1-second moments where these sensors detected something within {d['same_time']['gate_m']:g} m of each other")
    for i, c in enumerate(combos):
        y = 10 + i * row_h
        p.add(f'<text class="tick" x="{left - 8}" y="{y + 16}" text-anchor="end">{esc(c["combo"])}</text>')
        x0, x1 = p.X(0), p.X(c["n_groups"])
        p.add(f'<path class="s1" d="{bar_path(x0, y + 5, x1 - x0, row_h - 10)}" data-tip="{esc(c["combo"])}: {c["n_groups"]} moments" tabindex="0"/>')
        p.add(f'<text class="val" x="{x1 + 6:.1f}" y="{y + 16}">{c["n_groups"]}</text>')
    return p.svg("Sensors seen together at the same time and place")


def pairs_heatmap(d):
    pairs = d["same_time"]["pairs"]
    sensors = sorted({x["a"] for x in pairs} | {x["b"] for x in pairs})
    share = {(x["a"], x["b"]): (x["n_a_with_b"] / x["n_a"] if x["n_a"] else 0.0, x["n_a_with_b"], x["n_a"]) for x in pairs}
    cell, left, top = 46, 56, 30
    w = left + cell * len(sensors) + 20
    h = top + cell * len(sensors) + 20
    ramp = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
    parts = []
    for j, b in enumerate(sensors):
        parts.append(f'<text class="tick" x="{left + j * cell + cell / 2:.1f}" y="{top - 8}" text-anchor="middle">{b}</text>')
    for i, a in enumerate(sensors):
        parts.append(f'<text class="tick" x="{left - 8}" y="{top + i * cell + cell / 2 + 4:.1f}" text-anchor="end">{a}</text>')
        for j, b in enumerate(sensors):
            x, y = left + j * cell, top + i * cell
            if a == b:
                parts.append(f'<rect class="diag" x="{x + 1}" y="{y + 1}" width="{cell - 2}" height="{cell - 2}"/>')
                continue
            v, hit, n = share.get((a, b), (0.0, 0, 0))
            if hit == 0:
                parts.append(f'<rect class="empty" x="{x + 1}" y="{y + 1}" width="{cell - 2}" height="{cell - 2}" data-tip="{a} → {b}: no shared detections" tabindex="0"/>')
                continue
            k = min(int(math.floor(v * len(ramp))), len(ramp) - 1)
            fill = ramp[k]
            ink = "#0b0b0b" if k <= 2 else "#ffffff"
            parts.append(f'<rect x="{x + 1}" y="{y + 1}" width="{cell - 2}" height="{cell - 2}" style="fill:{fill}" rx="3" '
                         f'data-tip="{a} → {b}: {hit} of {n} {a} detections ({100 * v:.1f}%) have a {b} detection within 1 s and 50 m" tabindex="0"/>'
                         f'<text x="{x + cell / 2:.1f}" y="{y + cell / 2 + 4:.1f}" text-anchor="middle" style="fill:{ink};font-size:11px">{100 * v:.0f}%</text>')
    svg = (f'<p class="how">Rows = sensor A, columns = sensor B.</p><svg viewBox="0 0 {w} {h}" style="max-width:{w}px" role="img" '
           f'aria-label="pair heatmap"><title>Same-time pair shares</title>{"".join(parts)}</svg>')
    scale = '<div class="legend">' + "".join(f'<span><i class="sw" style="background:{c}"></i>{int(100 * i / len(ramp))}–{int(100 * (i + 1) / len(ramp))}%</span>' for i, c in enumerate(ramp)) + "</div>"
    rows = [[x["a"], x["b"], x["n_a"], x["n_a_with_b"], f"{100 * x['n_a_with_b'] / x['n_a']:.1f}%" if x["n_a"] else "-"] for x in pairs if x["n_a_with_b"]]
    return svg, scale, rows


def k4_charts(d):
    tr = d["k4"]["tracks"]
    feats = d["k4"]["features"]
    dur = np.array([10 ** t["f"][0] for t in tr])
    spd = np.array([10 ** t["f"][1] - 1 for t in tr])
    lab = np.array([t["cluster"] for t in tr])
    p = Plot(W, 400, (math.log10(2), math.log10(max(dur.max(), 10) * 1.2)), (0, min(np.percentile(spd, 99.5) * 1.1, 60)),
             xlabel="track duration (seconds, log scale)", ylabel="median speed (m/s)",
             xticks=[math.log10(v) for v in (2, 5, 10, 30, 100, 300, 1000) if v <= max(dur.max(), 10) * 1.2],
             xfmt=lambda v: f"{10 ** v:.0f}")
    order = np.argsort(-np.bincount(lab)[lab], kind="stable")      # big cluster first, small ones on top
    for i in order:
        if spd[i] <= p.y1:
            t = tr[i]
            p.add(dot(p.X(math.log10(dur[i])), p.Y(spd[i]), SERIES[lab[i]], r=3.5,
                      tip=f'{t["sensor"]} {t["iter"]}: cluster {lab[i]}, {dur[i]:.0f} s, {spd[i]:.1f} m/s, UAV share {t["f"][5]:.2f}, Air Vehicle {t["f"][6]:.2f}, bio {t["f"][7]:.2f}'))
    scatter = p.svg("Tracks: duration vs speed, coloured by K4 cluster")
    # profile: robust-scaled centroid per feature (0 = a typical track)
    cen = np.array(d["k4"]["centers_scaled"])
    lo, hi = min(-1.0, cen.min() - 0.5), max(1.0, cen.max() + 0.5)
    row_h, left = 30, 210
    h = 20 + len(feats) * row_h + 44
    q = Plot(W, h, (lo, hi), (0, 1), left=left, right=30, top=20, bottom=44, yticks=[],
             xlabel="cluster centre, in robust units (0 = the typical track; ±1 ≈ one spread away)")
    names = {"log_duration": "duration", "log_speed_med": "median speed", "log_speed_p90": "fast-speed (p90)",
             "straightness": "straightness", "log_turn_rate": "turn rate", "frac_uav": "sensor said UAV",
             "frac_air_vehicle": "sensor said Air Vehicle", "frac_bio": "sensor said bird / animal", "frac_unknown": "sensor said unknown"}
    q.add(f'<line class="zero" x1="{q.X(0):.1f}" x2="{q.X(0):.1f}" y1="20" y2="{h - 44}"/>')
    for fi, f in enumerate(feats):
        y = 20 + fi * row_h + row_h / 2
        q.add(f'<line class="grid" x1="{left}" x2="{W - 30}" y1="{y:.1f}" y2="{y:.1f}"/>'
              f'<text class="tick" x="{left - 8}" y="{y + 4:.1f}" text-anchor="end">{esc(names.get(f, f))}</text>')
        for j in range(len(cen)):
            q.add(dot(q.X(cen[j][fi]), y + (j - 1) * 6, SERIES[j], r=5, tip=f"cluster {j}: {names.get(f, f)} = {cen[j][fi]:+.2f} robust units"))
    profile = q.svg("What defines each K4 cluster")
    counts = Counter(lab.tolist())
    leg = legend([(SERIES[j], f"cluster {j} ({counts[j]:,} tracks)") for j in sorted(counts)])
    by_sensor = defaultdict(Counter)
    for t in tr:
        by_sensor[t["sensor"]][t["cluster"]] += 1
    rows = [[s] + [c.get(j, 0) for j in sorted(counts)] for s, c in sorted(by_sensor.items())]
    return scatter, profile, leg, rows, sorted(counts)


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #

CSS = """
.viz-root{color-scheme:light;--surface-1:#fcfcfb;--page:#f9f9f7;--text-primary:#0b0b0b;--text-secondary:#52514e;--muted:#898781;
--grid:#e1e0d9;--axis:#c3c2b7;--border:rgba(11,11,11,.10);--bg-pt:#d9d8d2;--diag:#f0efec;
--series-1:#2a78d6;--series-2:#eb6834;--series-3:#1baf7a;--series-4:#eda100;--series-5:#e87ba4;--series-6:#008300;--series-7:#6250d6}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .viz-root{color-scheme:dark;--surface-1:#1a1a19;--page:#0d0d0d;
--text-primary:#f0efec;--text-secondary:#c3c2b7;--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);--bg-pt:#3a3a37;--diag:#2c2c2a;
--series-1:#3987e5;--series-2:#d95926;--series-3:#199e70;--series-4:#c98500;--series-5:#d55181;--series-6:#008300;--series-7:#9085e9}}
:root[data-theme="dark"] .viz-root{color-scheme:dark;--surface-1:#1a1a19;--page:#0d0d0d;--text-primary:#f0efec;--text-secondary:#c3c2b7;
--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);--bg-pt:#3a3a37;--diag:#2c2c2a;
--series-1:#3987e5;--series-2:#d95926;--series-3:#199e70;--series-4:#c98500;--series-5:#d55181;--series-6:#008300;--series-7:#9085e9}
html,body{margin:0;background:var(--page)}
body{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;color:var(--text-primary);line-height:1.5}
.viz-root{background:var(--page);padding:24px 16px 64px}
main{max-width:960px;margin:0 auto}
h1{font-size:26px;margin:0 0 6px}h2{font-size:20px;margin:40px 0 8px}h3{font-size:16px;margin:0 0 6px}
p{margin:6px 0;color:var(--text-secondary)} b{color:var(--text-primary)}
.card{background:var(--surface-1);border:1px solid var(--border);border-radius:12px;padding:18px 18px 12px;margin:14px 0}
.what{color:var(--text-primary)} .how{font-size:14px}
.fig svg{width:100%;height:auto;display:block}
.minis{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:12px}.minis.wide{grid-template-columns:repeat(auto-fill,minmax(380px,1fr))}
.minititle{font-size:13px;color:var(--text-secondary);margin:4px 0}
svg text{font-family:inherit} .tick{fill:var(--text-secondary);font-size:12px;font-variant-numeric:tabular-nums}
.lab{fill:var(--muted);font-size:12px} .val{fill:var(--text-secondary);font-size:12px}
.tier{fill:var(--text-primary);font-size:13px;font-weight:600} .ann{fill:var(--text-secondary);font-size:12px} .ann.strong{fill:var(--text-primary);font-weight:600}
.ptlab{fill:var(--text-primary);font-size:12px}
.grid{stroke:var(--grid);stroke-width:1}.axis{stroke:var(--axis);stroke-width:1}.zero{stroke:var(--axis);stroke-width:1.5}
.guide{stroke:var(--axis);stroke-width:1}.arrow{stroke:var(--text-secondary);stroke-width:1.5}
.cross{stroke:var(--text-primary);stroke-width:2;fill:none;stroke-linecap:round}
.sitering{fill:none;stroke:var(--text-secondary);stroke-width:1}
.sensor{fill:var(--text-primary)} .bg{fill:var(--bg-pt)} .pt{opacity:.75}
.s1{fill:var(--series-1)}.s2{fill:var(--series-2)}.s3{fill:var(--series-3)}.s4{fill:var(--series-4)}.s5{fill:var(--series-5)}.s6{fill:var(--series-6)}.s7{fill:var(--series-7)}
.faint{opacity:.45} .ring{stroke:var(--surface-1);stroke-width:2}
.line1{fill:none;stroke:var(--series-1);stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.diag{fill:var(--diag)} .empty{fill:none;stroke:var(--grid)}
.legend{display:flex;flex-wrap:wrap;gap:6px 16px;font-size:13px;color:var(--text-secondary);margin:8px 0}
.legend span{display:inline-flex;align-items:center;gap:6px}.sw{width:12px;height:12px;border-radius:3px;display:inline-block}
.sw.s1{background:var(--series-1)}.sw.s2{background:var(--series-2)}.sw.s3{background:var(--series-3)}.sw.s4{background:var(--series-4)}
.sw.s5{background:var(--series-5)}.sw.s6{background:var(--series-6)}.sw.s7{background:var(--series-7)}
details{margin:8px 0}summary{cursor:pointer;color:var(--text-secondary);font-size:14px}
.tw{overflow-x:auto}table{border-collapse:collapse;font-size:13px;margin-top:6px;font-variant-numeric:tabular-nums}
th,td{border-bottom:1px solid var(--grid);padding:4px 10px;text-align:left;color:var(--text-primary)}th{color:var(--text-secondary);font-weight:600}
dl.gloss{display:grid;grid-template-columns:minmax(120px,190px) 1fr;gap:8px 16px;margin:8px 0}
dl.gloss dt{font-weight:600}dl.gloss dd{margin:0;color:var(--text-secondary)}
.two{display:flex;flex-wrap:wrap;gap:12px}.two>div{flex:1 1 380px}
#tip{position:fixed;pointer-events:none;background:var(--surface-1);color:var(--text-primary);border:1px solid var(--border);
border-radius:8px;padding:6px 10px;font-size:13px;max-width:360px;box-shadow:0 2px 10px rgba(0,0,0,.12);display:none;z-index:9}
[data-tip]:focus{outline:2px solid var(--text-primary);outline-offset:1px}
.toggle{float:right;font:inherit;font-size:13px;background:var(--surface-1);color:var(--text-primary);border:1px solid var(--border);border-radius:8px;padding:4px 10px;cursor:pointer}
@media (max-width:600px){dl.gloss{grid-template-columns:1fr}}
"""

JS = """
const tip=document.getElementById('tip');
function show(e,t){tip.textContent=t;tip.style.display='block';const r=t.length>60?380:240;
 let x=(e.clientX??e.target.getBoundingClientRect().right)+14,y=(e.clientY??e.target.getBoundingClientRect().top)+14;
 if(x+r>innerWidth)x=innerWidth-r-8;tip.style.left=x+'px';tip.style.top=y+'px';}
document.querySelectorAll('[data-tip]').forEach(el=>{
 el.addEventListener('mousemove',e=>show(e,el.dataset.tip));el.addEventListener('mouseleave',()=>tip.style.display='none');
 el.addEventListener('focus',e=>show({target:el},el.dataset.tip));el.addEventListener('blur',()=>tip.style.display='none');});
document.getElementById('theme').addEventListener('click',()=>{const r=document.documentElement;
 const dark=r.dataset.theme?r.dataset.theme==='dark':matchMedia('(prefers-color-scheme: dark)').matches;r.dataset.theme=dark?'light':'dark';});
"""


def main() -> int:
    d = json.loads((EVAL / "plot_data.json").read_text())
    ladder = list(csv.DictReader((EVAL / "field_ladder.csv").open()))
    n_doc = int(round(100 * int(ladder[0]["n_pdf"]) / float(ladder[0]["pct_pdf"])))
    sc = d["scans"]

    S = []
    S.append(f'<header><button class="toggle" id="theme" type="button">Light / dark</button><h1>Evaluation data — what the K-means clusters mean</h1>'
             f'<p>Only the three <b>evaluation</b> iterations (DI, DII, DIII) are used; no testing data, labels or calibration. '
             f'Hover or tab onto any bar or dot for its exact value; every chart has a data table under it. '
             f'Numbers come from <code>tools/eval_tiers.py</code>; this page from <code>tools/eval_charts.py</code>.</p></header>')

    # ---- glossary
    S.append("<h2>1. The words, in plain language</h2>")
    gloss = [
        ("Detection", "One line of a sensor file: the sensor saying “I see something here, now”. The evaluation data has about 194,000 of them."),
        ("Track", "All detections a sensor gives the same <code>objectId</code>: the sensor's own idea of “one object over time”."),
        ("Field", "A column of the sensor file, e.g. <code>timestamp</code>, <code>sensor_location_x</code>, <code>classification</code>."),
        ("Tier", "A rung of the field ladder: the fields that the same number of sensors have. Tier 1 = fields every sensor has (100 %), the last tier = fields only one sensor has."),
        ("Feature", "A number that describes one item for the clustering, e.g. east position, north position, median speed, share of detections the sensor called “UAV”."),
        ("K-means", "An algorithm that splits items into K groups so that items in a group are close to each other. It repeats two steps: put every item in the group of the nearest centre; move every centre to the middle of its group."),
        ("Cluster", "One of those K groups. It is a grouping by similarity, not a verified object class: the algorithm does not know what a drone is."),
        ("Centroid", "The centre (average) of a cluster — the cross (+) in the charts."),
        ("K", "How many clusters we ask for. We try several K and keep the one with the best silhouette."),
        ("Silhouette", "How well separated the clusters are, from −1 to 1. For each item: a = its average distance to its own cluster, b = to the nearest other cluster; silhouette = (b − a) / max(a, b), averaged. Near 1 = clean, separate groups; near 0 = no real groups, just an arbitrary cut of a continuous cloud."),
        ("Robust scaling", "Features have different units (seconds, m/s, shares). Before K-means each is rescaled: 0 = the median item, 1 = one inter-quartile range. So no feature dominates just because of its unit."),
        ("Site", "A cluster of sensor positions (tier 1) whose members are all within 30 m of its centre: sensors standing together = one multi-sensor site."),
        ("Region", "A K3 cluster of detection positions over the whole recording: an area several sensors' detections fall into. Not one object."),
        ("Same-time intersection", "The real “two sensors see the same thing” test: detections of different sensors within 1 second and 50 m of each other."),
    ]
    S.append('<section class="card"><dl class="gloss">' + "".join(f"<dt>{esc(a)}</dt><dd>{b}</dd>" for a, b in gloss) + "</dl></section>")
    tsvg, ta, tb, ts = toy_silhouette_svg()
    S.append('<div class="two"><div>' + card("How K-means groups points (toy example)",
             "Synthetic points, not data: K-means with K = 3 finds three groups and their centres (+).", toy_kmeans_svg(),
             how="each colour is one cluster; the cross is its centroid.") + "</div><div>" +
             card("What the silhouette measures (toy example)",
                  f"Synthetic points, not data: the big dot, its own group (blue) and the nearest other group (orange). "
                  f"Its silhouette = (b − a) / max(a, b) = ({tb:.1f} − {ta:.1f}) / {max(ta, tb):.1f} = <b>{ts:.2f}</b>.",
                  tsvg, how="the dot belongs clearly to blue when b is much larger than a — then its silhouette is close to 1. "
                            "The silhouette of a clustering is this number averaged over all points.") + "</div></div>")

    # ---- ladder
    S.append("<h2>2. The field ladder — which fields are everywhere, which are rare</h2>")
    S.append(card(f"Share of the {n_doc} evaluation sensors documented in README_Sensor.pdf that have each field",
                  "Tier 1 (top) is what every sensor provides: time and the sensor's own position. Each tier further down is shared by fewer sensors.",
                  ladder_chart(ladder, n_doc),
                  table(["tier", "field", "sensors (PDF)", "%", "sensors (data headers)", "which sensors"],
                        [[r["tier"], r["field"], r["n_pdf"], r["pct_pdf"], r["n_data"], r["sensors"]] for r in ladder]),
                  how="bar length = how many of the sensors have that field. Hover a bar to see which sensors are missing it."))

    # ---- K choice
    S.append("<h2>3. How K was chosen for each clustering</h2>")
    minis = [scan_chart(sc["k1"]["rows"], sc["k1"]["k_best"], "K1 · sensor positions"),
             scan_chart(sc["k2"]["rows"], sc["k2"]["k_best"], "K2 · class mix per sensor"),
             scan_chart(sc["k3"]["rows"], sc["k3"]["k_best"], "K3 · detection positions"),
             scan_chart(sc["k3_no_egoi"]["rows"], sc["k3_no_egoi"]["k_best"], "K3 without EGOI"),
             scan_chart(sc["k4"]["rows"], sc["k4"]["k_best"], "K4 · tracks")]
    S.append(card("Silhouette score for each K tried", "Each small chart is one clustering. The highlighted dot is the K we kept (highest silhouette).",
                  '<div class="minis">' + "".join(minis) + "</div>",
                  table(["clustering", "K", "silhouette", "inertia"],
                        [[name, r["k"], r["silhouette"], r["inertia"]] for name, key in
                         [("K1", "k1"), ("K2", "k2"), ("K3", "k3"), ("K3 without EGOI", "k3_no_egoi"), ("K4", "k4")] for r in sc[key]["rows"]]),
                  how="higher is better. K1, K2 and K4 reach 0.7–0.9 (clear groups); K3 stays near 0.4 (weak — the detection cloud has no sharp regions)."))

    # ---- K1
    overview, zoom, rows1, busiest = k1_charts(d)
    no_site = ", ".join(d["k1"]["no_site"])
    S.append("<h2>4. K1 — tier 1 fields (sensor position): where the sensors stand</h2>")
    S.append(card("All sensor positions, coloured by area (K = 3)",
                  f"One dot per sensor per iteration. K-means finds three areas a few hundred metres to 2 km apart. {esc(no_site)} report position 0,0 (UTM feeds without a site) and are not drawn.",
                  overview, legend([(SERIES[j], f"area {j}") for j in range(len(d["k1"]["area_centers"]))]),
                  how="dots of one colour belong to one area; the cross is the area's centroid; the label says how many distinct sensors it holds."))
    S.append(card(f"Zoom on area {busiest}: the 30 m sites",
                  f"Inside the busiest area, K-means with more clusters (K = {d['k1']['k_site']}) splits the sensors into sites where every sensor is within {d['k1']['site_radius_m']:g} m of the centre. "
                  "A site with several distinct sensors is a multi-sensor site.", zoom,
                  table(["iteration", "sensor", "type", "east (m)", "north (m)", "area", "site"], rows1),
                  how="each ring is one site (30 m radius); the dots inside are the sensors standing there."))

    # ---- K2
    svg2, leg2, rows2, groups = k2_chart(d)
    S.append("<h2>5. K2 — tier 2 field (classification): what each sensor reports</h2>")
    S.append(card("Class mix of each sensor, grouped by K2 cluster",
                  "One bar per sensor and iteration; the coloured parts are the shares of its detections in each class group. K-means groups sensors that report the same kind of things.",
                  svg2, leg2 + table(["group", "sensor", "iteration", "detections"] + groups, rows2),
                  how="read a bar left to right: e.g. a fully blue bar = a sensor that only ever says “UAV”. Sensors in the same group have similar bars."))

    # ---- K3
    S.append("<h2>6. K3 — tier 3 fields (object position): where the detections fall</h2>")
    S.append(card(f"The {len(d['k3']['centers'])} regions found by K-means, one panel each",
                  f"Each panel highlights one region in blue over all other detections in grey; triangles are the sensors. "
                  f"Drawing sample: {len(d['k3']['points']):,} of {d['k3']['n_total']:,} detection samples. Time is ignored, so a region is an area over the whole recording, not an object.",
                  '<div class="minis wide">' + "".join(k3_charts(d)) + "</div>",
                  how="the silhouette is only about 0.4, so these regions are a coarse division of one large cloud. "
                      "Most far-out detections are EGOI radar returns; the central region is where all sensors' detections overlap."))

    # ---- same time
    svg_h, scale_h, rows_h = pairs_heatmap(d)
    S.append("<h2>7. The real intersections — same time, same place (not K-means)</h2>")
    gb = d["same_time"]["groups_by_size"]
    tot = sum(gb.values())
    multi = sum(v for k, v in gb.items() if int(k) >= 2)
    S.append(card("Which sensors see something together",
                  f"Every 1-second moment, detections of different sensors within {d['same_time']['gate_m']:g} m are joined. "
                  f"{multi:,} of {tot:,} moments ({100 * multi / tot:.1f} %) involve two or more distinct sensors. No clock correction is applied.",
                  combos_chart(d), table(["sensors seen together", "moments"], [[c["combo"], c["n_groups"]] for c in d["same_time"]["combos"]]),
                  how="a longer bar = these sensors agreed more often that something was at the same spot at the same second."))
    S.append(card("Pairwise: how often sensor B confirms sensor A", "Each cell: share of row-sensor A's detections that column-sensor B also detected at the same second within 50 m.",
                  svg_h, scale_h + table(["A", "B", "A detections", "with B", "share"], rows_h),
                  how="darker = B confirms A more often. The matrix is not symmetric: a sparse sensor can be confirmed often by a dense one, not the other way round."))

    # ---- K4
    scatter, profile, leg4, rows4, cl4 = k4_charts(d)
    kin = d["k4"]["sil_kinematic_only"]
    clo = d["k4"]["sil_class_only"]
    S.append("<h2>8. K4 — velocity fields (tier 7): pre-classification of tracks</h2>")
    S.append(card("Every track: how long it lasted and how fast it moved, coloured by cluster",
                  "One dot per track of the sensors that report velocity. Hover a dot for its sensor and class shares.",
                  scatter, leg4 + table(["sensor"] + [f"cluster {j}" for j in cl4], rows4, "Tracks per sensor and cluster"),
                  how="the clusters overlap a lot in speed and duration — motion alone does not separate them (see below)."))
    S.append(card("What defines each cluster",
                  f"Cluster centres for every feature, in robust units (0 = a typical track). K-means on these 9 features gives silhouette {sc['k4']['rows'][0]['silhouette']:.2f}; "
                  f"on the 5 motion features alone only {kin['silhouette']:.2f}; on the 4 'what the sensor said' shares alone {clo['silhouette']:.2f}.",
                  profile, leg4,
                  how="a dot far right = that cluster has much more of the feature than a typical track. The clusters are separated mainly by what the sensor itself "
                      "called the track (UAV / Air Vehicle, bird, unknown), not by how it moved. The 'UAV / Air Vehicle' cluster is a candidate list, not a verified class: "
                      "the evaluation data has no ground truth."))

    page = (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>Evaluation cluster charts</title><style>{CSS}</style></head><body><div class="viz-root"><main>{"".join(S)}</main></div>'
            f'<div id="tip" role="status"></div><script>{JS}</script></body></html>')
    OUT.write_text(page, encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
