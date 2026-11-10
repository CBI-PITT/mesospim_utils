"""BigStitcher alignment scoring: log + XML metrics and a report-only CLI.

Phase 1 of the BigStitcher parameter-automation roadmap
(`/h20/home/lab/.opencode/plan/bigstitcher-parameter-automation.md`).

Parses SLURM BigStitcher logs into analysis-doc section-7 metrics, reads final
tile positions from the stitched XML the log produced, fits a scaled grid
(scale ratio `s` + translation per axis) to estimate the zoom/ETL pixel-size
miscalibration, and combines everything into a composite alignment-quality
score. Report-only: nothing is executed and no files are written.

The score is a weighted mix of residual, connectivity, dropped links, signal
health, and ICP health. It is never residual alone - Avg Error drops to zero
when every link is removed, so dropped links and connectivity are scored
separately.

Stdlib only: runnable with any Python 3.9+ without the mesospim_utils
environment (logs and XMLs can be graded from copies on any machine).

Usage:
    python mesospim_utils/bigstitcher_scoring.py <dataset_dir or log_path> [...]
    python mesospim_utils/bigstitcher_scoring.py --json <...>
"""
import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

LOG_SUFFIX_GLOB = "*_align_*bigstitcher.log"
# legacy combined runs use the _align_fuse suffix; align-only runs use
# _align_only_bigstitcher. Both are matched by the glob above.

_AVG_MAX_ERR_RE = re.compile(r"Avg Error[:=]\s*([\d.]+)px.*?Max Error[:=]\s*([\d.]+)px", re.DOTALL)
_REMOVED_LINK_RE = re.compile(r"Removed link .*?error=([\d.]+)")
_ADDED_LINK_RE = re.compile(r"added strong link between")
_PEAKS_RE = re.compile(r"Found (\d+) initial peaks")
_NO_INTERESTPOINTS_RE = re.compile(r"WARNING: no interestpoints available")
_ICP_NOT_ENOUGH_RE = re.compile(r"Not enough corresponding points found")
_ICP_NO_CORRESPONDING_RE = re.compile(r"No corresponding points\.")
_VIEW_INTENSITY_RE = re.compile(r"View tpId=\d+ setupId=(\d+), Min=([\d.]+) max=([\d.]+)")
_DOWNSAMPLING_RE = re.compile(r"Correcting coordinates for downsampling[^\n]*")
_SAVED_XML_RE = re.compile(r"Saved xml 'file:([^']+)'")
_PARAMS_RE = re.compile(r"BIGSTITCHER_PARAMS: ([^\n]+)")
_BS_VERSION_RE = re.compile(r"[Bb]ig[_ ]?[Ss]titcher[-_ ](\d+\.\d+\.\d+)")
_MVRECON_VERSION_RE = re.compile(r"multiview-reconstruction version: (\d+\.\d+\.\d+)")


def parse_align_log(log_path) -> dict:
    """Parse one BigStitcher SLURM log into analysis-doc section-7 metrics.

    ``first_round_*`` is the first optimization round (cold-start state),
    ``final_*`` the last (converged) round. ``n_views`` comes from the
    per-view intensity lines (absent in some versions). ``params`` holds the
    effective macro values from the ``BIGSTITCHER_PARAMS:`` line.
    """
    log_path = Path(log_path)
    text = log_path.read_text(errors="replace")

    m = {
        "log": str(log_path),
        "bigstitcher_version": None,
        "mvrecon_version": None,
        "first_round_avg_error": None,
        "first_round_max_error": None,
        "final_avg_error": None,
        "final_max_error": None,
        "opt_rounds": 0,
        "dropped_links": 0,
        "dropped_link_errors": [],
        "added_strong_links": 0,
        "dog_peaks_all": [],
        "dog_peaks_min": None,
        "dog_peaks_max": None,
        "zero_detection_views": 0,
        "icp_not_enough": 0,
        "icp_no_corresponding": 0,
        "view_intensity": [],
        "n_views": None,
        "starved_views": 0,
        "downsampling_lines": [],
        "params": {},
        "params_line": None,
        "saved_xml": None,
        "success_marker": "BIGSTITCHER_SUCCESS:" in text,
        "align_success_marker": "BIGSTITCHER_ALIGN_SUCCESS:" in text,
        "fusion_banner": "Creating Fused Dataset" in text,
        "exceptions": len(re.findall(r"(?i)exception", text)),
    }

    bs = _BS_VERSION_RE.search(text) or re.search(r"BigStitcher[-_ ](\d+\.\d+\.\d+)", text)
    if bs:
        m["bigstitcher_version"] = bs.group(1)
    mv = _MVRECON_VERSION_RE.search(text)
    if mv:
        m["mvrecon_version"] = mv.group(1)

    errs = _AVG_MAX_ERR_RE.findall(text)
    if errs:
        m["first_round_avg_error"] = float(errs[0][0])
        m["first_round_max_error"] = float(errs[0][1])
        m["final_avg_error"] = float(errs[-1][0])
        m["final_max_error"] = float(errs[-1][1])
        m["opt_rounds"] = len(errs)

    drops = _REMOVED_LINK_RE.findall(text)
    m["dropped_links"] = len(drops)
    m["dropped_link_errors"] = [float(e) for e in drops]
    m["added_strong_links"] = len(_ADDED_LINK_RE.findall(text))

    peaks = _PEAKS_RE.findall(text)
    m["dog_peaks_all"] = [int(p) for p in peaks]
    if peaks:
        m["dog_peaks_min"] = min(int(p) for p in peaks)
        m["dog_peaks_max"] = max(int(p) for p in peaks)

    m["zero_detection_views"] = len(_NO_INTERESTPOINTS_RE.findall(text))
    m["icp_not_enough"] = len(_ICP_NOT_ENOUGH_RE.findall(text))
    m["icp_no_corresponding"] = len(_ICP_NO_CORRESPONDING_RE.findall(text))

    views = [(int(s), float(lo), float(hi)) for s, lo, hi in _VIEW_INTENSITY_RE.findall(text)]
    m["view_intensity"] = views
    if views:
        setups = sorted(set(v[0] for v in views))
        m["n_views"] = len(setups)
        # one intensity line per view per STAGE 2 level; a view is starved
        # only when every line for it shows no detectable dynamic range
        by_setup = {}
        for s, lo, hi in views:
            by_setup.setdefault(s, []).append(hi - lo < 10.0)
        m["starved_views"] = sum(1 for flags in by_setup.values() if all(flags))

    m["downsampling_lines"] = _DOWNSAMPLING_RE.findall(text)[:6]

    params_match = _PARAMS_RE.search(text)
    if params_match:
        m["params_line"] = params_match.group(1).strip()
        for token in m["params_line"].split():
            if "=" in token:
                key, value = token.split("=", 1)
                m["params"][key] = value

    saved = _SAVED_XML_RE.findall(text)
    if saved:
        m["saved_xml"] = saved[-1]

    return m


def verify_params_effective(log_metrics: dict, intended_params: dict, tolerance: float = 1e-6) -> bool:
    """Drift guard: intended macro values against the log-recorded effective values.

    ``intended_params`` maps macro keys (link_relative, ...) to numeric or
    string values. Returns False when any logged value differs, or when the
    log carries no BIGSTITCHER_PARAMS line at all (older logs cannot prove
    their parameters took effect).
    """
    logged = log_metrics.get("params") or {}
    if not logged:
        return False
    for key, intended in intended_params.items():
        if key not in logged:
            return False
        actual = logged[key]
        try:
            if math.isclose(float(actual), float(intended), rel_tol=0, abs_tol=tolerance):
                continue
            return False
        except (TypeError, ValueError):
            if str(actual) != str(intended):
                return False
    return True


_STITCHING_TRANSFORM_NAME = "Stitching Transform"


def _parse_affine_translation(affine_text: str):
    """Translation column of a 3x4 affine row-major string."""
    vals = [float(x) for x in affine_text.split()]
    if len(vals) < 12:
        raise ValueError(f"affine needs 12 values, got {len(vals)}: {affine_text!r}")
    return vals[3], vals[7], vals[11]


def _registration_transforms(view_registration) -> list:
    """(name | None, (tx, ty, tz)) per affine ViewTransform of one registration."""
    out = []
    for vt in view_registration.findall("ViewTransform"):
        aff = vt.find("affine")
        if aff is None or aff.text is None:
            continue
        name_el = vt.find("Name")
        name = name_el.text if name_el is not None else None
        out.append((name, _parse_affine_translation(aff.text)))
    return out


def _fit_scale_translation(initials, finals):
    """Least-squares fit ``final = s * initial + t`` for one axis.

    Returns (s, t, residuals) or None when the initial values have no spread
    (e.g. a single-row grid has all-zero x offsets - the x scale is undefined).
    """
    n = len(initials)
    if n < 2:
        return None
    mean_i = sum(initials) / n
    mean_f = sum(finals) / n
    var_i = sum((i - mean_i) ** 2 for i in initials)
    if var_i <= 1e-9:
        return None
    cov_if = sum((i - mean_i) * (f - mean_f) for i, f in zip(initials, finals))
    s = cov_if / var_i
    t = mean_f - s * mean_i
    residuals = [f - (s * i + t) for i, f in zip(initials, finals)]
    return s, t, residuals


def _residual_stats(residuals) -> dict:
    if not residuals:
        return {}
    abs_r = [abs(r) for r in residuals]
    mean = sum(abs_r) / len(abs_r)
    var = sum((r - mean) ** 2 for r in abs_r) / len(abs_r)
    return {
        "n": len(residuals),
        "residual_mean_px": mean,
        "residual_std_px": math.sqrt(var),
        "residual_max_px": max(abs_r),
    }


def score_from_xml(stitched_xml_path) -> dict:
    """Scaled-grid fit of the final tile positions in a stitched BigStitcher XML.

    The final position of each registered view is the sum of its affine
    translations (BigStitcher composes transforms in document order); the
    nominal grid is the sum of the non-``Stitching Transform`` translations
    (the metadata grid offsets). Per axis with at least two distinct nominal
    offsets, ``final = s * nominal + t`` is fit by least squares: the fitted
    scale ratio ``s`` estimates the zoom/ETL pixel-size miscalibration and the
    residual scatter around the fit is the tile-grid misalignment signal.
    """
    stitched_xml_path = Path(stitched_xml_path)
    root = ET.parse(stitched_xml_path).getroot()

    registrations = root.find("ViewRegistrations")
    if registrations is None:
        raise ValueError(f"no ViewRegistrations in {stitched_xml_path}")

    views = []
    for reg in registrations.findall("ViewRegistration"):
        transforms = _registration_transforms(reg)
        if not transforms:
            continue
        final = [0.0, 0.0, 0.0]
        nominal = [0.0, 0.0, 0.0]
        for name, (tx, ty, tz) in transforms:
            t = (tx, ty, tz)
            final = [f + v for f, v in zip(final, t)]
            if name != _STITCHING_TRANSFORM_NAME:
                nominal = [n + v for n, v in zip(nominal, t)]
        views.append({
            "setup": reg.get("setup"),
            "nominal": tuple(nominal),
            "final": tuple(final),
            "stitching_correction": next((t for name, t in transforms if name == _STITCHING_TRANSFORM_NAME), None),
            "n_transforms": len(transforms),
        })

    if not views:
        raise ValueError(f"no ViewRegistrations with affine transforms in {stitched_xml_path}")

    out = {
        "xml": str(stitched_xml_path),
        "n_views": len(views),
        "n_stitching_transforms": sum(1 for v in views if v["stitching_correction"] is not None),
        "axes": {},
    }
    for axis in range(3):
        initials = [v["nominal"][axis] for v in views]
        finals = [v["final"][axis] for v in views]
        fit = _fit_scale_translation(initials, finals)
        if fit is None:
            out["axes"]["xyz"[axis]] = {"fit": "undefined (no spread in nominal offsets)"}
            continue
        s, t, residuals = fit
        stats = _residual_stats(residuals)
        out["axes"]["xyz"[axis]] = {
            "fitted_scale": s,
            "fitted_translation_px": t,
            "scale_offset_pct": (s - 1.0) * 100.0,
            **stats,
        }
    out["views"] = views
    return out


def dominant_scale_offset(xml_metrics: dict):
    offsets = [a.get("scale_offset_pct") for a in xml_metrics["axes"].values()
               if isinstance(a, dict) and a.get("scale_offset_pct") is not None]
    if not offsets:
        return None
    return max(offsets, key=abs)


def dominant_residual_px(xml_metrics: dict):
    scatters = [a.get("residual_mean_px") for a in xml_metrics["axes"].values()
                if isinstance(a, dict) and a.get("residual_mean_px") is not None]
    if not scatters:
        return None
    return max(scatters)


def alignment_quality_score(metrics: dict, xml_metrics: dict = None) -> dict:
    """Weighted composite of residual + connectivity + signal + ICP health.

    Returns ``(score, components)`` where ``score`` is 0-100. Missing inputs
    are neutral (weight 1.0) so older logs without a metric are not punished,
    except dropped links and ICP failures which default to zero events.

    - residual 0.4: exp(-residual / 25) on the XML fit scatter when available,
      else the log's final Avg Error.
    - connectivity 0.3: min(1, added_links / (n_views - 1)) when link counts
      exist, times exp(-dropped_links / 10) - dropping links is scored.
    - signal 0.2: exp(-zero_detection_views) times the fraction of non-starved
      views (from per-view intensity ranges when present).
    - icp 0.1: exp(-(not_enough + no_corresponding) / 20).
    """
    components = {}

    residual = None
    if xml_metrics is not None:
        residual = dominant_residual_px(xml_metrics)
        components["residual_source"] = "xml_grid_fit_scatter"
    if residual is None and metrics.get("final_avg_error") is not None:
        residual = metrics["final_avg_error"]
        components["residual_source"] = "log_final_avg_error"
    components["residual_px"] = residual
    residual_c = math.exp(-residual / 25.0) if residual is not None else 1.0
    components["residual_component"] = residual_c

    added = metrics.get("added_strong_links") or 0
    dropped = metrics.get("dropped_links") or 0
    n_views = metrics.get("n_views")
    dropped_c = math.exp(-dropped / 10.0)
    if added and n_views:
        retained = min(1.0, added / max(1, n_views - 1))
        components["links_retained_ratio"] = retained
        connectivity_c = retained * dropped_c
    else:
        connectivity_c = dropped_c
    components["dropped_links"] = dropped
    components["added_strong_links"] = added
    components["connectivity_component"] = connectivity_c

    zero = metrics.get("zero_detection_views") or 0
    signal_c = math.exp(-float(zero))
    starved = metrics.get("starved_views")
    n_intensity_views = len(metrics.get("view_intensity") or [])
    if n_intensity_views:
        signal_c *= 1.0 - (starved / n_intensity_views)
        components["starved_views"] = starved
    components["zero_detection_views"] = zero
    components["signal_component"] = signal_c

    icp_fails = (metrics.get("icp_not_enough") or 0) + (metrics.get("icp_no_corresponding") or 0)
    icp_c = math.exp(-icp_fails / 20.0)
    components["icp_failures"] = icp_fails
    components["icp_component"] = icp_c

    score = 100.0 * (0.4 * residual_c + 0.3 * connectivity_c + 0.2 * signal_c + 0.1 * icp_c)
    return score, components


def recommend_params(metrics: dict, xml_metrics: dict = None) -> list:
    """Deterministic recommendations from the roadmap's tune-vs-fix table."""
    recs = []
    first = metrics.get("first_round_avg_error")
    final = metrics.get("final_avg_error")
    dropped = metrics.get("dropped_links") or 0
    added = metrics.get("added_strong_links") or 0
    zero = metrics.get("zero_detection_views") or 0
    starved = metrics.get("starved_views") or 0

    if first is not None and first > 50 and dropped > 5:
        recs.append(
            f"first-round Avg {first:.1f}px with {dropped} dropped links: the link thresholds "
            f"({metrics['params'].get('link_relative', '?')}/{metrics['params'].get('link_absolute', '?')}px) "
            "or the initial grid are fighting a miscalibration - try the RELAXED preset "
            "(relative 7.0 / absolute 5.0) or fix the grid/pixel-size calibration"
        )
    if zero > 0:
        recs.append(
            f"{zero} zero-detection view(s): per-image relative DoG thresholds silence dark "
            "tiles - set explicit DoG min/max intensity or exclude those views from STAGE 2"
        )
    if starved > 0:
        recs.append(f"{starved} view(s) with near-zero dynamic range in the intensity stats - "
                    "check illumination before trusting STAGE 2 on them")
    if xml_metrics is not None:
        s = dominant_scale_offset(xml_metrics)
        if s is not None and abs(s) > 2.0:
            recs.append(
                f"fitted grid scale offset {s:+.2f}%: zoom/ETL pixel-size miscalibration - "
                "cannot be tuned away; recalibrate per (zoom, ETL CFG) or apply the "
                "self-calibration guard to the stitched output voxel size"
            )
        scatter = dominant_residual_px(xml_metrics)
        if scatter is not None and scatter > 3.0 and (first is None or first < 50):
            recs.append(
                f"tile-grid residual scatter {scatter:.2f}px with small first-round error: "
                "phase-correlation precision limit - lower STAGE 1 downsampling so overlap "
                "strips stay >= ~150-300px"
            )
    if added and metrics.get("n_views") and added < metrics["n_views"] - 1:
        recs.append(
            f"only {added} links for {metrics['n_views']} views: the view graph may be "
            "disconnected - lower the link thresholds or add link-neighbors"
        )
    icp_fails = (metrics.get("icp_not_enough") or 0) + (metrics.get("icp_no_corresponding") or 0)
    if icp_fails > 5:
        recs.append(
            f"{icp_fails} ICP channel-refinement failure(s): STAGE 2 found no corresponding "
            "points on those views - check STAGE 2 scoping, the ICP maximum error, or the "
            "sample's per-channel structure"
        )
    if final is not None and final > 5.0:
        recs.append(
            f"final Avg Error {final:.2f}px is above the ~3px residual ceiling: alignment "
            "did not fully converge - lower STAGE 1 downsampling or revisit the link thresholds"
        )
    if not recs:
        recs.append("healthy signature: no deterministic action recommended")
    return recs


def find_newest_align_log(dataset_dir) -> Path:
    """Newest ``*_align_*bigstitcher.log`` under a dataset's logs_mesospim_utils/."""
    dataset_dir = Path(dataset_dir)
    log_dir = dataset_dir / "logs_mesospim_utils"
    candidates = sorted(
        (p for p in log_dir.glob(LOG_SUFFIX_GLOB) if p.is_file()),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(f"no {LOG_SUFFIX_GLOB} under {log_dir}")
    return candidates[-1]


def grade_run(log_path) -> dict:
    """Score one log (+ the stitched XML it references). Nothing is executed."""
    metrics = parse_align_log(log_path)
    result = {"metrics": metrics, "xml": None, "score": None, "components": None, "recommendations": []}

    xml_path = metrics.get("saved_xml")
    if xml_path and Path(xml_path).exists():
        try:
            result["xml"] = score_from_xml(xml_path)
        except (ET.ParseError, ValueError) as exc:
            result["xml_error"] = f"{type(exc).__name__}: {exc}"
    elif xml_path:
        result["xml_error"] = f"saved xml not found: {xml_path}"

    score, components = alignment_quality_score(metrics, result["xml"])
    result["score"] = score
    result["components"] = components
    result["recommendations"] = recommend_params(metrics, result["xml"])
    return result


def format_report(graded: dict) -> str:
    metrics = graded["metrics"]
    xml = graded["xml"]
    lines = []
    dataset = Path(metrics["log"]).parent.parent.name
    lines.append(f"=== {dataset} ===")
    lines.append(f"log: {metrics['log']}")
    lines.append(
        f"bigstitcher {metrics['bigstitcher_version']} | mvrecon {metrics['mvrecon_version']} "
        f"| opt_rounds {metrics['opt_rounds']} | success {metrics['success_marker'] or metrics['align_success_marker']}"
    )
    if metrics["params_line"]:
        lines.append(f"params: {metrics['params_line']}")
    err = lambda v: "n/a" if v is None else f"{v:.2f}px"
    lines.append(
        f"first-round Avg/Max: {err(metrics['first_round_avg_error'])} / {err(metrics['first_round_max_error'])}  "
        f"final Avg/Max: {err(metrics['final_avg_error'])} / {err(metrics['final_max_error'])}"
    )
    lines.append(
        f"links: {metrics['added_strong_links']} added, {metrics['dropped_links']} dropped "
        f"(max dropped error {max(metrics['dropped_link_errors'], default=0):.1f}px)"
    )
    peaks = f"{metrics['dog_peaks_min']}-{metrics['dog_peaks_max']}" if metrics["dog_peaks_all"] else "n/a"
    lines.append(
        f"DoG peaks/view: {peaks} | zero-detection views: {metrics['zero_detection_views']} | "
        f"starved views: {metrics['starved_views']}/{metrics['n_views']}"
    )
    lines.append(
        f"ICP failures: not_enough {metrics['icp_not_enough']}, no_corresponding {metrics['icp_no_corresponding']}"
    )
    if metrics["downsampling_lines"]:
        lines.append(f"downsampling: {metrics['downsampling_lines'][0].strip()}")

    if xml is not None:
        lines.append(f"xml grid fit ({xml['xml']}):")
        for axis, a in xml["axes"].items():
            if "fitted_scale" not in a:
                lines.append(f"  {axis}: {a['fit']}")
                continue
            lines.append(
                f"  {axis}: scale {a['fitted_scale']:.4f} ({a['scale_offset_pct']:+.2f}%)  "
                f"residual mean/std/max {a['residual_mean_px']:.2f}/{a['residual_std_px']:.2f}/{a['residual_max_px']:.2f}px"
            )
    elif graded.get("xml_error"):
        lines.append(f"xml grid fit: unavailable ({graded['xml_error']})")

    comp = graded["components"]
    lines.append(f"SCORE: {graded['score']:.1f}/100")
    lines.append(
        f"  components (0-1): residual {comp['residual_component']:.2f} "
        f"({comp['residual_source']}) | connectivity {comp['connectivity_component']:.2f} | "
        f"signal {comp['signal_component']:.2f} | icp {comp['icp_component']:.2f}"
    )
    for rec in graded["recommendations"]:
        lines.append(f"  -> {rec}")
    return "\n".join(lines)


def report(paths, json_out: bool = False) -> None:
    """Grade past runs: dataset dirs and/or direct log paths. Prints only."""
    results = []
    for path in paths:
        path = Path(path)
        if path.is_dir():
            log_path = find_newest_align_log(path)
        else:
            log_path = path
        results.append(grade_run(log_path))

    if json_out:
        printable = []
        for r in results:
            entry = {"metrics": r["metrics"], "score": r["score"], "components": r["components"],
                     "recommendations": r["recommendations"], "xml_error": r.get("xml_error")}
            if r["xml"] is not None:
                entry["xml"] = {k: v for k, v in r["xml"].items() if k != "views"}
                entry["xml_views"] = r["xml"]["views"]
            printable.append(entry)
        print(json.dumps(printable, indent=2))
    else:
        for r in results:
            print(format_report(r))
            print()


if __name__ == "__main__":
    import typer

    app = typer.Typer(help="Report-only BigStitcher alignment scoring (grades past runs, executes nothing)")

    @app.command()
    def report_cli(
        paths: list[Path] = typer.Argument(..., help="Dataset dirs (newest *_align_*bigstitcher.log) or direct log paths"),
        json_out: bool = typer.Option(False, "--json", help="Machine-readable JSON output"),
    ):
        """Grade past BigStitcher alignment runs from their SLURM logs + stitched XML."""
        report(paths, json_out=json_out)

    app()
