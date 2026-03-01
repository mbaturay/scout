"""Football AI Pipeline — Streamlit UI.

Launch:
    cd football_ai_pipeline
    streamlit run ui/app.py

Or from repo root:
    streamlit run football_ai_pipeline/ui/app.py
"""

from __future__ import annotations

import html
import json
import os
import re
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

# Package root: the football_ai_pipeline/ directory (contains src/, configs/, etc.)
_PKG_DIR = Path(__file__).resolve().parent.parent

# Threading env vars that prevent torch/OpenBLAS hangs on Windows
_SAFE_THREAD_VARS: dict[str, str] = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}

# Key output files to discover after a run
_OUTPUT_FILES: list[tuple[str, str]] = [
    ("annotated.mp4", "Annotated video"),
    ("run_report.json", "Run report"),
    ("team_stats.json", "Team stats (JSON)"),
    ("player_stats.json", "Player stats (JSON)"),
    ("events.json", "Match events"),
    ("teams_summary.csv", "Team summary (CSV)"),
    ("players_summary.csv", "Player summary (CSV)"),
    ("stats/team_stats.csv", "Analytics team stats (CSV)"),
    ("stats/player_stats.csv", "Analytics player stats (CSV)"),
    ("frames.jsonl", "Frame-level data"),
    ("metadata.json", "Video metadata"),
]

# Contract artifact paths — ordered by preference (JSON first, CSV fallbacks)
_ARTIFACT_CONTRACT: dict[str, list[str]] = {
    "team_stats": [
        "team_stats.json",
        "team_summary.json",
        "stats/team_stats.csv",
        "teams_summary.csv",
        "rolling_summary.json",
    ],
    "player_stats": [
        "player_stats.json",
        "player_summary.json",
        "stats/player_stats.csv",
        "players_summary.csv",
    ],
    "events": ["events.json"],
    "run_report": ["run_report.json", "metadata.json"],
}


# ---------------------------------------------------------------------------
# Path resolution helper
# ---------------------------------------------------------------------------

def resolve_path(raw: str | None, base_dir: Path) -> Path | None:
    """Resolve a user-supplied path string against a base directory.

    Rules:
      - None / empty string -> None
      - Absolute path -> used as-is (resolved)
      - Relative path -> base_dir / relative (resolved)
      - If the relative path starts with the base_dir's folder name
        (e.g. "football_ai_pipeline/configs/..."), strip the leading
        duplicate segment so we don't get base/base/...
    """
    if not raw or not raw.strip():
        return None
    p = Path(raw.strip())
    if p.is_absolute():
        return p.resolve()
    # Guard against double-segment: if raw starts with the base_dir's
    # own folder name (e.g. "football_ai_pipeline/configs/..."), strip it.
    base_name = base_dir.name
    parts = p.parts
    if parts and parts[0] == base_name:
        p = Path(*parts[1:]) if len(parts) > 1 else Path(".")
    return (base_dir / p).resolve()


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------

def _build_env() -> dict[str, str]:
    """Build subprocess environment with safe threading defaults."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_PKG_DIR.parent) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    for key, val in _SAFE_THREAD_VARS.items():
        if key not in env:
            env[key] = val
    return env


def _run_pipeline_subprocess(
    video_path: str,
    output_dir: str,
    config_path: str,
    stride: int,
    max_frames: int | None,
    save_video: bool,
    log_lines: list[str],
    progress: dict[str, int],
    proc_holder: dict[str, Any],
) -> int:
    """Run the pipeline CLI as a subprocess, streaming stdout into *log_lines*."""
    cmd = [
        sys.executable, "-m", "football_ai_pipeline",
        "--input", video_path,
        "--output", output_dir,
        "--config", config_path,
        "--stride", str(stride),
        "--save-video", str(save_video).lower(),
    ]
    if max_frames is not None and max_frames > 0:
        cmd += ["--max-frames", str(max_frames)]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=str(_PKG_DIR),
        env=_build_env(),
    )
    proc_holder["proc"] = proc

    assert proc.stdout is not None
    for line in iter(proc.stdout.readline, ""):
        line = line.rstrip("\n")
        log_lines.append(line)
        progress["last_line"] = line

        # Parse tqdm progress: "Processing frames:  42%|..."
        m = re.search(r"(\d+)%\|", line)
        if m:
            progress["pct"] = int(m.group(1))

        # Parse "N/M" pattern from tqdm
        m2 = re.search(r"\s(\d+)/(\d+)", line)
        if m2:
            done, total = int(m2.group(1)), int(m2.group(2))
            if total > 0:
                progress["frames_done"] = done
                progress["frames_total"] = total
                progress["pct"] = min(int(done / total * 100), 100)

        # Detect pipeline stages from log messages
        line_lower = line.lower()
        if "initializing" in line_lower:
            progress["current_stage"] = "Initializing"
        elif "processing frames" in line_lower or "processing frame" in line_lower:
            progress["current_stage"] = "Processing frames"
        elif "team classifier fitted" in line_lower:
            progress["current_stage"] = "Team classification"
        elif "analytics" in line_lower and "finalize" in line_lower:
            progress["current_stage"] = "Computing analytics"
        elif "pipeline finished" in line_lower:
            progress["current_stage"] = "Complete"
        elif "writing" in line_lower or "export" in line_lower:
            progress["current_stage"] = "Exporting"

        # Count warnings / errors
        if "[WARNING]" in line or "WARNING:" in line:
            progress["warn_count"] = progress.get("warn_count", 0) + 1
        if "[ERROR]" in line or "ERROR:" in line:
            progress["error_count"] = progress.get("error_count", 0) + 1

    proc.wait()
    proc_holder["proc"] = None
    progress["pct"] = 100 if proc.returncode == 0 else progress.get("pct", 0)
    if proc.returncode == 0:
        progress["current_stage"] = "Complete"
    return proc.returncode


def _discover_outputs(output_dir: Path) -> dict[str, str]:
    """Scan output_dir for known output files. Returns {name: abs_path}."""
    found: dict[str, str] = {}
    for filename, _label in _OUTPUT_FILES:
        p = output_dir / filename
        if p.exists() and p.stat().st_size > 0:
            found[filename] = str(p.resolve())
    stats_dir = output_dir / "stats"
    if stats_dir.exists() and any(stats_dir.iterdir()):
        found["stats/"] = str(stats_dir.resolve())
    heatmap_dir = output_dir / "heatmaps"
    if heatmap_dir.exists() and any(heatmap_dir.glob("*.png")):
        found["heatmaps/"] = str(heatmap_dir.resolve())
    return found


# ---------------------------------------------------------------------------
# Terminal log rendering
# ---------------------------------------------------------------------------

def _render_console_html(
    lines: list[str],
    status: str = "running",
    max_lines: int = 500,
) -> str:
    """Render docked console with log lines.

    *status*: 'running' | 'done' | 'error' — controls the header badge.
    """
    tail = lines[-max_lines:] if len(lines) > max_lines else lines

    if not tail:
        body = '<span class="log-dim">Waiting for output...</span>'
    else:
        html_lines: list[str] = []
        for raw in tail:
            escaped = html.escape(raw)
            if "[WARNING]" in raw or "WARNING:" in raw:
                html_lines.append(f'<span class="log-warn">{escaped}</span>')
            elif "[ERROR]" in raw or "ERROR:" in raw or "Traceback" in raw:
                html_lines.append(f'<span class="log-error">{escaped}</span>')
            elif "[INFO]" in raw:
                html_lines.append(f'<span class="log-info">{escaped}</span>')
            else:
                html_lines.append(escaped)
        body = "\n".join(html_lines)

    badge_cls = f"console-badge-{status}"
    badge_text = status.upper()
    n_lines = len(lines)
    return (
        '<div class="console-dock">'
        '<div class="console-header">'
        f'<span>Console Output ({n_lines} lines)</span>'
        f'<span class="console-badge {badge_cls}">{badge_text}</span>'
        '</div>'
        f'<div class="terminal-panel" id="term-log">{body}</div>'
        '</div>'
        '<script>var e=document.getElementById("term-log");'
        'if(e)e.scrollTop=e.scrollHeight;</script>'
    )


def _read_json(path: Path) -> Any:
    """Read a JSON file, returning None on any error."""
    try:
        if path.exists() and path.stat().st_size > 0:
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return None


def _normalize_team_df(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure team DataFrame has consistent column types."""
    if df.empty:
        return df
    if "team_id" in df.columns:
        df["team_id"] = pd.to_numeric(df["team_id"], errors="coerce").fillna(0).astype(int)
    for col in ("possession_pct", "xG_total", "territory_avg_x", "press_intensity",
                "pass_completion_pct"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    for col in ("pass_count", "pass_completed", "shots_count", "interceptions",
                "tackles", "touches"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    return df


def _normalize_player_df(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure player DataFrame has consistent column types."""
    if df.empty:
        return df
    if "track_id" in df.columns:
        df["track_id"] = pd.to_numeric(df["track_id"], errors="coerce").fillna(0).astype(int)
    if "team_id" in df.columns:
        df["team_id"] = pd.to_numeric(df["team_id"], errors="coerce").fillna(0).astype(int)
    for col in ("distance_covered_m", "avg_speed_mps", "top_speed_mps",
                "distance_m", "top_speed_mps"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    for col in ("touches", "receptions", "passes", "interceptions", "tackles",
                "shots", "sprint_count"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    return df


def _load_df_from_json(path: Path, wrapper_keys: tuple[str, ...]) -> pd.DataFrame | None:
    """Load a DataFrame from a JSON file, trying wrapper keys then flat list."""
    data = _read_json(path)
    if data is None:
        return None
    for k in wrapper_keys:
        if isinstance(data, dict) and isinstance(data.get(k), list) and data[k]:
            return pd.DataFrame(data[k])
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return pd.DataFrame(data)
    return None


def _load_df_from_csv(path: Path) -> pd.DataFrame | None:
    """Load a DataFrame from CSV, returning None on error or empty."""
    try:
        if path.exists() and path.stat().st_size > 0:
            df = pd.read_csv(path)
            if not df.empty:
                return df
    except Exception:
        pass
    return None


def _load_output_artifacts(output_dir: Path) -> dict[str, Any]:
    """Load all output artifacts from a pipeline run.

    Returns a dict with keys:
      team_df, player_df, events, team_summary, player_summary,
      rolling_summary, metadata, run_report, loaded (list of loaded files),
      missing (list of missing files).
    """
    result: dict[str, Any] = {
        "team_df": None,
        "player_df": None,
        "events": None,
        "team_summary": None,
        "player_summary": None,
        "rolling_summary": None,
        "metadata": None,
        "run_report": None,
        "loaded": [],
        "missing": [],
    }

    # -- Team DataFrame: prefer team_stats.json, fallback to CSVs --
    team_df = None
    for rel, loader in [
        ("team_stats.json", lambda p: _load_df_from_json(p, ("teams", "team_stats"))),
        ("stats/team_stats.csv", _load_df_from_csv),
        ("teams_summary.csv", _load_df_from_csv),
    ]:
        p = output_dir / rel
        if p.exists() and p.stat().st_size > 0:
            team_df = loader(p)
            if team_df is not None:
                team_df = _normalize_team_df(team_df)
                result["loaded"].append(rel)
                break
    if team_df is not None:
        result["team_df"] = team_df
    else:
        result["missing"].append("team_stats")

    # -- Player DataFrame: prefer player_stats.json, fallback to CSVs --
    player_df = None
    for rel, loader in [
        ("player_stats.json", lambda p: _load_df_from_json(p, ("players", "player_stats"))),
        ("stats/player_stats.csv", _load_df_from_csv),
        ("players_summary.csv", _load_df_from_csv),
    ]:
        p = output_dir / rel
        if p.exists() and p.stat().st_size > 0:
            player_df = loader(p)
            if player_df is not None:
                player_df = _normalize_player_df(player_df)
                result["loaded"].append(rel)
                break
    if player_df is not None:
        result["player_df"] = player_df
    else:
        result["missing"].append("player_stats")

    # -- Events --
    ep = output_dir / "events.json"
    ev_raw = _read_json(ep)
    if ev_raw is not None:
        if isinstance(ev_raw, dict) and "events" in ev_raw:
            result["events"] = ev_raw["events"]
        elif isinstance(ev_raw, list):
            result["events"] = ev_raw
        if result["events"] is not None:
            result["loaded"].append("events.json")
    if result["events"] is None:
        result["missing"].append("events")

    # -- Team summary (stats/team_summary.json — nested per-module dict) --
    ts_path = output_dir / "stats" / "team_summary.json"
    ts_data = _read_json(ts_path)
    if ts_data is not None:
        result["team_summary"] = ts_data
        result["loaded"].append("stats/team_summary.json")

    # -- Player summary (stats/player_summary.json — track_id-keyed dict) --
    ps_path = output_dir / "stats" / "player_summary.json"
    ps_data = _read_json(ps_path)
    if ps_data is not None:
        result["player_summary"] = ps_data
        result["loaded"].append("stats/player_summary.json")

    # -- Rolling summary --
    rs_path = output_dir / "stats" / "rolling_summary.json"
    rs_data = _read_json(rs_path)
    if rs_data is not None:
        result["rolling_summary"] = rs_data
        result["loaded"].append("stats/rolling_summary.json")

    # -- Metadata --
    md_path = output_dir / "metadata.json"
    md_data = _read_json(md_path)
    if md_data is not None:
        result["metadata"] = md_data
        result["loaded"].append("metadata.json")

    # -- Run report --
    for rel in ("run_report.json", "metadata.json"):
        rp = output_dir / rel
        rdata = _read_json(rp)
        if rdata is not None and "coverage" in rdata:
            result["run_report"] = rdata
            if rel not in result["loaded"]:
                result["loaded"].append(rel)
            break

    return result


def _load_match_stats(output_dir: Path) -> dict[str, Any]:
    """Compatibility wrapper — returns the old-style dict expected by _render_match_analytics.

    Internally delegates to _load_output_artifacts and maps the result.
    """
    artifacts = _load_output_artifacts(output_dir)
    return {
        "team_stats": artifacts["team_df"],
        "player_stats": artifacts["player_df"],
        "events": artifacts["events"],
        "run_report": artifacts["run_report"],
        "team_summary": artifacts["team_summary"],
        "player_summary": artifacts["player_summary"],
        "rolling_summary": artifacts["rolling_summary"],
        "metadata": artifacts["metadata"],
        "loaded": artifacts["loaded"],
        "missing": artifacts["missing"],
        "mtimes": {},   # not needed anymore but kept for compat
    }


# Column display names for player stats table
_PLAYER_COL_RENAME: dict[str, str] = {
    "track_id": "Player",
    "team_id": "Team",
    "distance_covered_m": "Distance (m)",
    "avg_speed_mps": "Avg Speed (m/s)",
    "top_speed_mps": "Top Speed (m/s)",
    "touches": "Touches",
    "receptions": "Receptions",
    "passes": "Passes",
    "interceptions": "Interceptions",
    "tackles": "Tackles",
    "shots": "Shots",
    "pass_count": "Passes",
    "shots_count": "Shots",
}


def _render_match_analytics(
    stats: dict[str, Any],
    key_prefix: str = "analytics",
    read_only: bool = False,
) -> None:
    """Render match analytics section from loaded stats.

    When *read_only* is True, interactive widgets (selectbox, filters) are
    skipped — safe for use inside polling loops where Streamlit would
    otherwise register duplicate element keys.
    """
    team_df = stats.get("team_stats") if "team_stats" in stats else stats.get("team_df")
    player_df = stats.get("player_stats") if "player_stats" in stats else stats.get("player_df")
    events = stats.get("events")
    missing = stats.get("missing", [])
    loaded = stats.get("loaded", [])
    team_summary = stats.get("team_summary")  # nested per-module dict from stats/team_summary.json

    has_data = team_df is not None or player_df is not None or events

    # -- Debug indicator: which artifacts loaded --
    if loaded:
        st.caption("Artifacts loaded: " + ", ".join(f"`{f}`" for f in loaded))

    # -- Missing artifact warnings --
    if missing and not has_data:
        st.warning(
            "Missing analytics artifacts: **"
            + "**, **".join(missing)
            + "**. Run the pipeline to generate them."
        )
        return
    if not has_data:
        st.info("Match analytics will appear here once the pipeline produces output artifacts.")
        return
    if missing:
        st.warning("Missing: " + ", ".join(f"`{f}`" for f in missing))

    # -- Team stats cards --
    if team_df is not None and not team_df.empty:
        st.markdown("#### Team Overview")
        n_teams = len(team_df)
        team_cols = st.columns(max(n_teams, 1))
        for i, (_, row) in enumerate(team_df.iterrows()):
            with team_cols[i % len(team_cols)]:
                tid = row.get("team_id", i)
                st.markdown(f"**Team {tid}**")
                if "possession_pct" in row:
                    st.metric("Possession", f"{row['possession_pct']:.1f}%")
                if "xG_total" in row:
                    st.metric("xG", f"{row['xG_total']:.2f}")
                if "pass_count" in row:
                    st.metric("Passes", int(row["pass_count"]))
                if "pass_completed" in row and "pass_completion_pct" in row:
                    st.metric("Pass Accuracy", f"{row['pass_completion_pct']:.0f}%")
                if "shots_count" in row:
                    st.metric("Shots", int(row["shots_count"]))
                if "interceptions" in row:
                    st.metric("Interceptions", int(row["interceptions"]))
                if "tackles" in row:
                    st.metric("Tackles", int(row["tackles"]))
                if "touches" in row:
                    st.metric("Touches", int(row["touches"]))
                if "press_intensity" in row:
                    st.metric("Press Intensity", f"{row['press_intensity']:.2f}")
                if "territory_avg_x" in row:
                    st.metric("Territory Avg X", f"{row['territory_avg_x']:.1f}m")

        # Enrichment from team_summary (physical stats from stats/team_summary.json)
        if team_summary and isinstance(team_summary, dict):
            physical = team_summary.get("physical")
            if physical and isinstance(physical, dict):
                st.markdown("##### Physical Stats (from stats module)")
                phys_cols = st.columns(max(len(physical), 1))
                for i, (tid, pstats) in enumerate(sorted(physical.items())):
                    with phys_cols[i % len(phys_cols)]:
                        st.markdown(f"**Team {tid}**")
                        if isinstance(pstats, dict):
                            if "total_distance_m" in pstats:
                                st.metric("Total Distance", f"{pstats['total_distance_m']:.0f}m")
                            if "total_sprints" in pstats:
                                st.metric("Sprints", int(pstats["total_sprints"]))
                            if "team_top_speed_mps" in pstats:
                                st.metric("Top Speed", f"{pstats['team_top_speed_mps']:.1f} m/s")
                            if "num_players_tracked" in pstats:
                                st.metric("Players Tracked", int(pstats["num_players_tracked"]))

    # -- Event summary --
    if events:
        st.markdown("#### Events")
        counts: dict[str, int] = {}
        for e in events:
            counts[e.get("event_type", "unknown")] = counts.get(e.get("event_type", "unknown"), 0) + 1
        evt_cols = st.columns(min(len(counts), 6)) if counts else [st]
        for i, (etype, cnt) in enumerate(sorted(counts.items())):
            evt_cols[i % len(evt_cols)].metric(etype.replace("_", " ").title(), cnt)

    # -- Player stats table --
    if player_df is not None and not player_df.empty:
        st.markdown("#### Player Stats")
        if not read_only and "team_id" in player_df.columns:
            teams = sorted(player_df["team_id"].dropna().unique())
            team_filter = st.selectbox(
                "Filter by team",
                ["All"] + [f"Team {t}" for t in teams],
                key=f"{key_prefix}_team_filter",
            )
            if team_filter != "All":
                tid = int(team_filter.split()[-1])
                player_df = player_df[player_df["team_id"] == tid]
        # Rename columns for readability, drop heatmap_grid if present
        display_df = player_df.drop(columns=["heatmap_grid"], errors="ignore")
        rename = {k: v for k, v in _PLAYER_COL_RENAME.items() if k in display_df.columns}
        display_df = display_df.rename(columns=rename)
        st.dataframe(display_df, width="stretch", height=250)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

_STATE_DEFAULTS: dict[str, Any] = {
    "run_state": "idle",        # idle | running | done | error
    "run_started_at": "",
    "run_finished_at": "",
    "run_start_ts": 0.0,        # time.time() for elapsed calculation
    "log_lines": [],
    "progress": {               # enriched progress dict
        "pct": 0,
        "frames_done": 0,
        "frames_total": 0,
        "current_stage": "",
        "warn_count": 0,
        "error_count": 0,
        "last_line": "",
    },
    "output_dir": "",
    "input_video": "",
    "resolved_config": "",
    "last_error": "",
    "outputs": {},
    "proc_holder": {"proc": None},
    "stats_cache": {},              # latest parsed match analytics
}

for _k, _v in _STATE_DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


def _reset_state() -> None:
    """Reset all run-related session state to defaults."""
    for k, v in _STATE_DEFAULTS.items():
        if k == "proc_holder":
            proc = st.session_state.proc_holder.get("proc")
            if proc and proc.poll() is None:
                proc.terminate()
            st.session_state[k] = {"proc": None}
        elif isinstance(v, (list, dict)):
            st.session_state[k] = type(v)()
            if k == "progress":
                st.session_state[k] = {"pct": 0}
        else:
            st.session_state[k] = v


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Football AI Pipeline", layout="wide")

# -- Docked console CSS + match analytics styling --
st.markdown("""
<style>
/* Docked bottom console — sticky at viewport bottom */
.console-dock {
    position: sticky;
    bottom: 0;
    z-index: 100;
    background-color: #1e1e1e;
    border-top: 2px solid #444;
    border-radius: 8px 8px 0 0;
    margin-top: 24px;
}
.console-dock .console-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 6px 16px;
    background-color: #2d2d2d;
    border-radius: 8px 8px 0 0;
    border-bottom: 1px solid #444;
}
.console-dock .console-header span {
    color: #aaa;
    font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.console-dock .console-header .console-badge {
    display: inline-block;
    padding: 1px 8px;
    border-radius: 4px;
    font-size: 10px;
    font-weight: 700;
}
.console-badge-running { background: #2a4d2a; color: #7ec87e; }
.console-badge-done { background: #1a3a5c; color: #61afef; }
.console-badge-error { background: #5c1a1a; color: #e06c75; }

.terminal-panel {
    background-color: #1e1e1e;
    color: #cccccc;
    font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
    font-size: 12px;
    line-height: 1.4;
    padding: 12px 16px;
    min-height: 180px;
    max-height: 70vh;
    height: 33vh;
    resize: vertical;
    overflow-y: auto;
    white-space: pre-wrap;
    word-wrap: break-word;
}
.terminal-panel .log-warn { color: #e5c07b; }
.terminal-panel .log-error { color: #e06c75; }
.terminal-panel .log-info { color: #61afef; }
.terminal-panel .log-dim { color: #666; }

/* Ensure content above console has breathing room */
.main-content-area { padding-bottom: 16px; }
</style>
""", unsafe_allow_html=True)

st.title("Football AI Pipeline")
st.caption("Local-first football analytics — from broadcast video to advanced stats")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

# Track whether the run button was clicked this cycle
_run_requested = False

with st.sidebar:
    st.header("Pipeline Settings")

    # -- Quick Start button --
    if st.session_state.run_state == "idle":
        if st.button("Quick Start (test video)", use_container_width=True):
            st.session_state["_qs_config"] = "configs/default.yaml"
            st.session_state["_qs_video"] = "test-video-1.mp4"
            st.session_state["_qs_output"] = "out"
            st.session_state["_qs_stride"] = 2
            st.session_state["_qs_maxframes"] = 200
            st.session_state["_qs_savevideo"] = True
            st.rerun()

    # Read quick-start overrides if set
    _qs_config = st.session_state.pop("_qs_config", None)
    _qs_video = st.session_state.pop("_qs_video", None)
    _qs_output = st.session_state.pop("_qs_output", None)
    _qs_stride = st.session_state.pop("_qs_stride", None)
    _qs_maxframes = st.session_state.pop("_qs_maxframes", None)
    _qs_savevideo = st.session_state.pop("_qs_savevideo", None)

    st.divider()

    # -- Config --
    config_path_input: str = st.text_input(
        "Config file",
        value=_qs_config or "configs/default.yaml",
        key="cfg_config_path",
        help="Relative to football_ai_pipeline/, or absolute.",
    )

    # -- Video input --
    st.subheader("Input video")
    input_method: str = st.radio(
        "Source",
        ["File path", "Upload"],
        horizontal=True,
        key="cfg_input_method",
        label_visibility="collapsed",
    )

    video_file_path: str = ""
    uploaded_file = None

    if input_method == "File path":
        video_file_path = st.text_input(
            "Video file path",
            value=_qs_video or "test-video-1.mp4",
            key="cfg_video_path",
            help="Relative to football_ai_pipeline/, or absolute.",
        )
    else:
        uploaded_file = st.file_uploader(
            "Upload a video file",
            type=["mp4", "avi", "mkv", "mov"],
            key="cfg_video_upload",
        )

    # -- Output --
    output_dir_input: str = st.text_input(
        "Output directory",
        value=_qs_output or "out",
        key="cfg_output_dir",
        help="Relative to football_ai_pipeline/, or absolute.",
    )

    # -- Options --
    stride: int = st.number_input(
        "Stride (every Nth frame)",
        min_value=1, max_value=30,
        value=_qs_stride or 2,
        key="cfg_stride",
    )

    max_frames_input: int = st.number_input(
        "Max frames (0 = all)",
        min_value=0, max_value=100_000,
        value=_qs_maxframes or 0,
        key="cfg_max_frames",
        help="Limit total frames. 0 = process all.",
    )
    max_frames: int | None = max_frames_input if max_frames_input > 0 else None

    save_video: bool = st.checkbox(
        "Save annotated video",
        value=_qs_savevideo if _qs_savevideo is not None else True,
        key="cfg_save_video",
    )

    st.divider()

    # -- Analytics options --
    st.subheader("Analytics")
    attack_dir: str = st.radio(
        "Attacking direction",
        ["Left to Right", "Right to Left"],
        horizontal=True,
        key="cfg_attack_dir",
    )
    attack_left_to_right = attack_dir == "Left to Right"

    st.divider()

    # -- Debug expander --
    with st.expander("Debug: path resolution"):
        _dbg_resolved_cfg = resolve_path(config_path_input, _PKG_DIR)
        _dbg_resolved_vid = resolve_path(video_file_path, _PKG_DIR) if video_file_path else None
        _dbg_resolved_out = resolve_path(output_dir_input or "out", _PKG_DIR)
        st.code(
            f"cwd:              {Path.cwd()}\n"
            f"project_root:     {_PKG_DIR}\n"
            f"\n"
            f"config (raw):     {config_path_input!r}\n"
            f"config (resolved):{_dbg_resolved_cfg}\n"
            f"config exists:    {_dbg_resolved_cfg.exists() if _dbg_resolved_cfg else '—'}\n"
            f"\n"
            f"video (raw):      {video_file_path!r}\n"
            f"video (resolved): {_dbg_resolved_vid}\n"
            f"video exists:     {_dbg_resolved_vid.exists() if _dbg_resolved_vid else '—'}\n"
            f"\n"
            f"output (raw):     {output_dir_input!r}\n"
            f"output (resolved):{_dbg_resolved_out}",
            language="text",
        )

    st.divider()

    # -- Action buttons --
    if st.session_state.run_state in ("done", "error"):
        if st.button("Run again", type="primary", use_container_width=True):
            _reset_state()
            st.rerun()
    elif st.session_state.run_state == "running":
        if st.button("Stop pipeline", type="secondary", use_container_width=True):
            proc = st.session_state.proc_holder.get("proc")
            if proc and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
            st.session_state.log_lines.append("--- Pipeline stopped by user ---")
            st.session_state.run_state = "error"
            st.session_state.last_error = "Stopped by user"
            st.session_state.run_finished_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.rerun()
    else:
        # idle
        _run_requested = st.button("Run Pipeline", type="primary", use_container_width=True)


# ---------------------------------------------------------------------------
# Run handler — validate, resolve, launch
# ---------------------------------------------------------------------------

if st.session_state.run_state == "idle" and _run_requested:

    # --- Resolve config ---
    resolved_cfg = resolve_path(config_path_input, _PKG_DIR)
    if resolved_cfg is None:
        st.error(
            "Config file path is empty.\n\n"
            "Try: `configs/default.yaml`"
        )
        st.stop()
    if not resolved_cfg.exists():
        st.error(
            f"Config file not found.\n\n"
            f"**You entered:** `{config_path_input}`  \n"
            f"**Resolved to:** `{resolved_cfg}`  \n\n"
            f"Make sure the file exists. Default: `configs/default.yaml`"
        )
        st.stop()

    # --- Resolve video ---
    resolved_vid: Path | None = None

    if input_method == "Upload" and uploaded_file is not None:
        upload_dir = _PKG_DIR / ".cache_uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        saved = upload_dir / uploaded_file.name
        with open(saved, "wb") as fh:
            fh.write(uploaded_file.getbuffer())
        resolved_vid = saved.resolve()
    else:
        resolved_vid = resolve_path(video_file_path, _PKG_DIR)

    if resolved_vid is None:
        st.error(
            "Video file path is empty.\n\n"
            "Enter a path like `test-video-1.mp4` or upload a file."
        )
        st.stop()
    if not resolved_vid.exists():
        st.error(
            f"Video file not found.\n\n"
            f"**You entered:** `{video_file_path}`  \n"
            f"**Resolved to:** `{resolved_vid}`  \n\n"
            f"Check that the file exists at the path above."
        )
        st.stop()

    # --- Resolve output dir ---
    resolved_out = resolve_path(output_dir_input or "out", _PKG_DIR)
    if resolved_out is None:
        resolved_out = (_PKG_DIR / "out").resolve()
    resolved_out.mkdir(parents=True, exist_ok=True)

    # --- Show resolved paths ---
    st.info(
        f"**Config:** `{resolved_cfg}`  \n"
        f"**Video:** `{resolved_vid}`  \n"
        f"**Output:** `{resolved_out}`  \n"
        f"**Stride:** {stride} | **Max frames:** {max_frames or 'all'} | "
        f"**Save video:** {save_video}"
    )

    # --- Set state and launch ---
    st.session_state.run_state = "running"
    st.session_state.run_started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.run_start_ts = time.time()
    st.session_state.run_finished_at = ""
    st.session_state.log_lines = []
    st.session_state.progress = {
        "pct": 0, "frames_done": 0, "frames_total": 0,
        "current_stage": "Starting", "warn_count": 0,
        "error_count": 0, "last_line": "",
    }
    st.session_state.output_dir = str(resolved_out)
    st.session_state.input_video = str(resolved_vid)
    st.session_state.resolved_config = str(resolved_cfg)
    st.session_state.last_error = ""
    st.session_state.outputs = {}
    st.session_state.proc_holder = {"proc": None}

    # Capture into locals for the thread closure.
    # The worker thread must NEVER call st.* — only write to these
    # plain Python objects which the main thread reads on each rerun.
    _t_vid = str(resolved_vid)
    _t_out = str(resolved_out)
    _t_cfg = str(resolved_cfg)
    _t_log = st.session_state.log_lines
    _t_prog = st.session_state.progress
    _t_ph = st.session_state.proc_holder
    _t_stride = stride
    _t_mf = max_frames
    _t_sv = save_video

    def _worker() -> None:
        rc = _run_pipeline_subprocess(
            video_path=_t_vid,
            output_dir=_t_out,
            config_path=_t_cfg,
            stride=_t_stride,
            max_frames=_t_mf,
            save_video=_t_sv,
            log_lines=_t_log,
            progress=_t_prog,
            proc_holder=_t_ph,
        )
        # Signal completion via the shared dict — NO st.session_state here.
        _t_ph["finished_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        _t_ph["return_code"] = rc

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    st.rerun()


# ---------------------------------------------------------------------------
# Main panel — render based on run_state
# ---------------------------------------------------------------------------

run_state = st.session_state.run_state

# ---- IDLE ----
if run_state == "idle":
    st.info(
        "Configure the pipeline in the sidebar and click **Run Pipeline** to start.  \n"
        "Or use the **Quick Start** button for a 200-frame test run.\n\n"
        "The pipeline works without model weights (all stats will be zero). "
        "For real analytics, provide YOLO weights via `configs/default.yaml`."
    )

# ---- RUNNING ----
elif run_state == "running":

    # Check if the worker thread signalled completion via the shared dict.
    # This runs on every rerun (triggered by the fragment timer below).
    ph = st.session_state.proc_holder
    if "return_code" in ph:
        rc = ph["return_code"]
        st.session_state.run_finished_at = ph.get("finished_at", "")
        out_path = Path(st.session_state.output_dir)
        if rc == 0:
            st.session_state.run_state = "done"
            st.session_state.outputs = _discover_outputs(out_path)
        else:
            st.session_state.run_state = "error"
            st.session_state.last_error = f"Pipeline exited with code {rc}"
        # Cache final stats into session_state so DONE/ERROR can use them
        if out_path.exists():
            st.session_state["stats_cache"] = _load_match_stats(out_path)
        st.rerun()

    # --- Live-updating fragment: re-renders every 500ms without full rerun ---
    @st.fragment(run_every=timedelta(milliseconds=500))
    def _live_panel() -> None:
        """Render progress, analytics, and console from shared data.

        Runs as a Streamlit fragment so only this portion refreshes
        every 500ms; the rest of the page (sidebar, etc.) stays stable.
        """
        prog = st.session_state.progress      # plain dict, written by worker
        log_lines = st.session_state.log_lines  # plain list, written by worker
        pct = prog.get("pct", 0)

        # -- Progress bar --
        st.progress(min(pct, 100))

        # -- Pipeline metrics --
        elapsed = time.time() - st.session_state.run_start_ts
        mins, secs = divmod(int(elapsed), 60)
        elapsed_str = f"{mins}m {secs}s" if mins else f"{secs}s"

        f_done = prog.get("frames_done", 0)
        f_total = prog.get("frames_total", 0)
        frames_str = f"{f_done} / {f_total}" if f_total else str(f_done) if f_done else "—"

        stage = prog.get("current_stage", "") or "—"
        warns = prog.get("warn_count", 0)
        errors = prog.get("error_count", 0)

        st.subheader("Pipeline Running")
        st.caption(f"Started at {st.session_state.run_started_at}")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Elapsed", elapsed_str)
        c2.metric("Frames", frames_str)
        c3.metric("Stage", stage)
        c4.metric("Progress", f"{pct}%")

        if warns or errors:
            w1, w2 = st.columns(2)
            if warns:
                w1.metric("Warnings", warns)
            if errors:
                w2.metric("Errors", errors)

        # -- Poll output artifacts for match analytics --
        out_dir = Path(st.session_state.output_dir) if st.session_state.output_dir else None
        if out_dir and out_dir.exists():
            cached = _load_match_stats(out_dir)
            if cached.get("team_stats") is not None or cached.get("events"):
                _render_match_analytics(cached, read_only=True)

        # -- Docked console --
        st.markdown(
            _render_console_html(log_lines, status="running"),
            unsafe_allow_html=True,
        )

    _live_panel()

# ---- DONE ----
elif run_state == "done":
    out_dir = Path(st.session_state.output_dir)
    outputs = st.session_state.outputs

    # -- Success banner --
    st.success(
        f"Pipeline finished successfully\n\n"
        f"**Started:** {st.session_state.run_started_at}  \n"
        f"**Finished:** {st.session_state.run_finished_at}  \n"
        f"**Output folder:** `{out_dir}`"
    )

    # -- File list --
    st.subheader("Output files")
    if outputs:
        for name, abs_path in outputs.items():
            p = Path(abs_path)
            if p.is_file():
                size_kb = p.stat().st_size / 1024
                size_str = f"{size_kb / 1024:.1f} MB" if size_kb > 1024 else f"{size_kb:.0f} KB"
                st.markdown(f"- `{name}` — {size_str}")
            else:
                n_files = len(list(p.glob("*")))
                st.markdown(f"- `{name}` — folder ({n_files} files)")
    else:
        st.info("No output files found.")

    # -- Download buttons --
    st.subheader("Downloads")
    dl_cols = st.columns(3)
    _downloadable = [
        ("annotated.mp4", "video/mp4"),
        ("run_report.json", "application/json"),
        ("teams_summary.csv", "text/csv"),
    ]
    for idx, (fname, mime) in enumerate(_downloadable):
        fpath = out_dir / fname
        if fpath.exists() and fpath.stat().st_size > 0:
            with dl_cols[idx % 3]:
                st.download_button(
                    label=f"Download {fname}",
                    data=fpath.read_bytes(),
                    file_name=fname,
                    mime=mime,
                )

    # -- Load all artifacts once for this page --
    match_stats = st.session_state.get("stats_cache") or _load_match_stats(out_dir)
    has_analytics = (
        match_stats.get("team_stats") is not None
        or match_stats.get("team_df") is not None
        or match_stats.get("events")
    )

    if has_analytics:
        st.divider()
        st.subheader("Match Analytics")
        _render_match_analytics(match_stats, key_prefix="done")

    # -- Tabs for detailed results --
    st.divider()
    tabs = st.tabs([
        "Run Report", "Team Stats", "Player Stats", "Events",
        "Heatmaps", "Confidence", "Detailed Stats", "Video", "Logs",
    ])

    with tabs[0]:
        report = match_stats.get("run_report")
        if report is not None:

            cov = report.get("coverage", {})
            if cov:
                st.subheader("Coverage")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Frames", cov.get("total_frames_processed", 0))
                c2.metric("In-Play %", f"{cov.get('in_play_pct', 0)}%")
                c3.metric("Homography %", f"{cov.get('homography_available_pct', 0)}%")
                c4.metric("Ball Position %", f"{cov.get('ball_position_available_pct', 0)}%")

                c5, c6, c7, _ = st.columns(4)
                c5.metric("Detection %", f"{cov.get('detection_frames_pct', 0)}%")
                c6.metric("Processing Time", f"{cov.get('processing_time_sec', 0)}s")
                c7.metric("Throughput", f"{cov.get('fps_throughput', 0)} fps")

            deg = report.get("degradation", {})
            if deg:
                st.subheader("Quality")
                confidence = deg.get("overall_confidence", "unknown")
                if "good" in confidence:
                    st.success(f"Overall confidence: {confidence}")
                elif "low" in confidence:
                    st.warning(f"Overall confidence: {confidence}")
                else:
                    st.error(f"Overall confidence: {confidence}")
                for w in deg.get("warnings", []):
                    st.warning(w)

            with st.expander("Raw JSON"):
                st.json(report)
        else:
            st.info("run_report.json not found.")

    # -- Team Stats (analytics) — uses already-loaded match_stats --
    with tabs[1]:
        tab_team_df = match_stats.get("team_stats") if "team_stats" in match_stats else match_stats.get("team_df")
        if tab_team_df is not None and not tab_team_df.empty:
            st.subheader("Team Analytics")
            st.dataframe(tab_team_df, width="stretch")

            # Possession bar
            if "possession_pct" in tab_team_df.columns and "team_id" in tab_team_df.columns:
                st.subheader("Possession")
                for _, row in tab_team_df.iterrows():
                    tid = row["team_id"]
                    poss = row["possession_pct"]
                    st.progress(min(poss / 100.0, 1.0), text=f"Team {tid}: {poss:.1f}%")

            # xG display
            if "xG_total" in tab_team_df.columns:
                st.subheader("Expected Goals (xG)")
                xg_cols = st.columns(len(tab_team_df))
                for i, (_, row) in enumerate(tab_team_df.iterrows()):
                    xg_cols[i].metric(f"Team {row['team_id']}", f"{row['xG_total']:.2f}")
        else:
            st.info("No team stats found.")

    # -- Player Stats (analytics) — uses already-loaded match_stats --
    with tabs[2]:
        tab_player_df = match_stats.get("player_stats") if "player_stats" in match_stats else match_stats.get("player_df")
        if tab_player_df is not None and not tab_player_df.empty:
            st.subheader("Player Analytics")
            sortable = [c for c in tab_player_df.columns if c not in ("track_id", "team_id", "confidence")]
            if sortable:
                sort_col = st.selectbox(
                    "Sort by",
                    sortable,
                    index=0,
                    key="done_player_sort_col",
                )
                ascending = st.checkbox("Ascending", value=False, key="done_player_ascending")
                df_sorted = tab_player_df.sort_values(sort_col, ascending=ascending)
            else:
                df_sorted = tab_player_df
            # Rename for display
            display = df_sorted.drop(columns=["heatmap_grid"], errors="ignore")
            rename = {k: v for k, v in _PLAYER_COL_RENAME.items() if k in display.columns}
            display = display.rename(columns=rename)
            st.dataframe(display, width="stretch")
        else:
            st.info("No player stats found.")

    # -- Events — uses already-loaded match_stats --
    with tabs[3]:
        events_data = match_stats.get("events") or []
        if events_data:
            st.subheader(f"Match Events ({len(events_data)} total)")

            # Filter by type
            event_types = sorted(set(e.get("event_type", "unknown") for e in events_data))
            selected_types = st.multiselect(
                "Filter event types",
                event_types,
                default=event_types,
                key="done_event_type_filter",
            )
            filtered = [e for e in events_data if e.get("event_type") in selected_types]
            df_ev = pd.DataFrame(filtered)
            st.dataframe(df_ev, width="stretch")

            # Summary counts
            st.subheader("Event Summary")
            counts: dict[str, int] = {}
            for e in events_data:
                et = e.get("event_type", "unknown")
                counts[et] = counts.get(et, 0) + 1
            count_cols = st.columns(min(len(counts), 6))
            for i, (etype, cnt) in enumerate(sorted(counts.items())):
                count_cols[i % len(count_cols)].metric(etype.title(), cnt)
        else:
            st.info("No events detected. Run the pipeline to generate events.")

    # -- Heatmaps --
    with tabs[4]:
        heatmap_dir = out_dir / "heatmaps"
        if heatmap_dir.exists():
            pngs = sorted(heatmap_dir.glob("*.png"))
            if pngs:
                st.subheader("Heatmaps")
                # Show in 2-column grid
                cols = st.columns(2)
                for i, png_path in enumerate(pngs):
                    with cols[i % 2]:
                        st.image(str(png_path), caption=png_path.stem.replace("_", " ").title())
            else:
                st.info("No heatmap images found.")
        else:
            st.info("heatmaps/ folder not found. Run the pipeline to generate heatmaps.")

    # -- Confidence & Coverage --
    with tabs[5]:
        conf_report = match_stats.get("run_report")
        if conf_report is not None:
            analytics_cov = conf_report.get("coverage", {}).get("analytics", {})

            if analytics_cov and analytics_cov.get("status") != "no_data":
                st.subheader("Analytics Confidence & Coverage")

                ac1, ac2, ac3 = st.columns(3)
                ac1.metric("Ball Detected", f"{analytics_cov.get('ball_detected_pct', 0)}%")
                ac2.metric("Owner Assigned", f"{analytics_cov.get('ball_owner_pct', 0)}%")
                ac3.metric("High Confidence", f"{analytics_cov.get('high_confidence_pct', 0)}%")

                # Event counts
                evt_counts = analytics_cov.get("event_counts", {})
                if evt_counts:
                    st.subheader("Event Counts")
                    evt_cols = st.columns(min(len(evt_counts), 6))
                    for i, (k, v) in enumerate(sorted(evt_counts.items())):
                        evt_cols[i % len(evt_cols)].metric(k.title(), v)

                # Warnings
                warnings = analytics_cov.get("warnings", [])
                if warnings:
                    st.subheader("Warnings")
                    for w in warnings:
                        st.warning(w)
                else:
                    st.success("No analytics warnings.")
            else:
                st.info("No analytics coverage data available.")
        else:
            st.info("run_report.json not found.")

    # -- Detailed Stats --
    with tabs[6]:
        stats_path = out_dir / "stats"
        if stats_path.exists():
            json_files = sorted(stats_path.glob("*.json"))
            if json_files:
                for jf in json_files:
                    with st.expander(jf.stem):
                        try:
                            st.json(json.loads(jf.read_text(encoding="utf-8")))
                        except Exception as e:
                            st.error(f"Error reading {jf.name}: {e}")
            else:
                st.info("No stat files in stats/ folder.")
        else:
            st.info("stats/ folder not found.")

    # -- Video --
    with tabs[7]:
        vid = out_dir / "annotated.mp4"
        if vid.exists() and vid.stat().st_size > 0:
            st.video(str(vid))
        else:
            st.info("No annotated video. Enable 'Save annotated video' to produce one.")

    # -- Logs --
    with tabs[8]:
        st.code("\n".join(st.session_state.log_lines), language="text")

    # Bottom padding so sticky console doesn't obscure content
    st.markdown('<div style="padding-bottom: 320px;"></div>', unsafe_allow_html=True)

    # -- Docked console at bottom --
    st.markdown(
        _render_console_html(st.session_state.log_lines, status="done"),
        unsafe_allow_html=True,
    )

# ---- ERROR ----
elif run_state == "error":
    st.error(
        f"Pipeline failed\n\n"
        f"**Error:** {st.session_state.last_error}  \n"
        f"**Started:** {st.session_state.run_started_at}  \n"
        f"**Stopped:** {st.session_state.run_finished_at}"
    )

    # Bottom padding so sticky console doesn't obscure content
    st.markdown('<div style="padding-bottom: 320px;"></div>', unsafe_allow_html=True)

    # Docked console
    st.markdown(
        _render_console_html(st.session_state.log_lines, status="error"),
        unsafe_allow_html=True,
    )
