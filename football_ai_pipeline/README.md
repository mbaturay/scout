# Football AI Pipeline

A local-first, open-source football analytics engine that ingests a full match video recording (broadcast or single-camera wide angle) and outputs structured tracking data, advanced match statistics, an annotated video, and machine-readable exports.

## Features

- **Physical & Movement (A)**: Distance covered, sprint counts, top speed, team tempo
- **Spatial & Tactical Shape (B)**: Team centroids, width/length, compactness, defensive line height
- **Ball Progression & Territory (C)**: Progression rate, time in thirds, possession proxy
- **Pressure & Control (D)**: Voronoi-based pitch control, pressure index on ball carrier
- **Threat / Danger (E)**: Expected Threat (xT) from ball movement, zone occupancy analysis
- **Advanced Analytics (F)**: Possession %, pass/interception/tackle detection, xG model, press intensity, player & team heatmaps

---

## Analytics

The pipeline includes an analytics engine that computes match-level statistics from per-frame tracking data:

| Metric | Description | Requires |
|--------|-------------|----------|
| **Possession %** | Frames each team owns the ball / total in-play frames | Ball detection |
| **Passes / Receptions** | Same-team ownership transitions with ball travel > 3m | Ball + tracks |
| **Interceptions** | Cross-team transitions with in-flight gap | Ball + tracks |
| **Tackles** | Direct cross-team transitions (no gap) | Ball + tracks |
| **Shots & xG** | Ball leaves owner at speed toward goal; logistic xG model | Ball + homography |
| **Territory** | Average team centroid x-position on pitch | Homography |
| **Press intensity** | Avg defenders within 10m of ball owner | Ball + tracks |
| **Heatmaps** | 2D position histograms per team/player/ball | Tracks |

### Outputs

After a run, analytics files appear in the output directory:

```
out/
  events.json               # Detected match events (passes, shots, tackles, ...)
  stats/
    team_stats.csv           # Per-team: possession, passes, xG, territory, press
    player_stats.csv         # Per-player: distance, speed, touches, passes, shots
  heatmaps/
    team_0.png               # Team 0 position heatmap
    team_1.png               # Team 1 position heatmap
    ball.png                 # Ball position heatmap
    player_<id>.png          # Top-N player heatmaps
```

### Limitations with COCO ball detection

When using a generic COCO-trained YOLO model (`yolov8n.pt`), ball detection is typically sparse (< 30% of frames) because the "sports ball" class is small and low-priority in COCO training data. This means:

- Possession percentages are estimated from available frames only — confidence is reported.
- Pass/shot/tackle counts may be incomplete — the analytics engine produces warnings when ball coverage is low.
- xG values are only computed for detected shot events, which may miss shots in frames without ball detection.
- All analytics outputs include confidence metrics so you can judge reliability.

For better results, use a football-specific fine-tuned detection model with a dedicated "ball" class.

### Configuration

Analytics settings are in `configs/default.yaml` under the `analytics` key:

```yaml
analytics:
  left_to_right: true         # attacking direction
  pitch_threshold_m: 5.0      # max distance for ball ownership (metres)
  hysteresis_frames: 3         # frames required to switch ball owner
  press_radius_m: 10.0         # radius for press intensity (metres)
```

---

## Installation

### Windows (PowerShell)

```powershell
cd football_ai_pipeline
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### macOS / Linux (bash/zsh)

```bash
cd football_ai_pipeline
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Optional: GPU-accelerated detection (all platforms)

```bash
pip install ultralytics supervision torch
```

---

## Quickstart (no weights needed)

The pipeline runs end-to-end **without any model weights**. Detection, tracking, and keypoints fall back to heuristics or produce empty results. This is useful for testing the pipeline, inspecting the output schema, or working with the in-play filter and video annotation.

```bash
# Process first 100 frames, skip every 5th, no annotated video
python -m football_ai_pipeline -i match.mp4 --max-frames 100 --stride 5 --save-video false
```

What you get without weights:
- `metadata.json` — video properties
- `run_report.json` — coverage metrics + degradation warnings
- `frames.jsonl` — per-frame data (detections will be empty)
- `teams_summary.csv` — empty/zero stats
- `stats/` — analytics JSONs (values will be zeros)

The `run_report.json` will clearly list which modules were degraded and what impact that has.

---

## Full Accuracy (with weights)

For real analytics output you need a detection model. **Weights must be downloaded manually** — the pipeline never fetches anything from the network at runtime.

1. Install ultralytics: `pip install ultralytics supervision`
2. Download YOLOv8 weights manually from the public releases page:
   - https://github.com/ultralytics/assets/releases (look for `yolov8n.pt` or your preferred variant)
   - Or use a football-specific fine-tuned model if available
3. Place the `.pt` file at `models/yolo/yolov8n.pt` (already configured in `default.yaml`):
   ```
   football_ai_pipeline/
     models/yolo/yolov8n.pt     <-- place here
   ```
   Or set a custom path in your config:
   ```yaml
   # configs/default.yaml
   detection:
     weights: "models/yolo/yolov8n.pt"   # relative to package root, or absolute
   ```
4. Verify weights load correctly:
   ```bash
   cd football_ai_pipeline
   python scripts/verify_yolo.py --config configs/default.yaml
   ```
5. Run the full pipeline:
   ```bash
   python -m football_ai_pipeline --config configs/default.yaml --input match.mp4 --output out/
   ```

### Windows PowerShell note

If you get `ModuleNotFoundError`, make sure you `cd` into the `football_ai_pipeline` folder first:

```powershell
cd football_ai_pipeline
python -m football_ai_pipeline -c configs/default.yaml -i ..\match.mp4 -o out\ --max-frames 200 --stride 2 --save-video true
```

---

## CLI Reference

```
python -m football_ai_pipeline [OPTIONS]
```

| Argument | Short | Required | Default | Description |
|----------|-------|----------|---------|-------------|
| `--config` | `-c` | No | `configs/default.yaml` | Path to YAML config |
| `--input` | `-i` | Yes | --- | Path to input video |
| `--output` | `-o` | No | `out/` | Output directory |
| `--max-frames` | | No | all | Process at most N frames |
| `--stride` | | No | config value | Process every Nth frame |
| `--save-video` | | No | config value | `true`/`false` --- write annotated video |

### Examples

```bash
# Full run
python -m football_ai_pipeline -c configs/default.yaml -i match.mp4 -o out/

# Quick debug (10 frames, no video output)
python -m football_ai_pipeline -i match.mp4 --max-frames 10 --stride 1 --save-video false

# Fast scan (every 4th frame)
python -m football_ai_pipeline -i match.mp4 --stride 4 -o fast_out/
```

---

## Streamlit UI

A browser-based interface for running the pipeline and viewing results.

### Install

```bash
pip install streamlit
```

### Launch

```bash
cd football_ai_pipeline
streamlit run ui/app.py
```

This opens a local browser tab where you can:

1. **Select a video** — enter the path to your match video
2. **Set options** — stride, max frames, save annotated video
3. **Choose output directory** — where results are written
4. **Click Run** — pipeline executes with live log streaming and a progress bar
5. **View results** — run report summary, player/team stats tables, detailed analytics, annotated video player

The UI calls the existing CLI as a subprocess, so all configuration and graceful degradation behaviour is identical to the command-line workflow.

---

## Output Structure

```
out/
  metadata.json          # Video metadata (fps, resolution, etc.)
  annotated.mp4          # Annotated video with IDs, team colors, ball marker
  frames.jsonl           # Per-frame tracking + analytics data
  players_summary.csv    # Per-player aggregate stats
  teams_summary.csv      # Per-team aggregate stats
  events.json            # Match events (passes, shots, tackles, interceptions)
  run_report.json        # Coverage + quality + degradation report
  stats/                 # Detailed analytics
    player_summary.json
    team_summary.json
    rolling_summary.json
    team_stats.csv       # Per-team analytics (possession, xG, press intensity)
    player_stats.csv     # Per-player analytics (touches, passes, distance)
  heatmaps/              # Position heatmap PNGs
    team_0.png
    team_1.png
    ball.png
    player_<id>.png
```

### run_report.json

Contains four sections:
- **metadata**: Video source info
- **coverage**: Percentage of frames with homography, ball position, detections; analytics sub-section with ball_owner_pct, event_counts, and warnings
- **degradation**: Which modules were in fallback mode, what outputs are affected, and overall confidence level (`good`, `low`, `very_low`)

---

## Running Tests

```bash
cd football_ai_pipeline
pytest tests/ -v
```

---

## Project Structure

```
football_ai_pipeline/
  __main__.py              # CLI entry point
  configs/default.yaml     # Default configuration
  models/yolo/             # Local YOLO weights (.pt files)
  scripts/
    verify_yolo.py         # Verify YOLO weights loading
    debug_video_read.py    # Decode-only video diagnostic
    smoke_cli.py           # CLI smoke test with stage markers
  ui/
    app.py                 # Streamlit web UI
  src/
    data_models.py         # Core dataclasses (FrameState, PlayerState, BallState)
    video_io/              # FR1: Video ingestion & writing
    segmentation/          # FR2: In-play filter (scene change + green ratio)
    detection/             # FR3: Object detection (YOLO or fallback)
    tracking/              # FR4: Object tracking (ByteTrack or IoU fallback)
    team_classifier/       # FR5: Jersey-colour K-Means clustering
    keypoints/             # FR6: Pitch keypoint detection
    homography/            # FR6: Homography estimation
    transforms/            # FR7: Pixel-to-pitch coordinate mapping
    stats/                 # FR8: All analytics modules (A-E)
    analytics/             # FR9: Possession, events, xG, heatmaps
    visualization/         # FR10: Frame annotation
    exports/               # FR11: JSONL/CSV/JSON export
    pipeline/runner.py     # Pipeline orchestrator
  tests/                   # Unit tests
```

---

## Configuration

All parameters are in `configs/default.yaml`. Key settings:

| Config key | Default | Description |
|-----------|---------|-------------|
| `video.stride` | `2` | Process every Nth frame |
| `video.max_frames` | `null` | Limit total frames (null = all) |
| `detection.model` | `yolov8x` | YOLO model variant |
| `detection.weights` | `null` | Local path to .pt weights (null = skip detection) |
| `detection.device` | `auto` | `auto` / `cpu` / `cuda` / `mps` |
| `keypoints.weights` | `null` | Path to keypoint model weights |
| `stats.sprint_speed_threshold` | `7.0` | m/s threshold for sprint detection |
| `visualization.enabled` | `true` | Write annotated video |
| `pipeline.fail_on_missing_weights` | `false` | `true` = crash if weights missing |

---

## Graceful Degradation

The pipeline always completes and produces valid exports. When optional components are missing:

| Missing component | Fallback behaviour | Impact on stats |
|---|---|---|
| **ultralytics** (YOLO) | No detections produced | All player/ball stats = 0 |
| **supervision** (ByteTrack) | Simple IoU tracker | Track IDs less stable across occlusions |
| **Keypoint model** | Hough line intersections | Homography quality lower |
| **GPU** | Runs on CPU | Slower, but functionally identical |

When running without weights, the pipeline prints a detailed capability summary at startup showing exactly what is available and what is degraded.

---

## Windows Quickstart (PowerShell)

Complete step-by-step to get running on Windows:

```powershell
cd football_ai_pipeline
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install ultralytics

# 1. Verify YOLO weights load
python scripts/verify_yolo.py --config configs/default.yaml

# 2. Test video decoding (catches codec issues)
python scripts/debug_video_read.py test-video-1.mp4

# 3. CLI smoke test (200 frames, stage markers, no video output)
python scripts/smoke_cli.py --config configs/default.yaml --input test-video-1.mp4 --output out_smoke

# 4. Full pipeline run
python -m football_ai_pipeline -c configs/default.yaml -i test-video-1.mp4 -o out --max-frames 200 --stride 2 --save-video true

# 5. Launch Streamlit UI
python -m streamlit run ui/app.py
```

If the pipeline appears to hang, set threading environment variables before running:

```powershell
$env:OMP_NUM_THREADS=1
$env:MKL_NUM_THREADS=1
$env:OPENBLAS_NUM_THREADS=1
$env:NUMEXPR_NUM_THREADS=1
python -m football_ai_pipeline -c configs/default.yaml -i test-video-1.mp4 -o out --max-frames 200 --stride 2
```

---

## Troubleshooting

### 1. `ModuleNotFoundError: No module named 'src'`

You are running from the wrong directory. Either:
- `cd football_ai_pipeline` first, then run `python -m football_ai_pipeline ...`
- Or run from the parent: `python -m football_ai_pipeline ...` (the package adds its own dir to sys.path)

### 2. `Error: Config file not found: configs/default.yaml`

The CLI looks for the config relative to your current working directory. Fix:
```bash
cd football_ai_pipeline
python -m football_ai_pipeline -i ../match.mp4
```
Or pass an absolute path: `--config /full/path/to/default.yaml`

### 3. `Cannot open video: <path>` or `Video not found`

- Check the file path exists and is a valid video (mp4, avi, mkv)
- On Windows, use forward slashes or escaped backslashes: `--input "C:/videos/match.mp4"`
- Ensure opencv-python is installed: `pip install opencv-python`

### 4. YOLO / ultralytics errors

- `No module named 'ultralytics'` --- Run `pip install ultralytics`
- `Weights file not found` --- Download the `.pt` file manually (see "Full Accuracy" above) and set `detection.weights` to its local path
- CUDA out of memory --- Set `detection.device: cpu` in config

### 5. Pipeline runs but all stats are zero

This is expected when running without a detection model. The pipeline produces valid (but empty) exports. To get real stats:
1. `pip install ultralytics supervision`
2. Download weights manually (see "Full Accuracy" section) and set `detection.weights` in config
3. Re-run the pipeline

### 6. Pipeline freezes / hangs mid-run

**Step 1 — Is it a video decode issue?**
```bash
python scripts/debug_video_read.py your_video.mp4
```
If this hangs, re-encode the video:
```bash
ffmpeg -i your_video.mp4 -c:v libx264 -preset fast -crf 23 fixed.mp4
```

**Step 2 — Is it a threading/torch issue?**
Set safe threading vars (Windows PowerShell):
```powershell
$env:OMP_NUM_THREADS=1; $env:MKL_NUM_THREADS=1; $env:OPENBLAS_NUM_THREADS=1; $env:NUMEXPR_NUM_THREADS=1
```
Then run again. The Streamlit UI sets these automatically.

**Step 3 — Identify the exact stage:**
```bash
python scripts/smoke_cli.py --config configs/default.yaml --input your_video.mp4 --output out_debug
```
This prints per-frame stage timings. If it stalls, the last printed stage shows where.

### 7. Streamlit UI freezes

The UI runs the pipeline as a subprocess (not in-process), so it should not freeze the browser tab.
If it does, use the **Stop** button to terminate the subprocess, then check the logs tab.

---

## Free & Local Compliance

This pipeline is designed to run fully offline with zero network dependencies at runtime.

**Confirmed paid services used**: NONE
- No API keys, tokens, or cloud credentials are required or accepted
- No Roboflow, Hugging Face Inference, OpenAI, or any other hosted API is used
- All dependencies are free and open-source (MIT / Apache / BSD licensed)

**Confirmed network calls at runtime**: NONE
- The pipeline makes zero HTTP/HTTPS requests during execution
- No `requests`, `urllib`, `httpx`, or similar networking libraries are imported
- YOLO model loading requires an explicit local file path (`detection.weights`); passing a bare model name that would trigger ultralytics auto-download is blocked
- If `detection.weights` is null/unset, detection is skipped entirely (no download attempted)

**Where weights are expected locally** (all optional — pipeline runs without them):

| Config key | Expected location | How to obtain |
|---|---|---|
| `detection.weights` | e.g. `weights/yolov8x.pt` | Manual download from https://github.com/ultralytics/assets/releases |
| `keypoints.weights` | e.g. `weights/keypoints.pt` | Train or obtain a pitch keypoint model separately |

Place weight files anywhere on disk and set the corresponding config key to the absolute or relative path.

---

## Known Limitations

- Player identity / jersey number OCR is not implemented
- Multi-camera stitching is not supported
- Real-time live processing is not supported
- Keypoint matching uses a naive heuristic; a trained model will significantly improve homography quality
- Team classification requires sufficient sample frames for K-Means convergence
- xT grid uses simplified Karun Singh values; custom grids can be substituted

---

## Dependencies

**Required**: numpy, scipy, scikit-learn, opencv-python, shapely, matplotlib, pyyaml, pandas, tqdm

**Optional**: ultralytics, supervision, torch, umap-learn
