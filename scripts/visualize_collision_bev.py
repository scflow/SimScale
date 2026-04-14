#!/usr/bin/env python3
"""
Visualize collision counterfactual traces on top of the original NAVSIM BEV scene.

This script uses the original NAVSIM scene renderer for the base map / annotations and
overlays the attacked rollout from `generated_collision_pseudo_data*/*.json`.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import io
import json
import math
import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from PIL import Image

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import navsim.common.dataclasses as navsim_dataclasses
from navsim.common.dataclasses import Scene, SensorConfig
from navsim.common.dataloader import MetricCacheLoader
from navsim.visualization.bev import add_map_to_bev_ax, add_oriented_box_to_bev_ax
from navsim.visualization.plots import configure_ax, configure_bev_ax

try:
    import generate_ood_mini as base
except ModuleNotFoundError:
    from scripts import generate_ood_mini as base


TraceDict = Dict[str, Any]
_WORKER_METRIC_CACHE_LOADER: Optional[MetricCacheLoader] = None
_WORKER_LOG_ROOT: Optional[Path] = None
_WORKER_SENSOR_ROOT: Optional[Path] = None

COUNTERFACTUAL_TRAJ_COLOR = "#D7263D"
ACTOR_TRAJ_COLOR = "#FF8C42"
EGO_BOX_COLOR = "#111111"
ATTACK_VEHICLE_CONFIG = {
    "fill_color": "#D7263D",
    "fill_color_alpha": 0.20,
    "line_color": "#D7263D",
    "line_color_alpha": 0.95,
    "line_width": 1.3,
    "line_style": "-",
    "zorder": 35,
}
OTHER_VEHICLE_CONFIG = {
    "fill_color": "#355C7D",
    "fill_color_alpha": 0.08,
    "line_color": "#355C7D",
    "line_color_alpha": 0.55,
    "line_width": 0.9,
    "line_style": "-",
    "zorder": 30,
}
EGO_VEHICLE_CONFIG = {
    "fill_color": EGO_BOX_COLOR,
    "fill_color_alpha": 0.10,
    "line_color": EGO_BOX_COLOR,
    "line_color_alpha": 0.95,
    "line_width": 1.4,
    "line_style": "-",
    "zorder": 45,
}

PACIFICA = get_pacifica_parameters()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize collision traces on NAVSIM BEV.")
    parser.add_argument("--input", type=Path, required=True, help="Collision .json/.pkl file or directory.")
    parser.add_argument("--glob", type=str, default="collision_*.json", help="Glob when --input is a directory.")
    parser.add_argument("--max-files", type=int, default=None, help="Max number of files to render.")
    parser.add_argument(
        "--metric-cache-path",
        type=Path,
        required=True,
        help="Metric cache root used to resolve token -> log_name.",
    )
    parser.add_argument(
        "--data-split",
        type=str,
        default="mini",
        help="Dataset split under OPENSCENE_DATA_ROOT/navsim_logs. Example: mini, navmini, trainval.",
    )
    parser.add_argument(
        "--log-root",
        type=Path,
        default=None,
        help="Override log root. Defaults to $OPENSCENE_DATA_ROOT/navsim_logs/<data-split>.",
    )
    parser.add_argument(
        "--sensor-root",
        type=Path,
        default=None,
        help="Override sensor root. Defaults to $OPENSCENE_DATA_ROOT/sensor_blobs/<data-split>.",
    )
    parser.add_argument(
        "--map-root",
        type=Path,
        default=None,
        help="Override map root. Defaults to ./maps, then $NUPLAN_MAPS_ROOT, then $OPENSCENE_DATA_ROOT/maps.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <input>/visuals_collision_bev.",
    )
    parser.add_argument(
        "--frame-idx",
        type=str,
        default="mid",
        help="Counterfactual frame to show vehicle boxes: start|mid|end|<int>.",
    )
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--make-gif", action="store_true", help="Also export animated GIF.")
    parser.add_argument("--gif-fps", type=float, default=None, help="GIF frame rate. Default: infer from trace.")
    parser.add_argument("--make-mp4", action="store_true", help="Also export animated MP4.")
    parser.add_argument("--mp4-fps", type=float, default=None, help="MP4 frame rate. Default: infer from trace.")
    parser.add_argument("--mp4-crf", type=int, default=20, help="MP4 quality; lower is better.")
    parser.add_argument("--ffmpeg-bin", type=str, default="ffmpeg", help="ffmpeg executable path.")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of worker processes. Use 0 for auto.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def list_input_files(input_path: Path, pattern: str, max_files: Optional[int]) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    files = sorted(input_path.glob(pattern))
    if max_files is not None:
        files = files[:max_files]
    return files


def load_trace(path: Path) -> TraceDict:
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if path.suffix.lower() == ".pkl":
        with path.open("rb") as f:
            return pickle.load(f)
    raise ValueError(f"Unsupported trace format: {path}")


def parse_overlay_frame(frame_idx: str, num_frames: int) -> int:
    if frame_idx == "start":
        return 0
    if frame_idx == "mid":
        return max(0, min(num_frames - 1, num_frames // 2))
    if frame_idx == "end":
        return max(0, num_frames - 1)
    idx = int(frame_idx)
    return max(0, min(num_frames - 1, idx))


def resolve_roots(log_root: Optional[Path], sensor_root: Optional[Path], data_split: str) -> Tuple[Path, Path]:
    if log_root is not None and sensor_root is not None:
        return log_root.expanduser().resolve(), sensor_root.expanduser().resolve()

    openscene_root = Path(str(Path.home()))  # placeholder to keep type checker happy
    import os

    env_root = os.environ.get("OPENSCENE_DATA_ROOT")
    if env_root is None:
        raise RuntimeError(
            "OPENSCENE_DATA_ROOT is not set. Either export it or pass --log-root and --sensor-root explicitly."
        )
    openscene_root = Path(env_root).expanduser().resolve()

    final_log_root = log_root.expanduser().resolve() if log_root is not None else openscene_root / "navsim_logs" / data_split
    final_sensor_root = (
        sensor_root.expanduser().resolve() if sensor_root is not None else openscene_root / "sensor_blobs" / data_split
    )
    return final_log_root, final_sensor_root


def resolve_map_root(map_root: Optional[Path]) -> Path:
    if map_root is not None:
        resolved = map_root.expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"--map-root does not exist: {resolved}")
        return resolved

    repo_maps = REPO_ROOT / "maps"
    if repo_maps.exists():
        return repo_maps.resolve()

    env_map_root = os.environ.get("NUPLAN_MAPS_ROOT")
    if env_map_root:
        resolved = Path(env_map_root).expanduser().resolve()
        if resolved.exists():
            return resolved

    env_data_root = os.environ.get("OPENSCENE_DATA_ROOT")
    if env_data_root:
        candidate = Path(env_data_root).expanduser().resolve() / "maps"
        if candidate.exists():
            return candidate

    raise RuntimeError(
        "Unable to resolve map root. Pass --map-root explicitly or set NUPLAN_MAPS_ROOT."
    )


def configure_map_root(map_root: Path) -> None:
    os.environ["NUPLAN_MAPS_ROOT"] = str(map_root)
    navsim_dataclasses.NUPLAN_MAPS_ROOT = str(map_root)


def auto_workers(num_workers: int) -> int:
    if num_workers > 0:
        return num_workers
    return max(1, min(os.cpu_count() or 1, 32))


def global_to_local(x: float, y: float, origin_x: float, origin_y: float, origin_heading: float) -> Tuple[float, float]:
    dx = x - origin_x
    dy = y - origin_y
    c = math.cos(origin_heading)
    s = math.sin(origin_heading)
    local_x = c * dx + s * dy
    local_y = -s * dx + c * dy
    return local_x, local_y


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def load_scene_for_token(
    token: str,
    metric_cache_loader: MetricCacheLoader,
    log_root: Path,
    sensor_root: Path,
) -> Scene:
    metric_cache = metric_cache_loader.get_from_token(token)
    log_path = log_root / f"{metric_cache.log_name}.pkl"
    if not log_path.exists():
        raise FileNotFoundError(f"Missing log pickle for token={token}: {log_path}")

    with log_path.open("rb") as f:
        log_frames: List[Dict[str, Any]] = pickle.load(f)

    frame_dict = None
    for candidate in log_frames:
        if str(candidate.get("token")) == token:
            frame_dict = candidate
            break
    if frame_dict is None:
        raise RuntimeError(f"Token {token} not found in {log_path}")

    return Scene.from_scene_dict_list(
        scene_dict_list=[frame_dict],
        sensor_blobs_path=sensor_root,
        num_history_frames=1,
        num_future_frames=0,
        sensor_config=SensorConfig.build_no_sensors(),
    )


def init_worker(metric_cache_root: str, log_root: str, sensor_root: str, map_root: str) -> None:
    global _WORKER_METRIC_CACHE_LOADER, _WORKER_LOG_ROOT, _WORKER_SENSOR_ROOT
    configure_map_root(Path(map_root))
    _WORKER_METRIC_CACHE_LOADER = base.build_metric_cache_loader(Path(metric_cache_root))
    _WORKER_LOG_ROOT = Path(log_root)
    _WORKER_SENSOR_ROOT = Path(sensor_root)


def plot_trace_overlay(ax: plt.Axes, trace: TraceDict, overlay_idx: int, path_end_idx: Optional[int] = None) -> None:
    frames = trace["frames"]
    path_end_idx = overlay_idx if path_end_idx is None else path_end_idx
    path_end_idx = max(0, min(len(frames) - 1, path_end_idx))
    origin = frames[0]["ego"]
    origin_x = float(origin["x"])
    origin_y = float(origin["y"])
    origin_heading = float(origin["heading"])

    ego_local: List[Tuple[float, float]] = []
    actor_local: List[Tuple[float, float]] = []
    actor_token = str(trace["attack_spec"]["actor_token"])

    for frame in frames[: path_end_idx + 1]:
        ego = frame["ego"]
        ex, ey = global_to_local(
            float(ego["x"]),
            float(ego["y"]),
            origin_x,
            origin_y,
            origin_heading,
        )
        ego_local.append((ex, ey))

        actor_match = next((v for v in frame.get("vehicles", []) if str(v.get("id")) == actor_token), None)
        if actor_match is not None:
            ax_x, ax_y = global_to_local(
                float(actor_match["x"]),
                float(actor_match["y"]),
                origin_x,
                origin_y,
                origin_heading,
            )
            actor_local.append((ax_x, ax_y))

    if ego_local:
        xs = [p[1] for p in ego_local]
        ys = [p[0] for p in ego_local]
        ax.plot(xs, ys, color=COUNTERFACTUAL_TRAJ_COLOR, linewidth=2.4, zorder=40, label="Counterfactual ego")

    if actor_local:
        xs = [p[1] for p in actor_local]
        ys = [p[0] for p in actor_local]
        ax.plot(xs, ys, color=ACTOR_TRAJ_COLOR, linewidth=2.0, linestyle="--", zorder=41, label="Attack actor")

    frame = frames[overlay_idx]
    ego = frame["ego"]
    ego_lx, ego_ly = global_to_local(
        float(ego["x"]),
        float(ego["y"]),
        origin_x,
        origin_y,
        origin_heading,
    )
    ego_heading_local = wrap_angle(float(ego["heading"]) - origin_heading)
    ego_box = OrientedBox(
        StateSE2(ego_lx, ego_ly, ego_heading_local),
        float(PACIFICA.length),
        float(PACIFICA.width),
        float(PACIFICA.height),
    )
    add_oriented_box_to_bev_ax(ax, ego_box, EGO_VEHICLE_CONFIG)

    for vehicle in frame.get("vehicles", []):
        lx, ly = global_to_local(
            float(vehicle["x"]),
            float(vehicle["y"]),
            origin_x,
            origin_y,
            origin_heading,
        )
        local_heading = wrap_angle(float(vehicle["heading"]) - origin_heading)
        box = OrientedBox(
            StateSE2(lx, ly, local_heading),
            float(vehicle["length"]),
            float(vehicle["width"]),
            1.5,
        )
        cfg = ATTACK_VEHICLE_CONFIG if str(vehicle.get("id")) == actor_token else OTHER_VEHICLE_CONFIG
        add_oriented_box_to_bev_ax(ax, box, cfg)


def build_caption(trace: TraceDict, overlay_idx: int) -> str:
    nominal = trace.get("nominal_metrics", {})
    attacked = trace.get("attacked_metrics", {})
    attack = trace.get("attack_spec", {})
    lines = [
        f"mode={attack.get('mode')} actor={attack.get('actor_token')} start={attack.get('start_time_s')}s",
        f"risk_gain={float(trace.get('risk_gain', float('nan'))):.3f} overlay_frame={overlay_idx}",
        (
            "nominal: "
            f"pdm={float(nominal.get('pdm_score', float('nan'))):.3f} "
            f"ttc={float(nominal.get('time_to_collision_within_bound', float('nan'))):.2f} "
            f"dist={float(trace.get('nominal_min_dist_m', float('nan'))):.2f}m"
        ),
        (
            "attacked: "
            f"pdm={float(attacked.get('pdm_score', float('nan'))):.3f} "
            f"ttc={float(attacked.get('time_to_collision_within_bound', float('nan'))):.2f} "
            f"dist={float(trace.get('attacked_min_dist_m', float('nan'))):.2f}m"
        ),
    ]
    return "\n".join(lines)


def infer_trace_fps(trace: TraceDict) -> float:
    if "time_step_s" in trace:
        try:
            dt = float(trace["time_step_s"])
            if dt > 0:
                return 1.0 / dt
        except Exception:
            pass

    frames = trace.get("frames", [])
    timestamps: List[float] = []
    for frame in frames:
        ts = frame.get("timestamp")
        if ts is None:
            continue
        try:
            timestamps.append(float(ts))
        except Exception:
            continue
    if len(timestamps) >= 2:
        deltas = [b - a for a, b in zip(timestamps[:-1], timestamps[1:]) if b > a]
        if deltas:
            median_dt = float(np.median(deltas))
            if median_dt > 1e6:
                return 1e6 / median_dt
            if median_dt > 1e3:
                return 1e3 / median_dt
            if median_dt > 0:
                return 1.0 / median_dt

    if frames and all("t_idx" in frame for frame in frames):
        # Collision traces in this repo are generated at 0.1s spacing.
        return 10.0

    return 10.0


def build_figure(scene: Scene, trace: TraceDict, overlay_idx: int, dpi: int, path_end_idx: Optional[int] = None) -> plt.Figure:
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    add_map_to_bev_ax(ax, scene.map_api, StateSE2(*scene.frames[0].ego_status.ego_pose))
    configure_bev_ax(ax)
    configure_ax(ax)
    plot_trace_overlay(ax, trace, overlay_idx, path_end_idx=path_end_idx)
    ax.legend(loc="upper left", fontsize=8)
    ax.text(
        0.01,
        0.01,
        build_caption(trace, overlay_idx),
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
        ha="left",
        bbox=dict(facecolor="white", alpha=0.82, edgecolor="none", pad=3.0),
        zorder=60,
    )
    fig.set_dpi(dpi)
    fig.tight_layout()
    return fig


def figure_to_image(fig: plt.Figure, dpi: int) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    with Image.open(buf) as image:
        result = image.convert("RGB")
    buf.close()
    return result


def render_gif(
    scene: Scene,
    trace: TraceDict,
    out_path: Path,
    dpi: int,
    fps: float,
) -> None:
    images = [
        figure_to_image(build_figure(scene, trace, idx, dpi, path_end_idx=idx), dpi)
        for idx in range(int(trace["num_frames"]))
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = int(round(1000.0 / max(1.0, fps)))
    images[0].save(out_path, save_all=True, append_images=images[1:], duration=duration_ms, loop=0)
    for image in images:
        image.close()


def render_mp4(
    scene: Scene,
    trace: TraceDict,
    out_path: Path,
    dpi: int,
    fps: float,
    crf: int,
    ffmpeg_bin: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="collision_bev_mp4_") as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        for idx in range(int(trace["num_frames"])):
            image = figure_to_image(build_figure(scene, trace, idx, dpi, path_end_idx=idx), dpi)
            frame_path = tmp_dir_path / f"frame_{idx:06d}.png"
            image.save(frame_path, format="PNG")
            image.close()

        cmd = [
            ffmpeg_bin,
            "-y",
            "-framerate",
            str(max(1.0, fps)),
            "-i",
            str(tmp_dir_path / "frame_%06d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            str(crf),
            "-movflags",
            "+faststart",
            str(out_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed for {out_path}: {result.stderr.strip()}")


def render_one(
    file_path: Path,
    output_dir: Path,
    metric_cache_loader: MetricCacheLoader,
    log_root: Path,
    sensor_root: Path,
    frame_idx_arg: str,
    dpi: int,
    make_gif: bool,
    gif_fps: float,
    make_mp4: bool,
    mp4_fps: float,
    mp4_crf: int,
    ffmpeg_bin: str,
    overwrite: bool,
) -> Path:
    trace = load_trace(file_path)
    token = str(trace["token"])
    overlay_idx = parse_overlay_frame(frame_idx_arg, int(trace["num_frames"]))
    inferred_fps = infer_trace_fps(trace)
    effective_gif_fps = float(gif_fps) if gif_fps is not None else inferred_fps
    effective_mp4_fps = float(mp4_fps) if mp4_fps is not None else inferred_fps
    out_path = output_dir / f"{file_path.stem}_bev.png"
    if out_path.exists() and not overwrite:
        return out_path

    scene = load_scene_for_token(token, metric_cache_loader, log_root, sensor_root)
    fig = build_figure(scene, trace, overlay_idx, dpi)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    if make_gif:
        gif_path = output_dir / f"{file_path.stem}_bev.gif"
        if overwrite or not gif_path.exists():
            render_gif(scene, trace, gif_path, dpi=dpi, fps=effective_gif_fps)
        print(f"[saved] {gif_path}")

    if make_mp4:
        mp4_path = output_dir / f"{file_path.stem}_bev.mp4"
        if overwrite or not mp4_path.exists():
            render_mp4(
                scene,
                trace,
                mp4_path,
                dpi=dpi,
                fps=effective_mp4_fps,
                crf=mp4_crf,
                ffmpeg_bin=ffmpeg_bin,
            )
        print(f"[saved] {mp4_path}")

    return out_path


def render_one_worker(
    file_path: str,
    output_dir: str,
    frame_idx_arg: str,
    dpi: int,
    make_gif: bool,
    gif_fps: Optional[float],
    make_mp4: bool,
    mp4_fps: Optional[float],
    mp4_crf: int,
    ffmpeg_bin: str,
    overwrite: bool,
) -> str:
    assert _WORKER_METRIC_CACHE_LOADER is not None
    assert _WORKER_LOG_ROOT is not None
    assert _WORKER_SENSOR_ROOT is not None
    out_path = render_one(
        file_path=Path(file_path),
        output_dir=Path(output_dir),
        metric_cache_loader=_WORKER_METRIC_CACHE_LOADER,
        log_root=_WORKER_LOG_ROOT,
        sensor_root=_WORKER_SENSOR_ROOT,
        frame_idx_arg=frame_idx_arg,
        dpi=dpi,
        make_gif=make_gif,
        gif_fps=gif_fps,
        make_mp4=make_mp4,
        mp4_fps=mp4_fps,
        mp4_crf=mp4_crf,
        ffmpeg_bin=ffmpeg_bin,
        overwrite=overwrite,
    )
    return str(out_path)


def main() -> None:
    args = parse_args()
    files = list_input_files(args.input, args.glob, args.max_files)
    if not files:
        raise RuntimeError(f"No files found under {args.input} with pattern {args.glob}")

    metric_cache_root = base.resolve_metric_cache_root(args.metric_cache_path)
    metric_cache_loader = base.build_metric_cache_loader(metric_cache_root)
    log_root, sensor_root = resolve_roots(args.log_root, args.sensor_root, args.data_split)
    map_root = resolve_map_root(args.map_root)
    configure_map_root(map_root)

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else (args.input.parent if args.input.is_file() else args.input) / "visuals_collision_bev"
    )

    print(f"Rendering {len(files)} file(s) to {output_dir.resolve()}")
    print(f"Using map root: {map_root}")
    worker_count = auto_workers(args.num_workers)
    if worker_count <= 1 or len(files) <= 1:
        for file_path in files:
            out_path = render_one(
                file_path=file_path,
                output_dir=output_dir,
                metric_cache_loader=metric_cache_loader,
                log_root=log_root,
                sensor_root=sensor_root,
                frame_idx_arg=args.frame_idx,
                dpi=args.dpi,
                make_gif=args.make_gif,
                gif_fps=args.gif_fps,
                make_mp4=args.make_mp4,
                mp4_fps=args.mp4_fps,
                mp4_crf=args.mp4_crf,
                ffmpeg_bin=args.ffmpeg_bin,
                overwrite=args.overwrite,
            )
            print(f"[saved] {out_path}")
        return

    print(f"Using {worker_count} workers")
    with ProcessPoolExecutor(
        max_workers=worker_count,
        initializer=init_worker,
        initargs=(str(metric_cache_root), str(log_root), str(sensor_root), str(map_root)),
    ) as executor:
        futures = {
            executor.submit(
                render_one_worker,
                str(file_path),
                str(output_dir),
                args.frame_idx,
                args.dpi,
                args.make_gif,
                args.gif_fps,
                args.make_mp4,
                args.mp4_fps,
                args.mp4_crf,
                args.ffmpeg_bin,
                args.overwrite,
            ): file_path
            for file_path in files
        }
        for future in as_completed(futures):
            file_path = futures[future]
            out_path = future.result()
            print(f"[saved] {out_path} ({file_path.name})")


if __name__ == "__main__":
    main()
