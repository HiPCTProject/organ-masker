# sam2_runner.py
import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from skimage.measure import block_reduce
from scipy.ndimage import median_filter
from sam2.build_sam import build_sam2_video_predictor
import importlib, importlib.resources as ir, shutil


# ---------------- Checkpoint + config helpers ----------------

def _find_checkpoint(model_name: str) -> Path:
    ckpt_name = f"sam2.1_hiera_{model_name}.pt"
    candidates = []

    env_dir = os.environ.get("SAM2_CHECKPOINT_DIR", "")
    if env_dir:
        candidates.append(Path(env_dir) / ckpt_name)

    here = Path(__file__).resolve()
    for root in [here.parents[1], here.parents[0]]:
        candidates.append(root / "checkpoints" / ckpt_name)
    candidates.append(Path.cwd() / "checkpoints" / ckpt_name)

    ckpt = next((c for c in candidates if c.exists()), None)
    if ckpt is None:
        raise FileNotFoundError(
            "SAM2 checkpoint not found. Looked at:\n" +
            "\n".join(str(c) for c in candidates) +
            "\nSet SAM2_CHECKPOINT_DIR or place the .pt under a 'checkpoints' folder."
        )
    return ckpt


def _discover_keys_for_model(model_name: str):
    suf = {"tiny": "t", "small": "s", "base": "b", "large": "l"}[model_name]
    default = [
        f"sam2.1_hiera_{model_name}",
        f"sam2.1_hiera_{suf}",
        f"sam2.1/sam2.1_hiera_{model_name}",
        f"sam2.1/sam2.1_hiera_{suf}",
    ]
    keys = []
    try:
        pkg = importlib.import_module("sam2")
        base = ir.files(pkg) / "configs" / "sam2.1"
        if base.is_dir():
            for e in base.iterdir():
                if e.name.endswith(".yaml"):
                    stem = e.name[:-5]
                    keys.append(stem)
                    keys.append(f"sam2.1/{stem}")
    except Exception:
        pass
    prio = [k for k in keys if k.endswith(f"_{model_name}") or k.endswith(f"_{suf}")]
    seen = set()
    out = []
    for k in default + prio + keys:
        if k not in seen:
            out.append(k)
            seen.add(k)
    return out


def _extract_package_configs_to_local():
    target = Path(__file__).resolve().parents[1] / "configs" / "sam2.1"
    try:
        pkg = importlib.import_module("sam2")
        base = ir.files(pkg) / "configs" / "sam2.1"
        if not base.is_dir():
            return None
        target.mkdir(parents=True, exist_ok=True)
        for e in base.iterdir():
            if e.name.endswith(".yaml"):
                with ir.as_file(e) as src_path:
                    shutil.copyfile(src_path, target / e.name)
        return target
    except Exception:
        return None


def build_predictor(model_name: str):
    ckpt = _find_checkpoint(model_name)

    last_exc = None
    keys = _discover_keys_for_model(model_name)
    for key in keys:
        try:
            pred = build_sam2_video_predictor(key, str(ckpt))
            return pred, str(ckpt), key
        except Exception as e:
            last_exc = e

    local_dir = _extract_package_configs_to_local()
    if local_dir is None:
        raise RuntimeError(
            "Failed to load SAM2 using Hydra keys and could not extract package configs.\n"
            f"Last Hydra error: {last_exc}"
        )

    suf = {"tiny": "t", "small": "s", "base": "b", "large": "l"}[model_name]
    long_yaml = local_dir / f"sam2.1_hiera_{model_name}.yaml"
    short_yaml = local_dir / f"sam2.1_hiera_{suf}.yaml"
    cfg_path = long_yaml if long_yaml.exists() else short_yaml
    if not cfg_path.exists():
        raise RuntimeError(
            "Extracted configs but could not find a matching yaml:\n"
            f"  looked for: {long_yaml.name} or {short_yaml.name} in {local_dir}\n"
            f"Last Hydra error before extraction: {last_exc}"
        )
    pred = build_sam2_video_predictor(str(cfg_path), str(ckpt))
    return pred, str(ckpt), str(cfg_path)


# ---------------- Video writers (square-pixel, non-deforming) ----------------

def _make_video_writer(seg_dir, stem, fps, frame_size, codec):
    codec = codec.lower()
    if codec == "mjpg":
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        ext = ".avi"
    elif codec == "xvid":
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        ext = ".avi"
    elif codec == "mp4v":
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        ext = ".mp4"
    elif codec == "avc1":
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        ext = ".mp4"
    else:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        ext = ".avi"
        codec = "mjpg"

    out_path = (Path(seg_dir) / f"{stem}{ext}").as_posix()
    vw = cv2.VideoWriter(out_path, fourcc, float(fps), frame_size, isColor=True)
    if not vw.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {out_path} (codec={codec})")
    return out_path, vw


def _make_video_writer_mp4(seg_dir, stem, fps, size_hw, fourcc_name="mp4v"):
    seg_dir = Path(seg_dir)
    seg_dir.mkdir(parents=True, exist_ok=True)
    out_path = seg_dir / f"{stem}.mp4"
    H, W = size_hw
    frame_size = (W, H)
    fourcc = cv2.VideoWriter_fourcc(*fourcc_name)
    vw = cv2.VideoWriter(str(out_path), fourcc, float(fps), frame_size)
    if not vw.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {out_path} with fourcc={fourcc_name}")
    return str(out_path), vw


def _norm_to_u8(arr, vmin, vmax, ds, no_median=False):
    if ds and ds > 1:
        arr = block_reduce(arr, (ds, ds), np.mean).astype(np.float32)
    else:
        arr = arr.astype(np.float32) if no_median else median_filter(arr.astype(np.float32), size=2)
    return np.clip((arr - vmin) / max(vmax - vmin, 1e-6) * 255, 0, 255).astype(np.uint8)


def write_xy_videos_streaming_norm(
    im_dir,
    hoatools,
    datasetname,
    priv_meta,
    seg_dir,
    fps=10,
    window=(1, 99),
    downsample=None,
    vmin=None,
    vmax=None,
    codec="mp4v",
    no_median=False,
    hoa_downsample_level=2,
    center_index=None,
):
    """
    Two XY videos centered on ROI z index:
      - forward:  z_center, z_center+ds, ...
      - backward: z_center, z_center-ds, ...
    Returns:
      paths dict with:
        xy_fwd, _xy_fwd_len, _xy_fwd_z_indices
        xy_bwd, _xy_bwd_len, _xy_bwd_z_indices
      and dims (Z, Hd, Wd).
    """
    paths = {}
    ds = downsample if downsample else 1

    # ---- readers ----
    if hoatools:
        import hoa_tools.dataset as hoa_dataset
        if priv_meta:
            from pathlib import Path as _P
            hoa_dataset.change_metadata_directory(_P(priv_meta))
        d = hoa_dataset.get_dataset(datasetname)
        da = d.data_array(downsample_level=hoa_downsample_level)
        Z = int(da.sizes["z"])
        reader = lambda i: da.isel(z=i).values
    else:
        p = Path(im_dir)
        files = sorted([f for f in p.iterdir() if f.suffix.lower() in [".tif", ".tiff", ".jp2"]])
        Z = len(files)
        reader = lambda i: cv2.imread(str(files[i]), -1)

    if Z == 0:
        raise RuntimeError("No slices found for XY video writing.")

    # first slice -> size
    sl0_full = reader(0)
    sl0 = _norm_to_u8(sl0_full, vmin, vmax, ds, no_median=no_median)
    Hd, Wd = sl0.shape[:2]

    # ROI centered indices
    if center_index is None:
        z_center = Z // 2
    else:
        z_center = max(0, min(Z - 1, int(center_index)))

    z_fwd = [z_center] + list(range(z_center + ds, Z, ds))
    z_bwd = [z_center] + list(range(z_center - ds, -1, -ds))

    # Forward XY video
    xy_fwd_path, vw_fwd = _make_video_writer(seg_dir, "video_xy_fwd", fps, (Wd, Hd), codec)
    for zi in tqdm(z_fwd, desc="XY forward video", unit="slice"):
        frame_full = reader(zi)
        frame = _norm_to_u8(frame_full, vmin, vmax, ds, no_median=no_median)
        if frame.shape[:2] != (Hd, Wd):
            frame = cv2.resize(frame, (Wd, Hd), interpolation=cv2.INTER_NEAREST)
        vw_fwd.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    vw_fwd.release()

    # Backward XY video
    xy_bwd_path, vw_bwd = _make_video_writer(seg_dir, "video_xy_bwd", fps, (Wd, Hd), codec)
    for zi in tqdm(z_bwd, desc="XY backward video", unit="slice"):
        frame_full = reader(zi)
        frame = _norm_to_u8(frame_full, vmin, vmax, ds, no_median=no_median)
        if frame.shape[:2] != (Hd, Wd):
            frame = cv2.resize(frame, (Wd, Hd), interpolation=cv2.INTER_NEAREST)
        vw_bwd.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    vw_bwd.release()

    # Store paths and index mappings
    paths["xy_fwd"] = xy_fwd_path
    paths["_xy_fwd_len"] = len(z_fwd)
    paths["_xy_fwd_z_indices"] = z_fwd

    paths["xy_bwd"] = xy_bwd_path
    paths["_xy_bwd_len"] = len(z_bwd)
    paths["_xy_bwd_z_indices"] = z_bwd

    # Backward compatible aliases (use forward)
    paths["xy"] = xy_fwd_path
    paths["_xy_len"] = len(z_fwd)

    return paths, (Z, Hd, Wd)


# --- Streaming orthogonal videos (ROI-centered forward/backward sweeps) ------------

def write_plane_videos_streaming(
    im_dir,
    hoatools,
    datasetname,
    priv_meta,
    seg_dir,
    fps=10,
    downsample=None,
    vmin=None,
    vmax=None,
    codec="mp4v",
    no_median=False,
    hoa_downsample_level=2,
    planes=None,
    roi_indices=None,
):
    """
    Stream-write ROI-centered YZ and/or XZ MP4 videos from disk/HOA.

    Geometry:
      - YZ frames: (Hd, Zds), frame index runs along x_d
      - XZ frames: (Zds, Wd), frame index runs along y_d

    For each requested plane we create:
      - forward video: starting at ROI index, increasing along x or y
      - backward video: starting at ROI index, decreasing along x or y

    We also store index mappings in `paths` so accumulation can map frame -> axis.
    """
    planes = planes or []
    paths = {}
    if not planes:
        return paths

    ds = int(downsample) if downsample else 1
    assert ds >= 1
    roi_indices = roi_indices or {}

    # reader
    if hoatools:
        import hoa_tools.dataset as hoa_dataset
        if priv_meta:
            from pathlib import Path as _P
            hoa_dataset.change_metadata_directory(_P(priv_meta))
        d = hoa_dataset.get_dataset(datasetname)
        da = d.data_array(downsample_level=hoa_downsample_level)
        Z = int(da.sizes["z"])

        def reader(i):
            return da.isel(z=i).values
    else:
        p = Path(im_dir)
        files = sorted([f for f in p.iterdir() if f.suffix.lower() in [".tif", ".tiff", ".jp2"]])
        if not files:
            raise FileNotFoundError(f"No slices found in {p}")
        Z = len(files)

        def reader(i):
            arr = cv2.imread(str(files[i]), -1)
            if arr is None:
                raise IOError(f"Failed to read image: {files[i]}")
            return arr

    # normalization helper
    _vmin, _vmax = vmin, vmax

    def _norm(arr, infer=False):
        nonlocal _vmin, _vmax
        a = arr.astype(np.float32)
        if not no_median:
            a = median_filter(a, size=2)
        if ds and ds > 1:
            a = block_reduce(a, (ds, ds), np.mean)
        if infer or _vmin is None or _vmax is None:
            _vmin = float(np.nanmin(a))
            _vmax = float(np.nanmax(a))
            if _vmax <= _vmin:
                _vmax = _vmin + 1.0
        a = (a - _vmin) / max((_vmax - _vmin), 1e-6)
        return (a * 255.0).clip(0, 255).astype(np.uint8)

    sl0 = _norm(reader(0), infer=True)
    Hd, Wd = int(sl0.shape[0]), int(sl0.shape[1])
    z_steps = list(range(0, Z, ds))
    Zds = len(z_steps)

    # YZ videos (x-axis scan)
    if "yz" in planes:
        x_center = roi_indices.get("yz", Wd // 2)
        x_center = max(0, min(Wd - 1, int(x_center)))

        x_fwd = list(range(x_center, Wd))
        x_bwd = list(range(x_center, -1, -1))

        # forward
        yz_fwd_path, vw = _make_video_writer_mp4(seg_dir, "video_yz_fwd", fps, (Hd, Zds), fourcc_name=codec)
        for x_d in tqdm(x_fwd, desc="YZ mp4 forward (columns)", unit="col"):
            fr = np.empty((Hd, Zds), dtype=np.uint8)
            fr[:, 0] = sl0[:, x_d]
            for zi, z in enumerate(z_steps[1:], start=1):
                sl = _norm(reader(z))
                if sl.shape[:2] != (Hd, Wd):
                    sl = cv2.resize(sl, (Wd, Hd), interpolation=cv2.INTER_NEAREST)
                fr[:, zi] = sl[:, x_d]
            vw.write(cv2.cvtColor(fr, cv2.COLOR_GRAY2BGR))
        vw.release()

        # backward
        yz_bwd_path, vw = _make_video_writer_mp4(seg_dir, "video_yz_bwd", fps, (Hd, Zds), fourcc_name=codec)
        for x_d in tqdm(x_bwd, desc="YZ mp4 backward (columns)", unit="col"):
            fr = np.empty((Hd, Zds), dtype=np.uint8)
            fr[:, 0] = sl0[:, x_d]
            for zi, z in enumerate(z_steps[1:], start=1):
                sl = _norm(reader(z))
                if sl.shape[:2] != (Hd, Wd):
                    sl = cv2.resize(sl, (Wd, Hd), interpolation=cv2.INTER_NEAREST)
                fr[:, zi] = sl[:, x_d]
            vw.write(cv2.cvtColor(fr, cv2.COLOR_GRAY2BGR))
        vw.release()

        paths["yz_fwd"] = yz_fwd_path
        paths["_yz_fwd_len"] = len(x_fwd)
        paths["_yz_fwd_x_indices"] = x_fwd

        paths["yz_bwd"] = yz_bwd_path
        paths["_yz_bwd_len"] = len(x_bwd)
        paths["_yz_bwd_x_indices"] = x_bwd

        # Backwards compatible alias
        paths["yz"] = yz_fwd_path
        paths["_yz_len"] = len(x_fwd)

    # XZ videos (y-axis scan)
    if "xz" in planes:
        y_center = roi_indices.get("xz", Hd // 2)
        y_center = max(0, min(Hd - 1, int(y_center)))

        y_fwd = list(range(y_center, Hd))
        y_bwd = list(range(y_center, -1, -1))

        # forward
        xz_fwd_path, vw = _make_video_writer_mp4(seg_dir, "video_xz_fwd", fps, (Zds, Wd), fourcc_name=codec)
        for y_d in tqdm(y_fwd, desc="XZ mp4 forward (rows)", unit="row"):
            fr = np.empty((Zds, Wd), dtype=np.uint8)
            fr[0, :] = sl0[y_d, :]
            for zi, z in enumerate(z_steps[1:], start=1):
                sl = _norm(reader(z))
                if sl.shape[:2] != (Hd, Wd):
                    sl = cv2.resize(sl, (Wd, Hd), interpolation=cv2.INTER_NEAREST)
                fr[zi, :] = sl[y_d, :]
            vw.write(cv2.cvtColor(fr, cv2.COLOR_GRAY2BGR))
        vw.release()

        # backward
        xz_bwd_path, vw = _make_video_writer_mp4(seg_dir, "video_xz_bwd", fps, (Zds, Wd), fourcc_name=codec)
        for y_d in tqdm(y_bwd, desc="XZ mp4 backward (rows)", unit="row"):
            fr = np.empty((Zds, Wd), dtype=np.uint8)
            fr[0, :] = sl0[y_d, :]
            for zi, z in enumerate(z_steps[1:], start=1):
                sl = _norm(reader(z))
                if sl.shape[:2] != (Hd, Wd):
                    sl = cv2.resize(sl, (Wd, Hd), interpolation=cv2.INTER_NEAREST)
                fr[zi, :] = sl[y_d, :]
            vw.write(cv2.cvtColor(fr, cv2.COLOR_GRAY2BGR))
        vw.release()

        paths["xz_fwd"] = xz_fwd_path
        paths["_xz_fwd_len"] = len(y_fwd)
        paths["_xz_fwd_y_indices"] = y_fwd

        paths["xz_bwd"] = xz_bwd_path
        paths["_xz_bwd_len"] = len(y_bwd)
        paths["_xz_bwd_y_indices"] = y_bwd

        # Backwards compatible alias
        paths["xz"] = xz_fwd_path
        paths["_xz_len"] = len(y_fwd)

    for k in ["yz_fwd", "yz_bwd", "xz_fwd", "xz_bwd"]:
        fp = paths.get(k)
        if fp and (not os.path.exists(fp) or os.path.getsize(fp) == 0):
            raise RuntimeError(f"Wrote {k} at {fp} but file is missing or empty.")

    return paths


# --- Memmap orthogonal videos (ROI-centered forward/backward sweeps) ---------------

def write_plane_videos_from_memmap(
    mmap_path: str,
    z_y_x: tuple[int, int, int],  # (Z, Hd, Wd)
    seg_dir,
    start_plane: str,
    start_idx: int,
    fps: int = 10,
    codec: str = "mp4v",
    planes: list[str] | None = None,
    downsample: int = 1,
    roi_indices: dict | None = None,
):
    planes = planes or []
    paths = {}
    roi_indices = roi_indices or {}

    Z, Hd, Wd = map(int, z_y_x)
    mm = np.memmap(mmap_path, dtype="uint8", mode="r").reshape((Z, Hd, Wd))

    ds = int(downsample) if downsample else 1
    z_steps = list(range(0, Z, ds))
    Zds = len(z_steps)

    # YZ from memmap
    if "yz" in planes:
        x_center = roi_indices.get("yz", Wd // 2)
        x_center = max(0, min(Wd - 1, int(x_center)))

        x_fwd = list(range(x_center, Wd))
        x_bwd = list(range(x_center, -1, -1))

        yz_fwd_path, vw = _make_video_writer_mp4(seg_dir, "video_yz_fwd", fps, (Hd, Zds), fourcc_name=codec)
        for x in tqdm(x_fwd, desc="YZ video (memmap) forward: columns", unit="col"):
            fr = np.empty((Hd, Zds), dtype=np.uint8)
            for zi, z in enumerate(z_steps):
                fr[:, zi] = mm[z, :, x]
            vw.write(cv2.cvtColor(fr, cv2.COLOR_GRAY2BGR))
        vw.release()

        yz_bwd_path, vw = _make_video_writer_mp4(seg_dir, "video_yz_bwd", fps, (Hd, Zds), fourcc_name=codec)
        for x in tqdm(x_bwd, desc="YZ video (memmap) backward: columns", unit="col"):
            fr = np.empty((Hd, Zds), dtype=np.uint8)
            for zi, z in enumerate(z_steps):
                fr[:, zi] = mm[z, :, x]
            vw.write(cv2.cvtColor(fr, cv2.COLOR_GRAY2BGR))
        vw.release()

        paths["yz_fwd"] = yz_fwd_path
        paths["_yz_fwd_len"] = len(x_fwd)
        paths["_yz_fwd_x_indices"] = x_fwd

        paths["yz_bwd"] = yz_bwd_path
        paths["_yz_bwd_len"] = len(x_bwd)
        paths["_yz_bwd_x_indices"] = x_bwd

        paths["yz"] = yz_fwd_path
        paths["_yz_len"] = len(x_fwd)

    # XZ from memmap
    if "xz" in planes:
        y_center = roi_indices.get("xz", Hd // 2)
        y_center = max(0, min(Hd - 1, int(y_center)))

        y_fwd = list(range(y_center, Hd))
        y_bwd = list(range(y_center, -1, -1))

        xz_fwd_path, vw = _make_video_writer_mp4(seg_dir, "video_xz_fwd", fps, (Zds, Wd), fourcc_name=codec)
        for y in tqdm(y_fwd, desc="XZ video (memmap) forward: rows", unit="row"):
            fr = np.empty((Zds, Wd), dtype=np.uint8)
            for zi, z in enumerate(z_steps):
                fr[zi, :] = mm[z, y, :]
            vw.write(cv2.cvtColor(fr, cv2.COLOR_GRAY2BGR))
        vw.release()

        xz_bwd_path, vw = _make_video_writer_mp4(seg_dir, "video_xz_bwd", fps, (Zds, Wd), fourcc_name=codec)
        for y in tqdm(y_bwd, desc="XZ video (memmap) backward: rows", unit="row"):
            fr = np.empty((Zds, Wd), dtype=np.uint8)
            for zi, z in enumerate(z_steps):
                fr[zi, :] = mm[z, y, :]
            vw.write(cv2.cvtColor(fr, cv2.COLOR_GRAY2BGR))
        vw.release()

        paths["xz_fwd"] = xz_fwd_path
        paths["_xz_fwd_len"] = len(y_fwd)
        paths["_xz_fwd_y_indices"] = y_fwd

        paths["xz_bwd"] = xz_bwd_path
        paths["_xz_bwd_len"] = len(y_bwd)
        paths["_xz_bwd_y_indices"] = y_bwd

        paths["xz"] = xz_fwd_path
        paths["_xz_len"] = len(y_fwd)

    for k in ["yz_fwd", "yz_bwd", "xz_fwd", "xz_bwd"]:
        fp = paths.get(k)
        if fp and (not os.path.exists(fp) or os.path.getsize(fp) == 0):
            raise RuntimeError(f"Wrote {k} at {fp} but file is missing or empty.")

    return paths


# ---------------- Vote accumulation helpers ----------------

def _padcrop_2d(v: np.ndarray, H: int, W: int) -> np.ndarray:
    v = np.asarray(v)
    h, w = v.shape[:2]
    out = np.zeros((H, W), dtype=v.dtype)
    hh = min(h, H)
    ww = min(w, W)
    out[:hh, :ww] = v[:hh, :ww]
    return out


def _padcrop_1d(a: np.ndarray, L: int) -> np.ndarray:
    a = np.asarray(a)
    n = a.shape[0]
    if n == L:
        return a
    out = np.zeros((L,), dtype=a.dtype)
    out[:min(n, L)] = a[:min(n, L)]
    return out


def accumulate_xy_votes(votes_mm: np.ndarray, masks_xy, z_indices):
    """
    Accumulate XY masks into votes_mm using an explicit z index list.
    """
    Z, H, W = votes_mm.shape
    if len(masks_xy) != len(z_indices):
        raise ValueError("masks_xy and z_indices must have the same length.")

    for m, z in zip(masks_xy, z_indices):
        z = int(z)
        if z < 0 or z >= Z:
            continue
        m = _padcrop_2d(m, H, W)
        votes_mm[z, :, :] += (m.astype(np.uint8) & 1)


def accumulate_yz_votes(votes_mm: np.ndarray, masks_yz, x_indices, ds: int):
    """
    Accumulate YZ masks into votes_mm.

    masks_yz[k] has shape (Hd, Zds), where columns run along z_steps = range(0, Z, ds).
    x_indices[k] is the x index in [0, W-1] corresponding to that frame.
    """
    Z, H, W = votes_mm.shape
    if len(masks_yz) != len(x_indices):
        raise ValueError("masks_yz and x_indices must have the same length.")
    step = max(1, int(ds))

    for fr, x in zip(masks_yz, x_indices):
        x = int(x)
        if x < 0 or x >= W:
            continue
        fr = _padcrop_2d(fr, H, fr.shape[1])
        Zds = fr.shape[1]
        for zds in range(Zds):
            z = zds * step
            if z >= Z:
                break
            col = (fr[:, zds].astype(np.uint8) & 1)
            col = _padcrop_1d(col, H)
            votes_mm[z, :, x] += col


def accumulate_xz_votes(votes_mm: np.ndarray, masks_xz, y_indices, ds: int):
    """
    Accumulate XZ masks into votes_mm.

    masks_xz[k] has shape (Zds, Wd), where rows run along z_steps = range(0, Z, ds).
    y_indices[k] is the y index in [0, H-1] corresponding to that frame.
    """
    Z, H, W = votes_mm.shape
    if len(masks_xz) != len(y_indices):
        raise ValueError("masks_xz and y_indices must have the same length.")
    step = max(1, int(ds))

    for fr, y in zip(masks_xz, y_indices):
        y = int(y)
        if y < 0 or y >= H:
            continue
        fr = _padcrop_2d(fr, fr.shape[0], W)
        Zds = fr.shape[0]
        for zds in range(Zds):
            z = zds * step
            if z >= Z:
                break
            row = (fr[zds, :].astype(np.uint8) & 1)
            row = _padcrop_1d(row, W)
            votes_mm[z, y, :] += row


# ---------------- SAM2 seeding + propagation ----------------

def _apply_single_selection(predictor, state, sel, frame_idx):
    boxes = sel.get("boxes") or []
    neg_boxes = sel.get("neg_boxes") or []
    points = sel.get("points")
    point_labels = sel.get("point_labels")
    pos_mask = sel.get("pos_mask")
    neg_mask = sel.get("neg_mask")

    for b in boxes:
        predictor.add_new_points_or_box(state, frame_idx=frame_idx, box=np.array(b), obj_id=1)
    for nb in neg_boxes:
        predictor.add_new_points_or_box(state, frame_idx=frame_idx, box=np.array(nb), obj_id=0)

    if points is not None:
        predictor.add_new_points_or_box(
            state,
            frame_idx=frame_idx,
            points=np.array(points),
            labels=np.array(point_labels),
            obj_id=1,
        )
    if pos_mask is not None:
        predictor.add_new_mask(state, frame_idx=frame_idx, mask=pos_mask.astype(bool), obj_id=1)
    if neg_mask is not None:
        predictor.add_new_mask(state, frame_idx=frame_idx, mask=neg_mask.astype(bool), obj_id=0)


def _read_binary_mask_any(path: Path, target_hw: tuple[int, int]) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Could not read mask: {path}")
    H, W = target_hw
    if img.shape[:2] != (H, W):
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_NEAREST)
    return (img > 0).astype(np.uint8)


def _propagate_collect(predictor, state, prefer_obj_id: int = 1):
    out = []
    for _, out_obj_ids, mask_logits in predictor.propagate_in_video(state):
        masks_np = [(m > 0).cpu().numpy().astype(np.uint8).squeeze() for m in mask_logits]

        chosen = None
        if out_obj_ids is not None and len(out_obj_ids) == len(masks_np):
            try:
                idx = list(out_obj_ids).index(prefer_obj_id)
                chosen = masks_np[idx]
            except ValueError:
                pos_idxs = [i for i, oid in enumerate(out_obj_ids) if oid > 0]
                if pos_idxs:
                    comb = np.zeros_like(masks_np[0], dtype=np.uint8)
                    for i in pos_idxs:
                        comb |= masks_np[i]
                    chosen = comb

        if chosen is None:
            chosen = masks_np[0] if len(masks_np) > 0 else None
        if chosen is None:
            raise RuntimeError("No masks returned by predictor for a frame.")
        out.append(chosen.astype(np.uint8))
    return out


def run_video_along_plane_with_multislice_prompts(
    predictor,
    plane,                 # 'xy' | 'yz' | 'xz'
    seg_dir,
    paths,
    selections,
    init_mask_path,
    init_mask_slice,
    seed_frame_hw,
    downsample=1,
    direction="fwd",       # 'fwd' or 'bwd'
):
    """
    Run SAM2 along a plane using ROI-centered forward or backward video.

    For each plane (xy, yz, xz), we now have:
      - {plane}_fwd: ROI-centered forward sweep
      - {plane}_bwd: ROI-centered backward sweep

    We always seed at frame 0 for that plane and direction. This assumes
    a single ROI slice per plane, with the corresponding slice aligned
    to frame 0 by the video writers.
    """
    if direction not in ("fwd", "bwd"):
        raise ValueError(f"direction must be 'fwd' or 'bwd', got {direction}")

    key = f"{plane}_{direction}"
    len_key = f"_{plane}_{direction}_len"
    video_path = paths.get(key)
    num_frames = paths.get(len_key)

    if video_path is None:
        raise RuntimeError(f"Video for plane '{plane}' and direction '{direction}' not available.")
    if num_frames is None or num_frames <= 0:
        raise RuntimeError(f"Cannot infer number of frames for plane '{plane}' ({direction}).")

    ds = max(1, int(downsample) if downsample is not None else 1)
    H, W = seed_frame_hw

    # Init predictor state
    state = predictor.init_state(video_path)

    # Helper: seed a selection at frame 0
    def _seed_selection_at_frame0(sel_dict):
        _apply_single_selection(predictor, state, sel_dict, frame_idx=0)

    # Seed from init_mask (file or folder) at frame 0
    if init_mask_path:
        p = Path(init_mask_path)
        if p.is_file():
            if init_mask_slice is None:
                raise ValueError("--init-mask-slice is required with a single --init-mask file.")
            m0 = _read_binary_mask_any(p, (H, W))
            sel_init = {
                "slice_idx": int(init_mask_slice),
                "boxes": [],
                "neg_boxes": [],
                "points": None,
                "point_labels": None,
                "pos_mask": m0,
                "neg_mask": None,
            }
            _seed_selection_at_frame0(sel_init)
        else:
            # Folder of mask_{zzzz}.*. Use masks that exist, seed at frame 0.
            for sel in selections:
                sli = int(sel.get("slice_idx", 0))
                found = None
                for ext in (".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"):
                    cand = p / f"mask_{sli:04d}{ext}"
                    if cand.exists():
                        found = cand
                        break
                if found is not None:
                    m0 = _read_binary_mask_any(found, (H, W))
                    sel_init = {
                        "slice_idx": sli,
                        "boxes": [],
                        "neg_boxes": [],
                        "points": None,
                        "point_labels": None,
                        "pos_mask": m0,
                        "neg_mask": None,
                    }
                    _seed_selection_at_frame0(sel_init)

    # Seed from interactive selections (all at frame 0)
    for sel in selections:
        _seed_selection_at_frame0(sel)

    print(f"[{plane} {direction}] video={video_path} frames={num_frames} ds={ds}")
    masks = _propagate_collect(predictor, state)
    return masks
