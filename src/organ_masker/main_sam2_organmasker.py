#!/usr/bin/env python3
# main_sam2_organmasker.py
#
# Streaming SAM2 organ/background masker with:
# - ROI-centred forward & backward videos per plane (XY, YZ, XZ)
# - One ROI slice per plane (XY + optional YZ/XZ)
# - Orthogonal accumulation with consistent dimensions
# - Memory-mapped vote fusion and optional hole filling

import os
import cv2
import json
import argparse
import numpy as np
from pathlib import Path
from tempfile import NamedTemporaryFile
from alive_progress import alive_bar

from roi_ui import (
    select_rects_overlay, select_points_overlay, select_circles_overlay,
    draw_mask_overlay, fit_to_screen, load_reused_prompts
)
from volume_io import (
    get_dims_streaming,
    sample_percentiles_streaming,
    get_plane_frame_streaming,
    infer_downsampled_hw_from_first_slice,
    build_memmap_normalized_minimal_dims,
    choose_windowing_streaming,
)
from postproc import fill_2d_holes
from scipy.signal import medfilt2d
from sam2_runner import (
    build_predictor,
    write_xy_videos_streaming_norm,
    write_plane_videos_streaming,
    write_plane_videos_from_memmap,
    run_video_along_plane_with_multislice_prompts,
    accumulate_xy_votes,
    accumulate_yz_votes,
    accumulate_xz_votes,
)


# ---------------- argparse ----------------
def parse_args():
    p = argparse.ArgumentParser(
        "SAM2 organ masker (ROI-centred forward/backward + orthogonal planes)"
    )

    # I/O & dataset
    p.add_argument("im_dir", type=str, help="Dir with TIFF/JP2 slices (ignored if --hoatools)")
    p.add_argument("--output", type=str, default=".", help="Output directory")
    p.add_argument("--hoatools", action="store_true", help="Use hoa_tools loader (HiP-CT)")
    p.add_argument("--datasetname", type=str, help="hoa_tools dataset name")
    p.add_argument("--privatemetadatapath", type=str, help="hoa_tools private metadata path")
    p.add_argument(
        "--hoa-downsample-level",
        type=int,
        default=2,
        help="hoa_tools data_array downsample level (0=full res; 1=2x; 2=4x; ...)",
    )

    # Model & processing
    p.add_argument("--model", choices=["tiny", "small", "base", "large"], default="small")
    p.add_argument("--window", type=int, nargs=2, default=(1, 99), metavar=("LOW", "HIGH"))
    p.add_argument(
        "--downsample",
        type=int,
        default=None,
        help="Additional in-plane downsample factor (>=1).",
    )
    p.add_argument("--fill-holes", action="store_true")
    p.add_argument(
        "--forward-only",
        action="store_true",
        help="If set, only run forward propagation (no backward) per plane.",
    )

    # Optional interactive rescale (manual override)
    p.add_argument(
        "--manual-intensityrescale",
        action="store_true",
        help="Use an interactive widget to pick vmin/vmax instead of pure percentiles.",
    )

    # Orthogonal control
    p.add_argument("--orthogonal", action="store_true", default=False,
                   help="Also run YZ and XZ planes.")
    p.add_argument("--orthogonal-planes", nargs="+", choices=["yz", "xz"], default=None,
                   help="Run only these orthogonal plane(s). Example: --orthogonal-planes xz")

    # Orthogonal writer policy
    p.add_argument("--orthogonal-writer", choices=["stream", "memmap", "auto"], default="auto",
                   help="'stream' re-reads slices; 'memmap' uses a temp memmap; 'auto' picks for you.")
    p.add_argument("--keep-memmap", action="store_true", help="Don't delete temp memmap.")

    # Codec & speed knobs
    p.add_argument("--codec", choices=["mp4v", "mjpg", "xvid", "avc1"], default="mp4v")
    p.add_argument("--no-median-intensity", action="store_true",
                   help="Skip tiny median on intensities when downsample<=1.")
    p.add_argument("--pctl-slices", type=int, default=32,
                   help="How many slices to sample for percentiles.")
    p.add_argument("--pctl-pixel-stride", type=int, default=8,
                   help="Pixel stride within sampled slices for percentiles.")

    # ROI selection – one ROI slice per plane
    p.add_argument("--roi-plane", choices=["xy", "yz", "xz"], default="xy",
                   help="Plane on which you will draw/select ROIs first.")
    p.add_argument("--roi-slice", type=int, nargs="+", default=None,
                   help="Slice index (Z) for ROI selection on XY plane; exactly one used.")
    p.add_argument("--roi-slice-yz", type=int, nargs="+", default=None,
                   help="X index (column) for ROI selection on YZ plane; exactly one used.")
    p.add_argument("--roi-slice-xz", type=int, nargs="+", default=None,
                   help="Y index (row) for ROI selection on XZ plane; exactly one used.")
    p.add_argument("--roi-mode", choices=["box", "multi-box", "points", "circle", "draw"], default="box")
    p.add_argument("--neg-box", action="append", default=[], help="Repeatable: 'x,y,w,h' to force BG.")

    # Existing mask/ROI seeding
    p.add_argument("--init-mask", type=str, default=None,
                   help="Seed mask file (PNG/TIF/...) with --init-mask-slice, or folder of mask_{zzzz}.*")
    p.add_argument("--init-mask-slice", type=int, default=None,
                   help="Required if --init-mask is a single image file.")
    p.add_argument("--reuse-roi", type=str, default=None,
                   help="Path to a previously exported ROI folder to reuse prompts from.")

    # Merge orthogonal predictions
    p.add_argument(
        "--merge-orth",
        choices=["none", "union", "intersect", "majority"],
        default="none",
        help="How to merge planes. 'none' keeps only selection plane.",
    )

    # Optional: restore masks to pre-local-downsample size (per HOA level)
    p.add_argument("--restore-original-shape", action="store_true",
                   help="Upsample final masks back to (H,W) if --downsample>1.")

    return p.parse_args()


# ---------------- small utils ----------------
def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def _delete_if_exists(path_str: str | None):
    if not path_str:
        return
    p = Path(path_str)
    if p.exists():
        try:
            p.unlink()
        except Exception:
            pass


def _write_plane_settings_json(roi_dir: Path, plane: str, settings: dict):
    f = roi_dir / f"roi_{plane}_settings.json"
    f.write_text(json.dumps(settings, indent=2))


# ---------------- ROI collection wrapper ----------------
def _collect_selections_for_plane(args, plane, slices_list, vmin, vmax, ds, Hds, Wds):
    selections = []
    for sli in slices_list:
        frame_u8 = get_plane_frame_streaming(
            args.im_dir, args.hoatools, args.datasetname, args.privatemetadatapath,
            plane=plane, idx=sli, vmin=vmin, vmax=vmax, downsample=ds,
            hoa_downsample_level=args.hoa_downsample_level
        )
        frame_bgr = cv2.cvtColor(frame_u8, cv2.COLOR_GRAY2BGR)
        frame_bgr_fit, scale = fit_to_screen(frame_bgr)

        boxes = []
        neg_boxes = []
        points = None
        point_labels = None
        pos_mask = None
        neg_mask = None

        def unscale_rect(r):
            x1, y1, x2, y2 = r
            if scale != 1.0:
                x1 = int(round(x1 / scale)); y1 = int(round(y1 / scale))
                x2 = int(round(x2 / scale)); y2 = int(round(y2 / scale))
            return [x1, y1, x2, y2]

        def unscale_points(pts):
            if scale == 1.0:
                return pts
            return [(int(round(x / scale)), int(round(y / scale))) for (x, y) in pts]

        confirmed = False
        while not confirmed:
            if args.roi_mode == "box":
                rects = select_rects_overlay(frame_bgr_fit, multi=False)
                boxes = [unscale_rect(rects[0])] if rects else []
                confirmed = bool(rects)
            elif args.roi_mode == "multi-box":
                rects = select_rects_overlay(frame_bgr_fit, multi=True)
                boxes = [unscale_rect(r) for r in rects] if rects else []
                confirmed = bool(rects)
            elif args.roi_mode == "points":
                pts, labs = select_points_overlay(frame_bgr_fit)
                if pts:
                    pts = unscale_points(pts)
                    points = np.array(pts, np.int32)
                    point_labels = np.array(labs, np.int32)
                    confirmed = True
            elif args.roi_mode == "circle":
                circles = select_circles_overlay(frame_bgr_fit)  # (x,y,r,label)
                if circles:
                    Hs, Ws = frame_u8.shape[:2]
                    pos_mask = np.zeros((Hs, Ws), np.uint8)
                    neg_mask = np.zeros((Hs, Ws), np.uint8)
                    for (x, y, r, lbl) in circles:
                        if scale != 1.0:
                            x = int(round(x / scale)); y = int(round(y / scale)); r = int(round(r / scale))
                        yy, xx = np.ogrid[:Hs, :Ws]
                        m = (xx - x) ** 2 + (yy - y) ** 2 <= r * r
                        if lbl == 1:
                            pos_mask[m] = 1
                        else:
                            neg_mask[m] = 1
                    confirmed = True
            else:  # draw
                pos_mask, neg_mask, _ = draw_mask_overlay(frame_bgr_fit)
                confirmed = (pos_mask is not None) and (pos_mask.any() or (neg_mask is not None and neg_mask.any()))

            if not confirmed:
                print(f"[ROI-{plane}] No confirmation (ENTER). Retry...")

        for nb in args.neg_box:
            x, y, w, h = map(int, nb.split(","))
            neg_boxes.append([x, y, x + w, y + h])

        meta = {
            "plane": plane,
            "slice": int(sli),
            "downsample": int(ds),
            "hoa_downsample_level": int(args.hoa_downsample_level),
            "window": list(map(int, args.window)),
            "codec": args.codec,
            "roi_mode": args.roi_mode,
            "boxes": boxes,
            "neg_boxes": neg_boxes,
            "points": (points.tolist() if points is not None else None),
            "point_labels": (point_labels.tolist() if point_labels is not None else None),
        }
        (args._roi_dir / f"roi_{plane}_{sli:06d}.json").write_text(json.dumps(meta, indent=2))
        if pos_mask is not None:
            cv2.imwrite(str(args._roi_dir / f"roi_{plane}_{sli:06d}_pos.png"), (pos_mask * 255).astype("uint8"))
        if neg_mask is not None:
            cv2.imwrite(str(args._roi_dir / f"roi_{plane}_{sli:06d}_neg.png"), (neg_mask * 255).astype("uint8"))

        selections.append({
            "slice_idx": int(sli),
            "boxes": boxes,
            "neg_boxes": neg_boxes,
            "points": points,
            "point_labels": point_labels,
            "pos_mask": pos_mask,
            "neg_mask": neg_mask,
        })
    return selections


# ---------------- main ----------------
def main():
    args = parse_args()

    out_root = Path(args.output) / (
        f"segmentation_SAM2_{args.datasetname}" if args.hoatools else f"segmentation_SAM2_{Path(args.im_dir).name}"
    )
    seg_dir = out_root
    masks_dir = seg_dir / "masks"
    roi_dir = seg_dir / "roi"
    ensure_dir(masks_dir)
    ensure_dir(roi_dir)
    args._roi_dir = roi_dir

    print("Processing image stack...")
    Z, H, W = get_dims_streaming(
        args.im_dir, args.hoatools, args.datasetname,
        args.privatemetadatapath, args.hoa_downsample_level
    )
    ds = args.downsample if args.downsample else 1

    # 1) percentile-based default windowing
    vmin, vmax = sample_percentiles_streaming(
        args.im_dir, args.hoatools, args.datasetname, args.privatemetadatapath,
        Z, tuple(args.window), ds, num_slices=args.pctl_slices,
        pixel_stride=args.pctl_pixel_stride, hoa_downsample_level=args.hoa_downsample_level
    )

    # 2) optional interactive/manual override
    if args.manual_intensityrescale:
        # choose which slice drives the GUI:
        # - if ROI is on XY and user specified --roi-slice, use that index
        # - otherwise, use the middle slice
        if args.roi_plane == "xy" and args.roi_slice:
            # we expect only one ROI slice in current setup; take the first
            center_slice = sorted(set(args.roi_slice))[0]
        else:
            center_slice = Z // 2

        center_slice = max(0, min(Z - 1, int(center_slice)))

        vmin, vmax = choose_windowing_streaming(
            im_dir=args.im_dir,
            hoatools=args.hoatools,
            datasetname=args.datasetname,
            priv_meta=args.privatemetadatapath,
            Z=Z,
            downsample=ds,
            hoa_downsample_level=args.hoa_downsample_level,
            num_slices=args.pctl_slices,
            center_slice=center_slice,
        )

    Hd, Wd = infer_downsampled_hw_from_first_slice(
        args.im_dir, args.hoatools, args.datasetname,
        args.privatemetadatapath, ds, args.hoa_downsample_level
    )
    Zds = len(range(0, Z, ds)) if ds > 1 else Z

    def default_slices_for_plane(plane):
        if plane == "xy":
            return [Z // 2]
        if plane == "yz":
            return [W // 2]  # x-index
        return [H // 2]      # y-index

    def _ensure_single_slice(slices, plane_name):
        if len(slices) != 1:
            raise ValueError(
                f"Exactly one ROI slice must be specified for plane '{plane_name}'. "
                f"Got: {list(slices)}"
            )
        return [int(slices[0])]

    print("Selecting/loading ROI...")

    # Determine orth planes
    orth_planes = []
    if args.orthogonal:
        orth_planes = ["yz", "xz"]
    if args.orthogonal_planes:
        orth_planes = list(sorted(set(args.orthogonal_planes)))
    # We'll treat "selection plane" separately in inference, so don't double count later
    orth_planes_for_infer = [pl for pl in orth_planes if pl != args.roi_plane]

    roi_slices = {}           # per-plane (original indices: Z for xy, x for yz, y for xz)
    selections_by_plane = {}

    # ----- Selection plane ROI -----
    if args.roi_slice is None:
        sel_slices = default_slices_for_plane(args.roi_plane)
    else:
        sel_slices = sorted(set(args.roi_slice))
    sel_slices = _ensure_single_slice(sel_slices, args.roi_plane)
    roi_slices[args.roi_plane] = sel_slices[0]

    # Write settings json for selection plane
    def write_plane_settings(plane):
        _write_plane_settings_json(
            roi_dir,
            plane,
            {
                "plane": plane,
                "Z": int(Z), "H": int(H), "W": int(W),
                "Zds": int(Zds), "Hd": int(Hd), "Wd": int(Wd),
                "downsample": int(ds),
                "hoa_downsample_level": int(args.hoa_downsample_level),
                "window": [int(args.window[0]), int(args.window[1])],
                "codec": args.codec,
                "model": args.model,
                "vmin": float(vmin), "vmax": float(vmax),
                "roi_slice": int(roi_slices.get(plane, default_slices_for_plane(plane)[0])),
            }
        )

    write_plane_settings(args.roi_plane)

    # Collect ROI for selection plane
    selections_by_plane[args.roi_plane] = _collect_selections_for_plane(
        args, args.roi_plane, sel_slices, vmin, vmax, ds, Hd, Wd
    )

    # ROI reuse on selection plane
    if args.reuse_roi:
        reused = load_reused_prompts(
            roi_folder=args.reuse_roi,
            target_plane=args.roi_plane,
            target_downsample=ds,
            target_hoa_level=args.hoa_downsample_level,
            target_hw=(Hd, Wd),
        )
        if reused:
            print(f"Reused {len(reused)} ROI(s) from: {args.reuse_roi}")
            selections_by_plane[args.roi_plane].extend(reused)
        else:
            print(f"No matching {args.roi_plane.upper()} ROIs found in: {args.reuse_roi}")

    # ----- Orthogonal plane ROIs (if requested) -----
    for pl in orth_planes_for_infer:
        if pl == "yz":
            user_slices = args.roi_slice_yz
        else:  # xz
            user_slices = args.roi_slice_xz

        if user_slices:
            slices_list = sorted(set(user_slices))
        else:
            slices_list = default_slices_for_plane(pl)

        slices_list = _ensure_single_slice(slices_list, pl)
        roi_slices[pl] = slices_list[0]
        write_plane_settings(pl)

        selections_by_plane[pl] = _collect_selections_for_plane(
            args, pl, slices_list, vmin, vmax, ds, Hd, Wd
        )

    # ---------------- Write XY videos (ROI-centred) ----------------
    print("Writing videos...")

    # Centre z index for XY videos: if ROI on XY use that; otherwise mid-slab
    if args.roi_plane == "xy":
        center_z_xy = roi_slices["xy"]
    else:
        center_z_xy = Z // 2

    paths, (Z_ret, Hd_ret, Wd_ret) = write_xy_videos_streaming_norm(
        im_dir=args.im_dir,
        hoatools=args.hoatools,
        datasetname=args.datasetname,
        priv_meta=args.privatemetadatapath,
        seg_dir=seg_dir,
        fps=10,
        downsample=ds,
        vmin=vmin,
        vmax=vmax,
        codec=args.codec,
        no_median=args.no_median_intensity,
        hoa_downsample_level=args.hoa_downsample_level,
        center_index=center_z_xy,
    )

    # ---------------- Orthogonal videos (YZ/XZ, ROI-centred) ----------------
    tmp_mmap = None
    planes_for_videos = list(sorted(set(
        ([args.roi_plane] if args.roi_plane in ("yz", "xz") else []) + orth_planes_for_infer
    )))

    if len(planes_for_videos) > 0:
        # ROI indices for orthogonal writers are in downsampled coordinates
        roi_indices_ds = {}
        if "yz" in planes_for_videos:
            # roi_slices["yz"] is x index in original space
            x_orig = roi_slices.get("yz", W // 2)
            roi_indices_ds["yz"] = max(0, min(Wd - 1, int(x_orig // ds)))
        if "xz" in planes_for_videos:
            # roi_slices["xz"] is y index in original space
            y_orig = roi_slices.get("xz", H // 2)
            roi_indices_ds["xz"] = max(0, min(Hd - 1, int(y_orig // ds)))

        # Decide writer mode
        if args.hoatools:
            writer_mode = "stream"
        else:
            writer_mode = {"auto": "memmap", "stream": "stream", "memmap": "memmap"}[args.orthogonal_writer]

        if writer_mode == "stream":
            paths.update(
                write_plane_videos_streaming(
                    im_dir=args.im_dir,
                    hoatools=args.hoatools,
                    datasetname=args.datasetname,
                    priv_meta=args.privatemetadatapath,
                    seg_dir=seg_dir,
                    fps=10,
                    downsample=ds,
                    vmin=vmin,
                    vmax=vmax,
                    codec=args.codec,
                    no_median=args.no_median_intensity,
                    hoa_downsample_level=args.hoa_downsample_level,
                    planes=planes_for_videos,
                    roi_indices=roi_indices_ds,
                )
            )
        else:
            # memmap path
            with NamedTemporaryFile(delete=False, suffix=".mmap") as tf:
                tmp_mmap = tf.name
            (Zc, Hc, Wc), _, _ = build_memmap_normalized_minimal_dims(
                args.im_dir, tmp_mmap,
                window=tuple(args.window), downsample=ds,
                hoatools=args.hoatools, datasetname=args.datasetname,
                priv_meta=args.privatemetadatapath,
                hoa_downsample_level=args.hoa_downsample_level,
                vmin=vmin, vmax=vmax,
            )
            paths.update(
                write_plane_videos_from_memmap(
                    mmap_path=tmp_mmap,
                    z_y_x=(Zc, Hc, Wc),
                    seg_dir=seg_dir,
                    start_plane=args.roi_plane,
                    start_idx=center_z_xy,
                    fps=10,
                    codec=args.codec,
                    planes=planes_for_videos,
                    downsample=ds,
                    roi_indices=roi_indices_ds,
                )
            )

    # ---------------- Build predictor ----------------
    predictor, ckpt, ckpt_key = build_predictor(args.model)
    print(f"Loaded SAM2:\n  checkpoint: {ckpt}")

    # ---------------- Votes memmap (Z,Hd,Wd) ----------------
    with NamedTemporaryFile(delete=False, suffix=".mmap") as tf:
        votes_path = tf.name
    votes_mm = np.memmap(votes_path, dtype="uint8", mode="w+", shape=(Z, Hd, Wd))
    votes_mm[:] = 0

    # =====================================================================
    #  SAM2 propagation per plane/direction
    # =====================================================================

    do_backward = not args.forward_only

    # Helper for XY accumulation (uses z_indices)
    def _run_xy_plane(selections):
        # forward
        masks_fwd = run_video_along_plane_with_multislice_prompts(
            predictor=predictor,
            plane="xy",
            seg_dir=seg_dir,
            paths=paths,
            selections=selections,
            init_mask_path=args.init_mask,
            init_mask_slice=args.init_mask_slice,
            seed_frame_hw=(Hd, Wd),
            downsample=ds,
            direction="fwd",
        )
        z_idx_fwd = paths.get("_xy_fwd_z_indices", [])
        accumulate_xy_votes(votes_mm, masks_fwd, z_idx_fwd)

        # backward
        if do_backward:
            masks_bwd = run_video_along_plane_with_multislice_prompts(
                predictor=predictor,
                plane="xy",
                seg_dir=seg_dir,
                paths=paths,
                selections=selections,
                init_mask_path=args.init_mask,
                init_mask_slice=args.init_mask_slice,
                seed_frame_hw=(Hd, Wd),
                downsample=ds,
                direction="bwd",
            )
            z_idx_bwd = paths.get("_xy_bwd_z_indices", [])
            accumulate_xy_votes(votes_mm, masks_bwd, z_idx_bwd)

    # Helper for YZ accumulation
    def _run_yz_plane(selections, use_init_mask=False):
        init_mask_path = args.init_mask if use_init_mask else None
        init_mask_slice = args.init_mask_slice if use_init_mask else None

        # forward
        masks_fwd = run_video_along_plane_with_multislice_prompts(
            predictor=predictor,
            plane="yz",
            seg_dir=seg_dir,
            paths=paths,
            selections=selections,
            init_mask_path=init_mask_path,
            init_mask_slice=init_mask_slice,
            seed_frame_hw=(Hd, Zds),
            downsample=ds,
            direction="fwd",
        )
        x_idx_fwd = paths.get("_yz_fwd_x_indices", [])
        accumulate_yz_votes(votes_mm, masks_fwd, x_idx_fwd, ds)

        # backward
        if do_backward:
            masks_bwd = run_video_along_plane_with_multislice_prompts(
                predictor=predictor,
                plane="yz",
                seg_dir=seg_dir,
                paths=paths,
                selections=selections,
                init_mask_path=init_mask_path,
                init_mask_slice=init_mask_slice,
                seed_frame_hw=(Hd, Zds),
                downsample=ds,
                direction="bwd",
            )
            x_idx_bwd = paths.get("_yz_bwd_x_indices", [])
            accumulate_yz_votes(votes_mm, masks_bwd, x_idx_bwd, ds)

    # Helper for XZ accumulation
    def _run_xz_plane(selections, use_init_mask=False):
        init_mask_path = args.init_mask if use_init_mask else None
        init_mask_slice = args.init_mask_slice if use_init_mask else None

        # forward
        masks_fwd = run_video_along_plane_with_multislice_prompts(
            predictor=predictor,
            plane="xz",
            seg_dir=seg_dir,
            paths=paths,
            selections=selections,
            init_mask_path=init_mask_path,
            init_mask_slice=init_mask_slice,
            seed_frame_hw=(Zds, Wd),
            downsample=ds,
            direction="fwd",
        )
        y_idx_fwd = paths.get("_xz_fwd_y_indices", [])
        accumulate_xz_votes(votes_mm, masks_fwd, y_idx_fwd, ds)

        # backward
        if do_backward:
            masks_bwd = run_video_along_plane_with_multislice_prompts(
                predictor=predictor,
                plane="xz",
                seg_dir=seg_dir,
                paths=paths,
                selections=selections,
                init_mask_path=init_mask_path,
                init_mask_slice=init_mask_slice,
                seed_frame_hw=(Zds, Wd),
                downsample=ds,
                direction="bwd",
            )
            y_idx_bwd = paths.get("_xz_bwd_y_indices", [])
            accumulate_xz_votes(votes_mm, masks_bwd, y_idx_bwd, ds)

    # ----- Selection plane first -----
    if args.roi_plane == "xy":
        _run_xy_plane(selections_by_plane["xy"])
    elif args.roi_plane == "yz":
        _run_yz_plane(selections_by_plane["yz"], use_init_mask=True)
    else:  # 'xz'
        _run_xz_plane(selections_by_plane["xz"], use_init_mask=True)

    # ----- Orthogonal planes (if requested) -----
    if "yz" in orth_planes_for_infer:
        _run_yz_plane(selections_by_plane["yz"], use_init_mask=False)

    if "xz" in orth_planes_for_infer:
        _run_xz_plane(selections_by_plane["xz"], use_init_mask=False)

    # ---------------- Threshold votes -> final masks ----------------
    def threshold_from_mode(mode: str, planes_used: list[str]) -> int:
        if mode == "union":
            return 1
        if mode == "intersect":
            return len(planes_used)
        if mode == "majority":
            return (len(planes_used) // 2) + 1
        return 1

    planes_used = [args.roi_plane] + orth_planes_for_infer
    thr = threshold_from_mode(args.merge_orth, planes_used)

    print("Saving masks…")
    sampled = range(0, Z, ds)  # slices that actually had potential votes
    with alive_bar(len(list(sampled))) as bar:
        for z in sampled:
            sl = (votes_mm[z] >= thr).astype(np.uint8)

            if args.fill_holes:
                sl = fill_2d_holes(sl[np.newaxis, ...])[0]

            if args.restore_original_shape and ds > 1:
                out = cv2.resize(sl, (W, H), interpolation=cv2.INTER_NEAREST) * 255
            else:
                out = medfilt2d(sl) * 255

            cv2.imwrite(str((masks_dir / f"mask_{z:04d}.png")), out.astype(np.uint8))
            bar()

    del votes_mm  # flush memmap

    if tmp_mmap and (not args.keep_memmap):
        _delete_if_exists(tmp_mmap)

    print("Done:", masks_dir)


if __name__ == "__main__":
    main()
