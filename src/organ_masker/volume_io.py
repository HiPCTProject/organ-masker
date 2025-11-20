# volume_io.py — consistent downsampling/normalization for streaming + memmap
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm
from skimage.measure import block_reduce
from scipy.ndimage import median_filter


# ------------------ Basic dims helpers ------------------

def get_dims_streaming(im_dir, hoatools, datasetname, priv_meta, hoa_downsample_level=2):
    if hoatools:
        import hoa_tools.dataset as hoa_dataset
        if priv_meta:
            from pathlib import Path as _P
            hoa_dataset.change_metadata_directory(_P(priv_meta))
        ds = hoa_dataset.get_dataset(datasetname)
        da = ds.data_array(downsample_level=hoa_downsample_level)
        Z = da.sizes["z"]
        H, W = da.isel(z=0).values.shape
        return int(Z), int(H), int(W)
    else:
        p = Path(im_dir)
        files = sorted([f for f in p.iterdir() if f.suffix.lower() in [".tif", ".tiff", ".jp2"]])
        if not files:
            raise FileNotFoundError(f"No images found in {p}")
        first = cv2.imread(str(files[0]), -1)
        if first is None:
            raise RuntimeError(f"Could not read {files[0]}")
        H, W = first.shape[:2]
        return len(files), int(H), int(W)


def infer_downsampled_hw_from_first_slice(
    im_dir, hoatools, datasetname, priv_meta, downsample, hoa_downsample_level=2
):
    """
    Return (Hd, Wd) derived from the first slice's original size:
      H0,W0 = first slice size
      Hd = floor(H0/ds); Wd = floor(W0/ds)

    All subsequent slices are force-matched to (H0,W0) before downsampling,
    then cropped to (Hd*ds, Wd*ds), then block_reduce -> (Hd, Wd).
    """
    ds = downsample if downsample else 1
    if hoatools:
        import hoa_tools.dataset as hoa_dataset
        if priv_meta:
            from pathlib import Path as _P
            hoa_dataset.change_metadata_directory(_P(priv_meta))
        d = hoa_dataset.get_dataset(datasetname)
        da = d.data_array(downsample_level=hoa_downsample_level)
        first = da.isel(z=0).values
    else:
        p = Path(im_dir)
        files = sorted([f for f in p.iterdir() if f.suffix.lower() in [".tif", ".tiff", ".jp2"]])
        first = cv2.imread(str(files[0]), -1)

    H0, W0 = first.shape[:2]
    if ds > 1:
        Hd = H0 // ds
        Wd = W0 // ds
    else:
        Hd, Wd = H0, W0
    return int(Hd), int(Wd)


# ------------------ Percentile sampling (multi-slice, efficient) ------------------

def sample_percentiles_streaming(
    im_dir, hoatools, datasetname, priv_meta, Z, window, downsample,
    num_slices=32, pixel_stride=8, hoa_downsample_level=2
):
    """
    Compute global vmin/vmax by streaming a subset of slices. Every sampled
    slice is force-shaped to the first-slice size, then downsampled to the
    target (Hd,Wd) with the same path as used elsewhere.
    """
    ds = downsample if downsample else 1

    # Prepare readers and first slice (to get H0,W0 and target Hd,Wd)
    if hoatools:
        import hoa_tools.dataset as hoa_dataset
        ds_hoa = hoa_dataset.get_dataset(datasetname)
        da = ds_hoa.data_array(downsample_level=hoa_downsample_level)
        reader = lambda i: da.isel(z=i).values
        first = reader(0)
    else:
        p = Path(im_dir)
        files = sorted([f for f in p.iterdir() if f.suffix.lower() in [".tif", ".tiff", ".jp2"]])
        reader = lambda i: cv2.imread(str(files[i]), -1)
        first = reader(0)

    H0, W0 = first.shape[:2]
    Hd = H0 // ds if ds > 1 else H0
    Wd = W0 // ds if ds > 1 else W0
    Hc = Hd * ds
    Wc = Wd * ds

    def _to_consistent(arr):
        if arr.shape[:2] != (H0, W0):
            arr = cv2.resize(arr, (W0, H0), interpolation=cv2.INTER_NEAREST)
        arr = arr.astype(np.float32)
        if ds == 1:
            arr = median_filter(arr, size=2)
            out = arr
        else:
            if (H0 != Hc) or (W0 != Wc):
                arr = arr[:Hc, :Wc]
            out = block_reduce(arr, (ds, ds), np.mean)
        return out

    z_idx = np.linspace(0, Z - 1, num=min(num_slices, Z), dtype=int)
    samples = []
    for zi in tqdm(z_idx, desc="Percentile sampling", unit="slice"):
        s = reader(zi)
        s_ds = _to_consistent(s)
        samples.append(s_ds[::pixel_stride, ::pixel_stride].ravel())

    samples = np.concatenate(samples)
    vmin = float(np.percentile(samples, window[0]).astype(np.float32))
    vmax = float(np.percentile(samples, window[1]).astype(np.float32))
    return vmin, vmax


# ------------------ Core normalization/downsample helpers ------------------

def _downsample_norm_u8_consistent(arr, vmin, vmax, ds, H0, W0, Hd, Wd, no_median=False):
    """
    Force any slice to the exact (Hd,Wd) with the same logic as above.
    1) Resize to (H0,W0) if needed
    2) If ds>1: crop to (Hd*ds, Wd*ds), then block_reduce -> (Hd,Wd)
       If ds==1: optional 2D median
    3) Linear stretch to [0,255] using vmin/vmax
    4) Final guard-resize to (Wd,Hd) if needed
    """
    if arr.shape[:2] != (H0, W0):
        arr = cv2.resize(arr, (W0, H0), interpolation=cv2.INTER_NEAREST)

    arr = arr.astype(np.float32)

    if ds and ds > 1:
        Hc = Hd * ds
        Wc = Wd * ds
        if (H0 != Hc) or (W0 != Wc):
            arr = arr[:Hc, :Wc]
        arr = block_reduce(arr, (ds, ds), np.mean)
    else:
        if not no_median:
            arr = median_filter(arr, size=2)

    scale = max(vmax - vmin, 1e-6)
    out = np.clip((arr - vmin) / scale * 255.0, 0, 255).astype(np.uint8)

    if out.shape[:2] != (Hd, Wd):
        out = cv2.resize(out, (Wd, Hd), interpolation=cv2.INTER_NEAREST)

    return out


# ------------------ Plane frame builder (for ROI screens) ------------------

def get_plane_frame_streaming(
    im_dir, hoatools, datasetname, priv_meta, plane, idx, vmin, vmax, downsample,
    hoa_downsample_level=2, no_median=False
):
    """
    Returns a single plane frame with consistent axes and *no deformation*:
      - XY: (Hd, Wd)
      - YZ: (Hd, Zds)
      - XZ: (Zds, Wd)
    """
    ds = downsample if downsample else 1

    if hoatools:
        import hoa_tools.dataset as hoa_dataset
        ds_hoa = hoa_dataset.get_dataset(datasetname)
        da = ds_hoa.data_array(downsample_level=hoa_downsample_level)
        Z = int(da.sizes["z"])
        read = lambda i: da.isel(z=i).values
        first = read(0)
    else:
        p = Path(im_dir)
        files = sorted([f for f in p.iterdir() if f.suffix.lower() in [".tif", ".tiff", ".jp2"]])
        Z = len(files)
        read = lambda i: cv2.imread(str(files[i]), -1)
        first = read(0)

    H0, W0 = first.shape[:2]
    Hd = H0 // ds if ds > 1 else H0
    Wd = W0 // ds if ds > 1 else W0
    z_steps = list(range(0, Z, ds)) if ds > 1 else list(range(0, Z))
    Zds = len(z_steps)

    if plane == "xy":
        sl = read(idx)
        return _downsample_norm_u8_consistent(sl, vmin, vmax, ds, H0, W0, Hd, Wd, no_median=no_median)

    if plane == "yz":
        x_ds = idx // ds
        sl0 = _downsample_norm_u8_consistent(first, vmin, vmax, ds, H0, W0, Hd, Wd, no_median=no_median)
        x_ds = min(x_ds, Wd - 1)
        fr = np.empty((Hd, Zds), dtype=np.uint8)
        fr[:, 0] = sl0[:, x_ds]
        zi = 1
        for z in z_steps[1:]:
            sl = _downsample_norm_u8_consistent(read(z), vmin, vmax, ds, H0, W0, Hd, Wd, no_median=no_median)
            fr[:, zi] = sl[:, x_ds]
            zi += 1
        return fr

    # plane == "xz"
    y_ds = idx // ds
    sl0 = _downsample_norm_u8_consistent(first, vmin, vmax, ds, H0, W0, Hd, Wd, no_median=no_median)
    y_ds = min(y_ds, Hd - 1)
    fr = np.empty((Zds, Wd), dtype=np.uint8)
    fr[0, :] = sl0[y_ds, :]
    zi = 1
    for z in z_steps[1:]:
        sl = _downsample_norm_u8_consistent(read(z), vmin, vmax, ds, H0, W0, Hd, Wd, no_median=no_median)
        fr[zi, :] = sl[y_ds, :]
        zi += 1
    return fr


# ------------------ Memmap builder (orthogonal writers) ------------------

def build_memmap_normalized_minimal_dims(
    im_dir,
    mmap_path,
    window=(1, 99),
    downsample=1,
    hoatools=False,
    datasetname=None,
    priv_meta=None,
    hoa_downsample_level=2,
    vmin=None,
    vmax=None,
    no_median=False,
    pctl_slices=32,
    pctl_pixel_stride=8,
):
    """
    Create a uint8 memmap (Z,Hd,Wd) with deterministic sizing.
    """
    ds = downsample if downsample else 1

    if hoatools:
        import hoa_tools.dataset as hoa_dataset
        if priv_meta:
            from pathlib import Path as _P
            hoa_dataset.change_metadata_directory(_P(priv_meta))
        d = hoa_dataset.get_dataset(datasetname)
        da = d.data_array(downsample_level=hoa_downsample_level)
        Z = int(da.sizes["z"])
        read = lambda i: da.isel(z=i).values
        first = read(0)
    else:
        p = Path(im_dir)
        files = sorted([f for f in p.iterdir() if f.suffix.lower() in [".tif", ".tiff", ".jp2"]])
        if not files:
            raise FileNotFoundError(f"No images found in {p}")
        Z = len(files)
        read = lambda i: cv2.imread(str(files[i]), -1)
        first = read(0)

    H0, W0 = first.shape[:2]
    Hd = H0 // ds if ds > 1 else H0
    Wd = W0 // ds if ds > 1 else W0

    if vmin is None or vmax is None:
        vmin, vmax = sample_percentiles_streaming(
            im_dir=im_dir,
            hoatools=hoatools,
            datasetname=datasetname,
            priv_meta=priv_meta,
            Z=Z,
            window=tuple(window),
            downsample=ds,
            num_slices=pctl_slices,
            pixel_stride=pctl_pixel_stride,
            hoa_downsample_level=hoa_downsample_level,
        )

    mm = np.memmap(mmap_path, dtype="uint8", mode="w+", shape=(Z, Hd, Wd))
    for z in tqdm(range(Z), desc="Memmap writing", unit="slice"):
        arr = read(z)
        arr_u8 = _downsample_norm_u8_consistent(arr, vmin, vmax, ds, H0, W0, Hd, Wd, no_median=no_median)
        mm[z, :, :] = arr_u8
    del mm

    return (Z, Hd, Wd), (H0, W0), (float(vmin), float(vmax))


# =====================================================================
# Interactive windowing helpers (optional, GUI-based)
# =====================================================================

def qtimage(data, perc_limit=2, max_size=1e7, wait=False, **kwargs):
    """
    Minimal drop-in qtimage using pyqtgraph if available.
    """
    try:
        import numpy as np
        import pyqtgraph as pg
        from PyQt5.QtWidgets import QApplication
    except Exception:
        print("[windowing] pyqtgraph/PyQt5 not available, skipping GUI display.")
        return None

    import numpy as np

    if data.ndim > 3:
        if data.ndim != 4 and data.shape[-1] != 3:
            print("Can't display arrays with more than 3 dimensions unless rgb!")
            return None
        else:
            data_to_show = data.swapaxes(-2, -3)
    else:
        data_to_show = data.swapaxes(-1, -2)

    if np.isinf(data_to_show).any():
        data_to_show[np.isinf(data_to_show)] = 0
    if np.isnan(data_to_show).any():
        data_to_show[np.isnan(data_to_show)] = 0

    if data_to_show.size > max_size:
        ratio = float(data_to_show.size) / max_size
        sl = np.s_[::int(round(ratio)) + 1]
    else:
        sl = np.s_[:]

    lower = np.percentile(data_to_show[sl], perc_limit)
    upper = np.percentile(data_to_show[sl], 100 - perc_limit)

    win = pg.image(data_to_show, **kwargs)
    win.setLevels(lower, upper)

    if wait:
        def close_event(event):
            loop.quit()
            event.accept()
        loop = pg.Qt.QtCore.QEventLoop()
        win.closeEvent = close_event
        loop.exec_()

    return win


def qtcompare(image1, image2=None, align="middle", wait=False, **kwargs):
    """
    Compare one or two images side by side (padded) using qtimage.
    """
    import numpy as np

    data = []
    for img in (image1, image2):
        if img is None:
            continue
        if isinstance(img, np.ndarray):
            if img.ndim == 2:
                img = img[None, ...]
            data.append(img)
        elif isinstance(img, list):
            data.extend([i[None, ...] for i in img])

    if not data:
        return qtimage(np.zeros((1, 16, 16), np.uint8))

    for axis in [1, 2]:
        widths = [im.shape[axis] for im in data]
        max_w = max(widths)
        for i in range(len(data)):
            dwidth = max_w - widths[i]
            pad_widths = np.zeros((3, 2), dtype=int)
            if align == "middle":
                pad_widths[axis] = [dwidth // 2, dwidth - dwidth // 2]
            elif align == "top_left":
                pad_widths[axis] = [0, dwidth]
            else:
                pad_widths[axis] = [dwidth, 0]
            data[i] = np.pad(
                data[i],
                pad_width=pad_widths,
                mode="constant",
                constant_values=np.mean(data[i]),
            )
    data = np.concatenate(data, axis=0)
    return qtimage(data, wait=wait, **kwargs)


def get_windowing(images):
    """
    Console+GUI hybrid: ENTER -> Qt widget, else 'vmin,vmax'.
    """
    inp = input(
        "\nDo you want to select windowing with a selection widget or provide values?\n"
        "Hit ENTER for widget, or provide vmin, vmax otherwise. "
    ).strip()

    if inp == "":
        try:
            from PyQt5.QtWidgets import QApplication
            import sys
            app = QApplication.instance()
            if app is None:
                app = QApplication(sys.argv)
            imview = qtcompare(images, title="Select windowing")  # type: ignore
            print("Use histogram sliders to select windowing. Use arrow keys to browse.")
            _ = input("Hit ENTER when ready.")
            img_vmin, img_vmax = imview.getLevels()
            imview.close()
        except Exception as e:
            print(f"[windowing] GUI not available ({e}), fallback to manual.")
            img_vmin = float(input("vmin: "))
            img_vmax = float(input("vmax: "))
    else:
        parts = inp.split(",")
        img_vmin = float(parts[0])
        img_vmax = float(parts[1])

    print(f"Windowing: vmin {img_vmin:.3f}, vmax {img_vmax:.3f}")
    return img_vmin, img_vmax


def choose_windowing_streaming(
    im_dir,
    hoatools,
    datasetname,
    priv_meta,
    Z,
    downsample,
    hoa_downsample_level=2,
    num_slices=16,
    center_slice=None,
):
    """
    Load slices (same policy as percentile sampling), run them through the same
    downsample path, then call get_windowing(...) so user can pick vmin/vmax.

    If center_slice is provided, we only show that single slice to the user
    (after applying the same downsample/median logic). Otherwise, we sample
    multiple slices across Z.
    """
    ds = downsample if downsample else 1

    if hoatools:
        import hoa_tools.dataset as hoa_dataset
        if priv_meta:
            from pathlib import Path as _P
            hoa_dataset.change_metadata_directory(_P(priv_meta))
        d = hoa_dataset.get_dataset(datasetname)
        da = d.data_array(downsample_level=hoa_downsample_level)
        reader = lambda i: da.isel(z=i).values
        first = reader(0)
    else:
        p = Path(im_dir)
        files = sorted([f for f in p.iterdir() if f.suffix.lower() in [".tif", ".tiff", ".jp2"]])
        reader = lambda i: cv2.imread(str(files[i]), -1)
        first = reader(0)

    H0, W0 = first.shape[:2]
    Hd = H0 // ds if ds > 1 else H0
    Wd = W0 // ds if ds > 1 else W0

    # Decide which z-indices to use for the GUI
    if center_slice is not None:
        sli = max(0, min(Z - 1, int(center_slice)))
        z_idx = np.array([sli], dtype=int)
    else:
        z_idx = np.linspace(0, Z - 1, num=min(num_slices, Z), dtype=int)

    imgs = []
    for zi in z_idx:
        arr = reader(int(zi)).astype(np.float32)
        if ds > 1:
            Hc = Hd * ds
            Wc = Wd * ds
            if (H0 != Hc) or (W0 != Wc):
                arr = arr[:Hc, :Wc]
            arr = block_reduce(arr, (ds, ds), np.mean)
        else:
            arr = median_filter(arr, size=2)
        imgs.append(arr.astype(np.float32))

    vmin, vmax = get_windowing(imgs)
    return float(vmin), float(vmax)
