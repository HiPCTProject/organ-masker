# organ-masker

[![DOI](https://zenodo.org/badge/997322638.svg)](https://doi.org/10.5281/zenodo.16967994)

`organ-masker` is a command-line tool to segment organ/background from large volumetric image stacks using **Segment Anything Model 2 (SAM2)**.  
It is optimized for efficiency and minimal memory footprint with streaming I/O, optional memory-mapped intermediates for orthogonal planes, and multiple regions of interest (ROI) types.

Given a directory of 2D TIFF/JP2 slices (or a HiP-CT dataset via `hoa_tools`), it:

1. **Samples** intensities to compute robust percentile normalization without loading the full stack.  
2. **Streams** slices to write XY videos for SAM2 video propagation (optionally YZ/XZ).  
3. **Collects** user prompts (box /points / circle / freehand draw with adjustable brush) on one middle slice or on specified slice for ROI selection.  
4. **Propagates** masks with SAM2 across the requested plane(s).  
5. **Saves** one binary mask per slice under `masks/mask_0000.png`, etc.  
6. **Records** ROI metadata (`roi/roi_<plane>_<slice>.json` and optional `_pos/_neg.png`) for full reproducibility and reuse across runs and resolutions.

> **GPU Required:** SAM2 needs a CUDA-enabled NVIDIA GPU (CUDA ≥ 11.1).  

---

## Installation

1. **Clone and enter the repo**
   ```bash
   git clone https://github.com/JosephBrunet/organ-masker.git
   cd organ-masker
   ```

2. **Create a Python 3.11+ virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   ```

   On Linux you may need Tkinter for OpenCV GUI:
   ```bash
   sudo apt install python3-tk
   ```

3. **Download SAM2 checkpoints**
   ```
   organ-masker/
   ├── checkpoints/
   │   ├── sam2.1_hiera_tiny.pt
   │   ├── sam2.1_hiera_small.pt
   │   ├── sam2.1_hiera_base.pt
   │   └── sam2.1_hiera_large.pt
   ```
   You may also define an environment variable:
   ```bash
   export SAM2_CHECKPOINT_DIR=/path/to/checkpoints
   ```
   The code auto-detects the correct `.pt` checkpoint based on `--model`.

4. **Install dependencies**
   ```bash
   pip install --upgrade pip
   pip install .
   ```

5. **Check installation**
   ```bash
   organ-masker -h
   ```

---

## Usage

```bash
organ-masker /path/to/image_folder --output /path/to/results
```

- **Input:** Folder containing `.tif`, `.tiff`, or `.jp2` slices (sorted alphanumerically).  
  Alternatively, use `--hoatools --datasetname NAME` for public HiP-CT datasets (private datasets can be accessed internally).

- **Output directory:**
  ```
  /path/to/results/segmentation_SAM2_<dataset_or_folder_name>/
      ├── video_xy_forward.mp4|avi
      ├── video_xy_backward.mp4|avi
      ├── video_yz.mp4|avi
      ├── video_xz.mp4|avi
      ├── masks/
      │   ├── mask_0000.png
      │   ├── mask_0001.png
      │   └── ...
      └── roi/
          ├── roi_xy_000512.json
          ├── roi_xy_000512_pos.png
          └── roi_xy_000512_neg.png
  ```

### Interactive ROI Selection
A viewer opens with the target slice(s) and overlays a green contour for ROI selection.  
Press **ENTER** to confirm, **ESC** to cancel, **r** to reset, **[ / ]** or **- / +** to adjust brush size (if in freehand drawing mode), **c** to clear.  
Contours remain unfilled to preserve visibility of underlying details.

<p align="center">
  <img src="assets/images/sam2_window.png" alt="SAM2 ROI Selection Window" width="60%" />
</p>

The script will then convert to video and segment the whole volume.

<p align="center">
  <img src="assets/images/sam2_result.png" alt="SAM2 Segmentation Result" width="60%" />
</p>

---

## Examples

### Basic single ROI on downsampled data
```bash
organ-masker E:\data\Kidney_1\raw_slices --output E:\results\Kidney_1 --model small --downsample 2 --roi-mode box 
```

### Basic single ROI with manual interactive intensity window-levelling  
```bash
organ-masker E:\data\Kidney_1\raw_slices --output E:\results\Kidney_1 --model small --roi-mode box --manual-intensityrescale
```

### Freehand drawing with adjustable brush
```bash
organ-masker E:\data\Kidney_1\raw_slices --output E:\results\Kidney_1_draw --roi-mode draw --roi-slice 500 --downsample 2
```

### Point-based selection (FG/BG clicks)
```bash
organ-masker E:\data\Kidney_1\raw_slices --output E:\results\Kidney_1_points --roi-mode points --roi-slice 400
```

### Circle ROI
```bash
organ-masker E:\data\Kidney_1\raw_slices --output E:\results\Kidney_1_circle --roi-mode circle --roi-slice 256
```

### Existing mask as input (single TIFF)
```bash
organ-masker E:\data\Kidney_1\raw_slices --output E:\results\Kidney_1_init --init-mask E:\seeds\mask_0450.tif --init-mask-slice 450 --roi-mode box
```

### Reuse previously saved ROIs
```bash
organ-masker E:\data\Kidney_1\raw_slices --output E:\results\Kidney_1_reuse --reuse-roi E:\results\Kidney_1_prev\segmentation_SAM2_Kidney_1\roi --model small
```

### HiP-CT dataset with HOA downsample level
```bash
organ-masker --hoatools --datasetname K292_kidney_complete-organ_10.22um_bm18 --privatemetadatapath E:/thierry/private-hoa-metadata/metadata1 --hoa-downsample-level 1 --output E:\results\Kidney_1_hoa --roi-mode box 
```

### Orthogonal predictions (YZ and XZ) with majority merging
```bash
organ-masker E:\data\Kidney_1\raw_slices --output E:\results\Kidney_1_orth --orthogonal --merge-orth majority --roi-mode box --roi-slice 512
```

### Specific orthogonal plane (XZ only) with union merging
```bash
organ-masker E:\data\Kidney_1\raw_slices --output E:\results\Kidney_1_xz --orthogonal-planes xz --merge-orth union --roi-mode box --roi-slice 400 
```

### Automated reuse without GUI
```bash
organ-masker E:\data\Kidney_1\raw_slices --output E:\results\Kidney_1_auto --reuse-roi E:\results\Kidney_1_prev\segmentation_SAM2_Kidney_1\roi --orthogonal-planes yz 
```



---

## Notes

- The pipeline avoids loading the full volume for XY by streaming slices directly to video.  
- Orthogonal planes (YZ/XZ) are generated by streaming or using a lightweight temporary memory-mapped array.  
- Reused ROIs are automatically rescaled to match new downsample or HOA settings.  
- `--init-mask` accepts multiple formats (`.png`, `.tif`, `.jpg`, `.bmp`) and detects slice index automatically when possible.  
- All ROI inputs and metadata are saved for reproducibility.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
