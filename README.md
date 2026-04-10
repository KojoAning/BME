# BME 1433 — Automated Karyotype Analysis

A fully automated pipeline for chromosome segmentation, morphometric feature extraction, Denver group classification, and statistical comparison of normal vs. abnormal karyotypes.

---

## What You Need

### Python
Python 3.9 or later. If you don't have it, download it from [python.org](https://python.org).

### Packages
Install everything at once by running this in your terminal:

```bash
pip install numpy pandas matplotlib scikit-image scipy scikit-learn napari
```

| Package | Version used |
|---|---|
| numpy | 2.1.3 |
| pandas | 2.2.3 |
| matplotlib | 3.10.0 |
| scikit-image | 0.25.0 |
| scipy | 1.17.0 |
| scikit-learn | 1.6.1 |
| napari | 0.7.0 |

> napari is only needed for the semi-automated seed editing section (Section 3, commented out by default). You can skip installing it if you don't plan to use that feature.



## Folder Structure

```
BME-1433/                    # this repo
├── bme.ipynb                # main notebook
├── algorithms.py            # segmentation functions
├── helper_functions.py      # feature extraction, classification, plotting
├── Overlap/                 # metaphase spreads for overlap bias analysis (add manually)
│   └── *.jpg
└── gt_masks/                # (optional) ground truth masks for DICE validation
    └── <imagename>_mask.png
```

The main dataset (`normal/`, `abnormal/`) is hosted on Zenodo (DOI: [10.5281/zenodo.19490325](https://doi.org/10.5281/zenodo.19490325)) and is downloaded automatically when you run the full notebook.


If you want the Zenodo files extracted somewhere other than the default location, change `EXTRACT_DIR` in the first code cell:

```python
EXTRACT_DIR = Path('/your/preferred/path')
```

Everything else in the notebook picks that path up automatically.

---

## Recommended Setup

Clone or download this repo into a **dedicated empty folder** before running anything. When the notebook runs, it downloads and extracts the dataset (`normal/`, `abnormal/`, `gt_masks/`) directly into whichever directory `EXTRACT_DIR` points to. If that directory is your Downloads folder or Desktop, the dataset folders will be scattered among unrelated files. Keeping everything inside one project folder makes paths cleaner and avoids that.

A clean setup looks like this:

```
my-project/
├── bme.ipynb
├── algorithms.py
├── helper_functions.py
├── Overlap/          ← add this manually
├── normal/           ← extracted automatically
├── abnormal/         ← extracted automatically
└── gt_masks/         ← extracted automatically
```

To point the notebook at this folder, set `EXTRACT_DIR` in the first code cell to the project folder:

```python
EXTRACT_DIR = Path('.')   # current directory — works if Jupyter is launched from the project folder
```

---

## How to Run

1. Open a terminal and navigate to the project folder:
   ```bash
   cd /path/to/BME-1433
   ```

2. Launch Jupyter:
   ```bash
   jupyter notebook
   ```
   Then open `bme.ipynb` in your browser. Or just open the folder in VS Code and click on `bme.ipynb`.


## Optional: DICE / IoU Validation

Ground truth masks are included in the Zenodo download. They will be extracted automatically alongside the images. The validation cell looks for `*_mask.png` files in `gt_masks/` and saves results to `gt_masks/dice_validation_results.csv`.



---

## Common Issues

**`ModuleNotFoundError`** — you're missing a package. Run `pip install <package-name>`.

**`FileNotFoundError` on image paths** — update `EXTRACT_DIR` in the Data Loading cell to point to where your images are. For the overlap bias cell, make sure an `Overlap/` folder containing `.jpg` images exists next to the notebook (or update `OVERLAP_DIR` to the correct path).

**Cells running out of order** — always run from top to bottom. If you restart the kernel, re-run all cells from the beginning (Kernel → Restart & Run All).

**Shape mismatch in DICE validation** — the ground truth mask was saved at a different resolution than the original image. Re-export the mask at the same size as the source image.
