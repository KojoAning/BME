from pathlib import Path
from skimage import exposure, io
from skimage import img_as_float
import matplotlib
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import skimage 
from skimage.filters import threshold_otsu
from scipy.signal import savgol_filter, find_peaks
from scipy.ndimage import binary_fill_holes
matplotlib.rcParams['font.size'] = 8
from algorithms import *


_DENVER_META   = 0.38   # CI >= this → metacentric
_DENVER_SUBMETA = 0.25  # CI >= this → submetacentric  (else acrocentric)
_DIPLOID_N = 46 # Expected number of chromosomes in a diploid human cell

def display_image_histogram(img_input, bins=256, title=None,
                            mask_background=True, log_scale=False):
    """
    Display an image alongside its intensity histogram and CDF,
    with key statistics annotated.

    Parameters
    ----------
    img_input        : str, Path, or 2D array
    bins             : int   — histogram bins (default 256)
    title            : str   — custom title (default: filename stem or "Image")
    mask_background  : bool  — exclude background pixels via Otsu before
                               computing histogram/stats (default True).
                               Useful for chromosome crops where the white
                               background dominates the histogram.
    log_scale        : bool  — log y-axis on histogram (default False)

    Returns
    -------
    stats : dict — mean, std, median, p2, p98, dynamic_range
    """
    if isinstance(img_input, (str, Path)):
        img = skimage.io.imread(img_input, as_gray=True)
        label = Path(img_input).stem if title is None else title
    else:
        img = np.asarray(img_input, dtype=float)
        label = title or "Image"

    img = np.clip(img, 0, 1)


    if mask_background:
        thresh  = threshold_otsu(img)
        fg_mask = img < thresh         
        pixels  = img[fg_mask]
        mask_note = f"foreground only (Otsu < {thresh:.2f})"
    else:
        pixels    = img.ravel()
        mask_note = "all pixels"

    if pixels.size == 0:           
        pixels    = img.ravel()
        mask_note = "all pixels (fallback)"

    mean   = pixels.mean()
    std    = pixels.std()
    median = np.median(pixels)
    p2, p98 = np.percentile(pixels, (2, 98))

    stats = dict(mean=mean, std=std, median=median,
                 p2=p2, p98=p98, dynamic_range=p98 - p2)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4),
                             gridspec_kw={"width_ratios": [1.2, 1.5, 1]})


    axes[0].imshow(img, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(label, fontsize=9)
    axes[0].axis("off")

   
    axes[1].hist(pixels, bins=bins, color="steelblue",
                 density=True, linewidth=0)
    axes[1].axvline(mean,   color="red",    lw=1.2, linestyle="-",
                    label=f"mean   {mean:.3f}")
    axes[1].axvline(median, color="orange", lw=1.2, linestyle="--",
                    label=f"median {median:.3f}")
    axes[1].axvline(p2,     color="gray",   lw=1.0, linestyle=":",
                    label=f"p2     {p2:.3f}")
    axes[1].axvline(p98,    color="gray",   lw=1.0, linestyle=":",
                    label=f"p98    {p98:.3f}")
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("Pixel intensity", fontsize=9)
    axes[1].set_ylabel("Density",         fontsize=9)
    axes[1].set_title(f"Histogram ({mask_note})", fontsize=8)
    axes[1].legend(fontsize=7, loc="upper left")
    if log_scale:
        axes[1].set_yscale("log")


    sorted_px  = np.sort(pixels)
    cdf_vals   = np.arange(1, len(sorted_px) + 1) / len(sorted_px)
    axes[2].plot(sorted_px, cdf_vals, color="steelblue", lw=1.5)
    axes[2].axhline(0.02, color="gray", lw=0.8, linestyle=":")
    axes[2].axhline(0.98, color="gray", lw=0.8, linestyle=":")
    axes[2].set_xlim(0, 1)
    axes[2].set_ylim(0, 1)
    axes[2].set_xlabel("Intensity", fontsize=9)
    axes[2].set_ylabel("CDF",       fontsize=9)
    axes[2].set_title("CDF",        fontsize=9)

    fig.suptitle(
        f"std={std:.3f}   dynamic range (p2–p98)={p98 - p2:.3f}",
        fontsize=9, y=1.01
    )
    plt.tight_layout()
    plt.show()

    return stats

def plot_img_and_hist(image, bins=256):
    """Plot an image along with its histogram and cumulative histogram."""
    if isinstance(image, (str, Path)):
        image = io.imread(image)
    image = img_as_float(image)
    fig, (ax_img, ax_hist) = plt.subplots(1, 2)
    ax_cdf = ax_hist.twinx()

    ax_img.imshow(image, cmap=plt.cm.gray)
    ax_img.set_axis_off()

    ax_hist.hist(image.ravel(), bins=bins, histtype='step', color='black')
    ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax_hist.set_xlabel('Pixel intensity')
    ax_hist.set_xlim(0, 1)
    ax_hist.set_yticks([])

    img_cdf, bins = exposure.cumulative_distribution(image, bins)
    ax_cdf.plot(bins, img_cdf, 'r')
    ax_cdf.set_yticks([])

    return fig, ax_img, ax_hist, ax_cdf

def compute_image_quality_metrics(image_path):

    img = skimage.io.imread(image_path)

    if img.ndim == 3:
        gray = skimage.color.rgb2gray(img)
    else:
        gray = img / 255.0

    metrics = {}

    metrics["contrast"] = np.std(gray)

    laplacian = skimage.filters.laplace(gray)
    metrics["sharpness"] = laplacian.var()

    mean_signal = np.mean(gray)
    noise = np.std(gray)

    metrics["SNR"] = mean_signal / noise if noise != 0 else 0

    block_size = (32,32)

    h, w = gray.shape
    h = h - (h % 32)
    w = w - (w % 32)

    cropped = gray[:h, :w]

    blocks = skimage.util.view_as_blocks(cropped, block_size)

    block_means = blocks.mean(axis=(2,3))
    metrics["background_uniformity"] = np.std(block_means)

    hist, _ = np.histogram(gray, bins=256, range=(0,1))
    metrics["histogram_spread"] = np.std(hist)

    windows = skimage.util.view_as_windows(gray, (15,15))
    local_std_map = np.std(windows, axis=(2,3))

    metrics["local_std_variation"] = np.std(local_std_map)

    return metrics

def extract_intensity_profile(chrom_img):
    """
    Compute a 1D mean intensity profile along the long axis of an aligned chromosome.
    The image is inverted internally so dark bands appear as peaks.

    Parameters
    ----------
    chrom_img : 2D array — grayscale crop, long axis along rows (already rotated)

    Returns
    -------
    profile : 1D array, length == chrom_img.shape[0]
    """
    p2, p98 = np.percentile(chrom_img, (2, 98))
    chrom_img = exposure.rescale_intensity(chrom_img, in_range=(p2, p98))
    # Mask out the white background so only chromosome pixels contribute
    thresh     = threshold_otsu(chrom_img)
    chrom_mask = chrom_img < thresh          # chromosome pixels are darker

    inv     = 1.0 - chrom_img               # invert: dark bands → high values
    profile = np.zeros(chrom_img.shape[0])

    for r in range(chrom_img.shape[0]):
        cols = chrom_mask[r]
        if cols.sum() > 0:
            profile[r] = inv[r, cols].mean()

    return profile

def detect_bands(profile, window=11, poly=3, prominence=0.02, min_width=2):
    """
    Smooth a 1D intensity profile and detect dark band peaks.

    Parameters
    ----------
    profile     : 1D array from extract_intensity_profile
    window      : int  — Savitzky-Golay window length (must be odd, default 11)
    poly        : int  — Savitzky-Golay polynomial order (default 3)
    prominence  : float — minimum peak prominence to count as a band (default 0.02)
    min_width   : int  — minimum peak width in pixels (default 2)

    Returns
    -------
    smoothed : 1D array — smoothed profile
    peaks    : int array — row indices of detected band centres
    props    : dict — find_peaks properties (widths, prominences, etc.)
    """
    smoothed        = savgol_filter(profile, window_length=window, polyorder=poly)
    peaks, props    = find_peaks(smoothed, prominence=prominence, width=min_width)
    return smoothed, peaks, props

def find_centromere(chrom_img):
    """
    Estimate the centromere position as the row of minimum chromosome width,
    searching only in the middle 50% of the chromosome to avoid the tips.

    Parameters
    ----------
    chrom_img : 2D array — grayscale crop (long axis along rows)

    Returns
    -------
    centromere_row : int
    """
    thresh       = threshold_otsu(chrom_img)
    chrom_mask = binary_fill_holes(chrom_img < thresh)
    widths       = chrom_mask.sum(axis=1).astype(float)

    # Smooth the width profile to suppress noise
    widths_smooth = savgol_filter(widths, window_length=11, polyorder=3)

    # Restrict search to middle 50%
    lo           = len(widths) // 4
    hi           = 3 * len(widths) // 4
    centromere   = int(np.argmin(widths_smooth[lo:hi]) + lo)

    return centromere

def classify_denver_groups(df,
                           length_col='length',
                           centromere_col='centromere_position',
                           image_col='image_name'):
    """
    Classify detected chromosomes into Denver groups A–G.

    Parameters
    ----------
    df              : pd.DataFrame — one row per detected chromosome, as produced
                      by the notebook's per-chromosome feature extraction loop.
    length_col      : str — column containing chromosome length in pixels
                      (major_axis_length from regionprops).
    centromere_col  : str — column containing the centromere row position within
                      the aligned chromosome crop (from find_centromere).
                      May contain NaN when centromere detection fails.
    image_col       : str — column identifying the source image; used to compute
                      relative lengths within each karyotype spread.

    Returns
    -------
    pd.DataFrame — copy of df with four additional columns:

        relative_length   : chromosome length as % of total detected length in
                            the same image (higher = larger chromosome)
        centromeric_index      : short_arm / total_length, in [0, 0.5]
                                 NaN when centromere_position is NaN
        centromeric_index_area : min(p_arm_pixels, q_arm_pixels) / total_pixels
                                 area-based CI from splitting chrom_img at the
                                 centromere row; captures arm width variation.
                                 NaN when centromere_position or chrom_img is NaN.
        centromere_type   : 'metacentric' | 'submetacentric' | 'acrocentric' | 'unknown'
        arm_ratio         : longer_arm / shorter_arm, in [1, ∞)
                            NaN when centromere_position is NaN
        denver_group      : 'A'–'G', 'D/E' (ambiguous without centromere), or 'unknown'

    Notes
    -----
    - Classification is based on morphometrics alone (size + centromere position).
      Without G-banding, chromosomes within the same Denver group cannot be
      individually identified.
    - The relative-length thresholds assume a diploid spread.  When significantly
      fewer than 46 chromosomes are detected (e.g. due to occlusion), a correction
      factor rescales the relative lengths before applying the rules.
    - Chromosomes that are very small and lack centromere information are assigned
      Group G by default (smallest size bin).

    """
    result = df.copy()


    # centromeric index  (short_arm / total_length)

    def _ci(row):
        cp = row[centromere_col]
        ln = row['aligned_length'] if 'aligned_length' in row.index else row[length_col]
        if pd.isna(cp) or pd.isna(ln) or ln == 0:
            return np.nan
        p = float(cp)
        q = float(ln) - p
        if p <= 0 or q <= 0:
            return np.nan
        short = min(p, q)
        return short / float(ln)

    result['centromeric_index'] = result.apply(_ci, axis=1)

    # area-based centromeric index  CI(A) = min(Ap, Aq) / (Ap + Aq)
    # splits chrom_img at the centromere row and counts nonzero pixelsin each arm, capturing width variation along the chromosome.
    def _ci_area(row):
        img = row['chrom_img']
        cp  = row[centromere_col] #centromere position (row index within chrom_img)
        if pd.isna(cp) or img is None:
            return np.nan
        cp = int(round(float(cp)))
        p_area = int(np.count_nonzero(img[:cp]))
        q_area = int(np.count_nonzero(img[cp:]))
        total  = p_area + q_area
        if total == 0:
            return np.nan
        return min(p_area, q_area) / total

    result['centromeric_index_area'] = result.apply(_ci_area, axis=1)

    # arm ratio  (longer_arm / shorter_arm)
    def _arm_ratio(row):
        cp = row[centromere_col]
        # use aligned_length if available which is in the same coordinate system as centromere_position
        ln = row['aligned_length'] if 'aligned_length' in row.index else row[length_col]
        if pd.isna(cp) or pd.isna(ln) or ln == 0:
            return np.nan
        p = float(cp)
        q = float(ln) - p
        if p <= 0 or q <= 0:
            return np.nan
        return max(p, q) / min(p, q)

    result['arm_ratio'] = result.apply(_arm_ratio, axis=1)

    # relative length within each image (% of total detected length)
    result['relative_length'] = result.groupby(image_col)[length_col].transform(
        lambda x: x / x.sum() * 100
    )

    # correction factor for partial detection
    # We used this because sometimes only k < 46 chromosomes are detected, leading to inflated relative lengths
    # so we scale by 46/k. eg. when k=46, the vcorrection factor would be 1 and relative lengths would be unchanged but
    #  when k<46, correction factor > 1 and relative lengths are scaled down to better fit the expected distribution.

    n_per_image = result.groupby(image_col)[length_col].transform('count')
    correction  = n_per_image / _DIPLOID_N 
    result['_rl_corrected'] = result['relative_length'] * correction

    result['denver_group'] = result.apply(
        lambda r: _denver_rule(r['_rl_corrected'], r['centromeric_index']),
        axis=1
    )
    result.drop(columns=['_rl_corrected'], inplace=True)

    return result


def _denver_rule(rel_len_corrected, ci):
    """
    Map (corrected relative length, centromeric index) → Denver group A–G.

    rel_len_corrected : relative_length scaled as if all 46 chromosomes were detected
    ci                : centromeric_index (NaN allowed — falls back to size only)
    """
    is_meta   = (not pd.isna(ci)) and ci >= _DENVER_META
    is_submeta = (not pd.isna(ci)) and (_DENVER_SUBMETA <= ci < _DENVER_META)
    is_acro   = (not pd.isna(ci)) and ci < _DENVER_SUBMETA
    ci_known  = not pd.isna(ci)

    # Group A: chromosomes 1, 2, 3 — largest, meta/submeta
    if rel_len_corrected > 3.5:
        return 'A'

    # Group B: chromosomes 4, 5 — large, submetacentric
    if rel_len_corrected > 2.8:
        if ci_known:
            return 'B' if (is_submeta or is_acro) else 'A'
        return 'B'

    # Group C: chromosomes 6–12, X — medium, submetacentric
    if rel_len_corrected > 1.8:
        if ci_known:
            if is_acro:
                return 'D'    # acrocentric at this size → Group D
            return 'C'
        return 'C'

    # Groups D and E overlap in size; centromere type is the key discriminator
    if rel_len_corrected > 1.3:
        if ci_known:
            return 'D' if is_acro else 'E'
        return 'D/E'   # ambiguous without centromere info

    # Group F: chromosomes 19, 20 — small, metacentric
    if rel_len_corrected > 1.0:
        if ci_known:
            return 'F' if (is_meta or is_submeta) else 'G'
        return 'F'

    # Group G: chromosomes 21, 22, Y — very small, acrocentric
    return 'G'

def extract_chromosome_features(img_paths, method='adaptive_histogram',
                                 pad=5, plot=False):
    """
    Segment chromosomes from a list of karyotype images and extract
    per-chromosome morphometric features into rhe DataFrame.

    Parameters
    ----------
    img_paths : list of Path — karyotype image file paths to process
    method    : str — segmentation method used; stored in the 'method' column.
                      'adaptive_histogram' | 'contrast_stretching'
    pad       : int — pixel padding added around each chromosome bounding box
    plot      : bool — if True, passes plot=True to the segmentation function

    Returns
    -------
    pd.DataFrame — one row per detected chromosome with columns:
        image_name, chromosome_label, chrom_img, method,
        area, length, width, eccentricity,
        centroid_row, centroid_col, orientation,
        bbox_minr, bbox_minc, bbox_maxr, bbox_maxc,
        number_of_chromosomes, solidity
    """
    records = []

    for img_path in img_paths:
        img_gray = skimage.io.imread(img_path, as_gray=True)
        img_gray = np.clip(img_gray, 0, 1)

        mask    = chromosome_mask_with_adaptive_histogram(img_path, plot=plot) #we selected adaptive thresholding as our final contrast enhancement method
        labeled = label(mask)
        regions = regionprops(labeled)

        for region in regions:
            minr, minc, maxr, maxc = region.bbox
            minr = max(0, minr - pad)
            minc = max(0, minc - pad)
            maxr = min(mask.shape[0], maxr + pad)
            maxc = min(mask.shape[1], maxc + pad)

            chrom_mask = (labeled[minr:maxr, minc:maxc] == region.label)
            img_crop   = img_gray[minr:maxr, minc:maxc]
            chrom_img  = img_crop * chrom_mask

            records.append({
                'image_name':             img_path.name,
                'chromosome_label':       region.label,
                'chrom_img':              chrom_img,
                'method':                 method,
                'area':                   region.area,
                'length':                 region.major_axis_length,
                'width':                  region.minor_axis_length,
                'eccentricity':           region.eccentricity,
                'centroid_row':           region.centroid[0],
                'centroid_col':           region.centroid[1],
                'orientation':            region.orientation,
                'bbox_minr':              region.bbox[0],
                'bbox_minc':              region.bbox[1],
                'bbox_maxr':              region.bbox[2],
                'bbox_maxc':              region.bbox[3],
                'number_of_chromosomes':  labeled.max(),
                'solidity':               region.solidity,
            })

    return pd.DataFrame(records)

def extract_centromere_and_bands(df, plot=False):
    """
    Align each chromosome, detect its centromere position and band count,
    and store the results back into the DataFrame.

    Parameters
    ----------
    df   : pd.DataFrame — one row per chromosome with columns:
               chrom_img, orientation, method, image_name, chromosome_label
    plot : bool — if True, display alignment + intensity profile + arm ratio
                  for each chromosome (default False)

    Returns
    -------
    pd.DataFrame — copy of df with two new columns:
        centromere_position : int row index of centromere in aligned image,
                              or NaN if detection failed
        n_bands             : number of dark bands detected, or NaN on failure
    """
    from helper_functions import find_centromere, extract_intensity_profile, detect_bands

    result               = df.copy()
    centromere_positions = []
    n_bands_list         = []
    aligned_lengths      = []

    for _, row in result.iterrows():
        chrom = row['chrom_img']

        # align — long axis vertical
        angle   = -row['orientation'] * 180 / np.pi
        aligned = skimage.transform.rotate(chrom, angle, resize=True, cval=0,
                                           preserve_range=True)

        # fix background: black (0) → white (1)
        aligned_white = np.where(aligned == 0, 1.0, aligned)

        # apply same contrast enhancement used during segmentation
        if row['method'] == 'adaptive_histogram':
            display_img = exposure.equalize_adapthist(aligned_white, clip_limit=0.03)
        else:
            p2, p98     = np.percentile(aligned_white[aligned_white < 1.0], (2, 98))
            display_img = exposure.rescale_intensity(aligned_white, in_range=(p2, p98))

        # store aligned image height — used as the true chromosome length
        # for arm ratio / centromeric index (avoids mismatch with regionprops length)
        aligned_lengths.append(aligned_white.shape[0])

        # centromere
        try:
            c_pos = find_centromere(aligned_white)
            centromere_positions.append(c_pos)
        except Exception:
            c_pos = np.nan
            centromere_positions.append(np.nan)

        # intensity profile
        try:
            profile             = extract_intensity_profile(aligned_white)
            smoothed, peaks, _  = detect_bands(profile)
            n_bands_list.append(len(peaks))

        except Exception:
            profile, smoothed, peaks = None, None, []
            n_bands_list.append(np.nan)

        if plot:
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))

            axes[0].imshow(display_img, cmap='gray', vmin=0, vmax=1)
            if not np.isnan(c_pos):
                axes[0].axhline(c_pos, color='red', lw=1.5,
                                label=f'centromere row {int(c_pos)}')
                axes[0].legend(fontsize=7)
            axes[0].set_title(
                f"{row['image_name']} | chrom {row['chromosome_label']} | {row['method']}")
            axes[0].axis('off')

            if profile is not None:
                axes[1].plot(profile,  color='steelblue', lw=1,   label='raw')
                axes[1].plot(smoothed, color='orange',    lw=1.5, label='smoothed')
                if not np.isnan(c_pos):
                    axes[1].axvline(c_pos, color='red', lw=1.2, linestyle='--',
                                    label='centromere')
                axes[1].set_title('Intensity profile')
                axes[1].set_xlabel('Row')
                axes[1].legend(fontsize=7)

            if not np.isnan(c_pos):
                total = aligned_white.shape[0]
                p_arm = c_pos
                q_arm = total - c_pos
                ar    = p_arm / q_arm if q_arm > 0 else np.nan
                axes[2].bar(['p arm', 'q arm'], [p_arm, q_arm],
                            color=['steelblue', 'coral'])
                axes[2].set_title(f'Arm ratio p/q = {ar:.2f}')
            else:
                axes[2].set_title('Arm ratio: centromere not detected')

            plt.tight_layout()
            plt.show()

    result['centromere_position'] = centromere_positions
    result['n_bands']             = n_bands_list
    result['aligned_length']      = aligned_lengths
    return result


