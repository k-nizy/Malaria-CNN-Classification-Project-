"""
common/data.py — shared data pipeline for the Malaria CNN Classification project.

Every group member imports from here so that:
  - the SAME train/val/test split is used across all three models (fair comparison),
  - label mapping is identical everywhere: 1 = Parasitized (positive/infected),
    0 = Uninfected, so recall == sensitivity and specificity == TNR on uninfected,
  - the ~700 MB NIH zip is downloaded ONCE and cached in Google Drive, instead of
    re-downloaded on every Colab session like the starter notebook does.

Works in two environments:
  - Colab: Drive cache enabled (IS_COLAB=True), NIH download on first run.
  - Local/offline tests: `make_synthetic_cell_dataset()` builds a tiny fake
    dataset with the same folder layout, so we can verify the pipeline without
    downloading anything.
"""

from __future__ import annotations

import os
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf

# ---------------------------------------------------------------------------
# Constants — single source of truth for the whole group.
# ---------------------------------------------------------------------------

SEED = 42
IMG_SIZE = (128, 128)          # revisit after EDA; DenseNet experiment may use 224
BATCH_SIZE = 32
LABEL_MAP = {"Parasitized": 1, "Uninfected": 0}   # 1 = infected = positive class
CLASS_NAMES = ["Parasitized", "Uninfected"]

DATA_URL = "https://data.lhncbc.nlm.nih.gov/public/Malaria/cell_images.zip"
ZIP_NAME = "cell_images.zip"

# Expected counts in the real NIH dataset (used as a soft sanity check).
EXPECTED_TOTAL = 27_558
EXPECTED_PER_CLASS = 13_779


@dataclass
class DataDirs:
    """Resolved locations for the dataset and its cache."""
    data_root: Path            # folder that CONTAINS cell_images/
    zip_path: Path             # where the zip lives (or will be downloaded to)
    cell_images: Path          # the extracted cell_images/ folder
    cache_dir: Path | None     # persistent cache (Drive on Colab), else None

    def describe(self) -> str:
        return (
            f"data_root={self.data_root}\n"
            f"zip={self.zip_path}\n"
            f"cell_images={self.cell_images}\n"
            f"cache={self.cache_dir}"
        )


# ---------------------------------------------------------------------------
# Environment detection & directory resolution
# ---------------------------------------------------------------------------

def is_colab() -> bool:
    """True when running inside Google Colab."""
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def resolve_dirs(drive_data_dir: str = "/content/drive/MyDrive/malaria_project/data",
                 local_data_dir: str = "data") -> DataDirs:
    """
    Resolve where the dataset should live.

    Colab:  cache the zip under <drive_data_dir> so future sessions skip the
            download; extract to <local_data_dir> for fast local reads.
    Local:  everything under ./data (no cache needed; disk is persistent).
    """
    if is_colab():
        from google.colab import drive  # type: ignore
        if not Path("/content/drive/MyDrive").exists():
            drive.mount("/content/drive")
        cache_dir = Path(drive_data_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        local_root = Path(local_data_dir)
    else:
        cache_dir = None
        local_root = Path(local_data_dir)

    local_root.mkdir(parents=True, exist_ok=True)
    zip_path = local_root / ZIP_NAME
    return DataDirs(
        data_root=local_root,
        zip_path=zip_path,
        cell_images=local_root / "cell_images",
        cache_dir=cache_dir,
    )


# ---------------------------------------------------------------------------
# Download + extract (with Drive caching)
# ---------------------------------------------------------------------------

def download_cell_images(dirs: DataDirs, force: bool = False) -> Path:
    """
    Ensure cell_images.zip is available locally, using the Drive cache when on
    Colab. Returns the path to the zip.

    Strategy:
      1. zip already local            -> reuse
      2. zip cached on Drive          -> copy down (fast, no re-download)
      3. nothing anywhere             -> download from NIH, then cache to Drive
    """
    if dirs.zip_path.exists() and not force:
        return dirs.zip_path

    if dirs.cache_dir is not None:
        cached = dirs.cache_dir / ZIP_NAME
        if cached.exists() and not force:
            shutil.copy2(cached, dirs.zip_path)
            return dirs.zip_path

    print(f"Downloading {DATA_URL} ...")
    tmp = dirs.zip_path.with_suffix(".part")
    urllib.request.urlretrieve(DATA_URL, tmp)
    os.replace(tmp, dirs.zip_path)

    if dirs.cache_dir is not None:
        cached = dirs.cache_dir / ZIP_NAME
        if not cached.exists():
            shutil.copy2(dirs.zip_path, cached)
    return dirs.zip_path


def extract_cell_images(dirs: DataDirs, force: bool = False) -> Path:
    """Extract the zip into dirs.cell_images if not already extracted."""
    marker = dirs.cell_images / ".extracted"
    if marker.exists() and not force:
        return dirs.cell_images

    if dirs.cell_images.exists():
        shutil.rmtree(dirs.cell_images)
    dirs.cell_images.mkdir(parents=True)

    with zipfile.ZipFile(dirs.zip_path) as zf:
        zf.extractall(dirs.data_root)

    marker.touch()
    return dirs.cell_images


def count_images(cell_images: Path) -> dict[str, int]:
    """Count images per class folder (any case, common extensions)."""
    exts = {".png", ".jpg", ".jpeg"}
    counts = {}
    for cls in CLASS_NAMES:
        folder = cell_images / cls
        counts[cls] = sum(
            1 for p in folder.rglob("*") if p.suffix.lower() in exts
        ) if folder.exists() else 0
    return counts


def get_data(local_data_dir: str = "data",
             drive_data_dir: str = "/content/drive/MyDrive/malaria_project/data",
             force_download: bool = False) -> Path:
    """One-call setup used at the top of every member notebook."""
    dirs = resolve_dirs(drive_data_dir=drive_data_dir,
                        local_data_dir=local_data_dir)
    download_cell_images(dirs, force=force_download)
    extract_cell_images(dirs)
    counts = count_images(dirs.cell_images)
    total = sum(counts.values())
    print(f"Dataset ready: {counts} (total {total})")
    if total != EXPECTED_TOTAL:
        print(f"  [warn] expected {EXPECTED_TOTAL} images — continuing anyway.")
    return dirs.cell_images


# ---------------------------------------------------------------------------
# Synthetic dataset (offline testing & pipeline debugging)
# ---------------------------------------------------------------------------

def make_synthetic_cell_dataset(root: str | Path,
                                per_class: int = 40,
                                img_size: tuple[int, int] = (32, 32),
                                seed: int = SEED) -> Path:
    """
    Build a tiny fake dataset with the SAME folder layout as the real one:
        root/Parasitized/*.png , root/Uninfected/*.png
    Parasitized gets a dark 'stain' blob so the classes are visually separable
    (enough for pipeline smoke tests and Grad-CAM sanity checks).
    """
    rng = np.random.default_rng(seed)
    root = Path(root)
    h, w = img_size
    margin = max(6, min(h, w) // 4)          # keep blobs away from borders at any size
    for cls, label in LABEL_MAP.items():
        folder = root / cls
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(per_class):
            # tight background variance -> the blob is the dominant learnable signal
            img = rng.integers(150, 190, size=(h, w, 3), dtype=np.uint8)
            if label == 1:  # Parasitized: add a dark puncture-like blob
                cy = int(rng.integers(margin, h - margin))
                cx = int(rng.integers(margin, w - margin))
                rr = int(max(3, min(h, w) // 8))
                yy, xx = np.mgrid[:h, :w]
                img[(yy - cy) ** 2 + (xx - cx) ** 2 <= rr ** 2] = (30, 30, 60)
            # scale=False: img is already uint8 in [0,255]; scale=True would
            # min-max normalize per image and distort the synthetic signal
            tf.keras.utils.save_img(str(folder / f"{cls.lower()}_{i:03d}.png"), img,
                                    scale=False)
    return root


# ---------------------------------------------------------------------------
# tf.data pipeline
# ---------------------------------------------------------------------------

def _count_files_in_split(cache_file: Path) -> int:
    lines = cache_file.read_text().strip().splitlines()
    return len([l for l in lines if l.strip()])


def _split_half_dataset(ds: tf.data.Dataset, total: int) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    """Split a dataset into two halves by cardinality (for val/test split)."""
    half = total // 2
    ds_val = ds.take(half)
    ds_test = ds.skip(half)
    return ds_val, ds_test


def _flip_label(x: tf.Tensor, y: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Remap Keras' alphabetical labels to our clinical convention.

    Keras assigns labels by class-name index: Parasitized=0, Uninfected=1.
    Malaria metrics need the INFEECTED class as positive (label 1) so that
    recall == sensitivity. We flip: Parasitized 0 -> 1, Uninfected 1 -> 0.
    Verified by tests/test_data.py::test_label_mapping.
    """
    return x, tf.cast(1, y.dtype) - y


def build_datasets(cell_images: str | Path,
                   img_size: tuple[int, int] = IMG_SIZE,
                   batch_size: int = BATCH_SIZE,
                   validation_fraction: float = 0.30,
                   seed: int = SEED,
                   augment: bool = False,
                   cache_splits: bool = False,
                   cache_dir: str | Path | None = None):
    """
    Build train/val/test datasets: an exact 70/15/15 split BY FILE.

    How it works:
      1. image_dataset_from_directory with validation_split shuffles file
         order with our seed BEFORE slicing (so class proportions carry over
         in expectation) and yields UNBATCHED examples.
      2. The held-out 30% is bisected by exact file count into val/test.
      3. Labels are flipped so Parasitized=1, Uninfected=0 (see _flip_label).
      4. Optional augmentation (train split only, labels preserved).
      5. Batching happens LAST so val/test counts are exact.

    Determinism: identical (seed, validation_fraction) inputs produce the
    same file lists, so every group member gets the identical split —
    proven by split_fingerprint() and tested in tests/test_data.py.
    """
    train_ds, heldout_ds = tf.keras.utils.image_dataset_from_directory(
        str(cell_images),
        labels="inferred",
        label_mode="binary",
        class_names=CLASS_NAMES,
        color_mode="rgb",
        batch_size=None,                    # unbatched -> exact file split
        image_size=img_size,
        shuffle=True,
        seed=seed,
        validation_split=validation_fraction,
        subset="both",
    )

    heldout_n = int(heldout_ds.cardinality().numpy())
    val_n = heldout_n // 2
    val_ds = heldout_ds.take(val_n)
    test_ds = heldout_ds.skip(val_n)

    # Flip labels: Parasitized 0 -> 1 (infected = positive class)
    train_ds = train_ds.map(_flip_label, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(_flip_label, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.map(_flip_label, num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        data_augmentation = tf.keras.Sequential(
            [
                tf.keras.layers.RandomFlip("horizontal_and_vertical", seed=seed),
                tf.keras.layers.RandomRotation(0.15, seed=seed),
                tf.keras.layers.RandomZoom(0.15, seed=seed),
            ],
            name="data_augmentation",
        )
        # IMPORTANT: return (x, y) — dropping y here would destroy train labels
        train_ds = train_ds.map(
            lambda x, y: (data_augmentation(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    if cache_splits and cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        train_ds = train_ds.cache(str(cache_dir / "train"))
        val_ds = val_ds.cache(str(cache_dir / "val"))
        test_ds = test_ds.cache(str(cache_dir / "test"))

    train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    n_train_batches = int(train_ds.cardinality().numpy())
    # exact file counts: val/test are exact; train approximated from batches
    print(f"Split sizes (files): train~{n_train_batches * batch_size} "
          f"({n_train_batches} batches, last may be partial), "
          f"val={val_n}, test={heldout_n - val_n}")
    return train_ds, val_ds, test_ds


def collect_labels(ds: tf.data.Dataset) -> tuple[np.ndarray, np.ndarray]:
    """
    Iterate a batched dataset once and return (y_true, x_stack).
    x_stack is returned for convenience in tests/eval; for large datasets use
    ds only to pull labels and keep images streaming.
    """
    ys, xs = [], []
    for x, y in ds:
        ys.append(y.numpy())
        xs.append(x.numpy())
    return np.concatenate(ys).ravel(), np.concatenate(xs)


def split_fingerprint(ds: tf.data.Dataset) -> str:
    """
    Deterministic hash of the file order inside a split.

    Two members can call this on their train_ds; if the fingerprints match,
    they trained/evaluated on EXACTLY the same data — this is how we prove
    the 'identical split across notebooks' claim in the report.
    """
    import hashlib
    h = hashlib.sha256()
    for x, _ in ds:
        # hash a small deterministic slice of each batch: first channel,
        # top-left 8x8 patch (x is a batched 4D tensor: B, H, W, C)
        h.update(np.ascontiguousarray(x.numpy()[:, :8, :8, 0]).tobytes())
    return h.hexdigest()[:16]
