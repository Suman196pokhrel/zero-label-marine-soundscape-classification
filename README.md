**Zero-Label Marine Soundscape Classification (ZMSC)** pipeline:

---

## 📂 Folders

### `data/`

This is your **working data directory**.

* **raw/** → Holds your untouched `.DAT` files (as downloaded).
* **clips/** → 10-second `.wav` slices after conversion & slicing.
* **mels/** → Log-mel spectrograms (as `.npy` or `.png` previews for quick checks).
* **indices/** → Ecoacoustic indices (e.g., spectral entropy, ACI, band energies) stored in `.parquet` or `.csv`.
* **previews/** → Lightweight sanity checks (e.g., small `.mp3` clips + spectrogram images sampled per hour).
* **manifests/** → Tables (`.csv`/`.parquet`) describing the dataset at each stage:

  * `clips.parquet` → all metadata for every clip.
  * `features.parquet` → embeddings (PANNs, SSL, indices).
  * `clusters.parquet` → clustering + auto-labels.

---

### `models/`

* Stores pretrained weights (e.g., **PANNs**, **YAMNet**) and your fine-tuned self-supervised models (BYOL-A, SimCLR).
* Keeps everything local so the pipeline can run reproducibly, even offline.

---

### `cfg/`

* YAML config files controlling the pipeline.
* Example: `default.yaml` defines paths, audio params (sr, hop), feature extraction settings, clustering hyperparameters, etc.
* Advantage: you don’t hard-code values inside scripts → you can swap configs (e.g., `cfg/perth_canyon.yaml`, `cfg/kangaroo_island.yaml`).

---

### `src/`

* Your **pipeline scripts**, named by order:

  * `00_dat_to_wav.py` → convert `.DAT` → `.wav`.
  * `01_slice.py` → cut into 10s clips with 5s hop.
  * `02_features_deep.py` → extract embeddings from pretrained models (PANNs, YAMNet).
  * `03_features_ssl.py` → self-supervised embeddings (BYOL-A, SimCLR).
  * `04_features_indices.py` → ecoacoustic indices.
  * `05_reduce_cluster.py` → PCA → UMAP → HDBSCAN.
  * `06_constrained_clustering.py` → apply must-link / cannot-link constraints.
  * `07_autolabel_train.py` → auto-label BIO / AMBIENT / HUMAN + train classifier.
  * `99_freeze_drop.py` → final “dataset drop” (Step 10).

---

### `scripts/`

* Helper shell scripts.
* Example: `run_all.sh` to run the full pipeline.
* Can also keep “one-liners” here (e.g., daily reslice, recompute indices).

---

## 📝 Root files

* **README.md** → Quick description of the project, folder structure, and how to run.
* **Makefile** → Defines pipeline stages so you can just run `make all`.

---

## 🧠 Why this structure matters

* **Reproducibility** → Data, configs, code, and outputs are cleanly separated.
* **Modularity** → You can rerun only a specific stage (e.g., just clustering) without touching others.
* **Scalability** → Easy to extend for new deployments (just add a new config).
* **Portability** → Step 10 freeze creates a “drop” you can share with supervisors, or re-use for new experiments.

---

👉 Do you want me to also sketch **what each `manifest` table should contain** (column names + datatypes), so you have a clear schema before you start generating them?
