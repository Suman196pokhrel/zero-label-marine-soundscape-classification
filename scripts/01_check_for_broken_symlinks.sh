#!/usr/bin/env bash
# 02_quick_validate_raw.sh
# Quick raw-data validation (steps 1–4) for reef_zmsc
# - Broken symlinks / structure sanity
# - Disk vs manifest diffs (orphans/missing)
# - Naming & directory regex checks
#
# Usage:
#   bash scripts/02_quick_validate_raw.sh [MANIFEST] [ROOT]
# Defaults:
#   MANIFEST=manifest.csv
#   ROOT=data/raw/PAPCA

set -euo pipefail

MANIFEST="${1:-manifest.csv}"
ROOT="${2:-data/raw/PAPCA}"

REPORT_DIR="reports"
TMP_DIR="${REPORT_DIR}/.tmp"
mkdir -p "$REPORT_DIR" "$TMP_DIR"

echo "== reef_zmsc: Quick raw-data validation (1–4) =="
echo "Manifest : ${MANIFEST}"
echo "Root     : ${ROOT}"
echo "Reports  : ${REPORT_DIR}"
echo

# Sanity checks
if [[ ! -f "$MANIFEST" ]]; then
  echo "ERROR: Manifest not found at $MANIFEST" >&2
  exit 1
fi
if [[ ! -d "$ROOT" ]]; then
  echo "ERROR: Root directory not found at $ROOT" >&2
  exit 1
fi

###############################################################################
# Step 1) Broken symlinks / basic structure signal
###############################################################################
echo "-- Step 1: Broken symlinks & structure checks"

# 1a) Broken symlinks
BROKEN_SYMLINKS="${REPORT_DIR}/broken_symlinks.txt"
# find: -xtype l => broken symlinks
find "$ROOT" -xtype l -print | sort -u > "$BROKEN_SYMLINKS" || true
n_broken=$(wc -l < "$BROKEN_SYMLINKS" || echo 0)

# 1b) Count directories that look like .../PAPCA/{logger}/{YYYYMMDD}/raw
# We'll just count potential "raw" leaf dirs 3 levels below ROOT
RAW_DIRS_COUNT=$(find "$ROOT" -type d -name raw | wc -l | awk '{print $1}')
echo "Broken symlinks  : $n_broken (see $BROKEN_SYMLINKS)"
echo "raw/ dir count   : $RAW_DIRS_COUNT"
echo

###############################################################################
# Prepare sorted file lists for Steps 2 & 3
###############################################################################
# Disk files list (absolute/relative paths as printed by find)
DISK_LIST="${TMP_DIR}/disk_files.txt"
# Use GNU find's -printf (available on Linux/most distros). If unavailable, use plain print.
find "$ROOT" -type f -iname '*.dat' -printf '%p\n' | LC_ALL=C sort -u > "$DISK_LIST"

# Manifest paths list (3rd CSV column)
MAN_LIST="${TMP_DIR}/manifest_paths.txt"
# Strip header, take 3rd column; trim spaces
awk -F',' 'NR>1 {gsub(/^[ \t]+|[ \t]+$/, "", $3); print $3}' "$MANIFEST" | LC_ALL=C sort -u > "$MAN_LIST"

###############################################################################
# Step 2) Files on disk but NOT in manifest (orphans)
###############################################################################
echo "-- Step 2: Orphans (on disk but not in manifest)"

ORPHANS="${REPORT_DIR}/orphan_on_disk.txt"
comm -23 "$DISK_LIST" "$MAN_LIST" > "$ORPHANS" || true
n_orphans=$(wc -l < "$ORPHANS" || echo 0)
echo "Orphans: $n_orphans (see $ORPHANS)"
echo

###############################################################################
# Step 3) Files in manifest but NOT on disk (missing)
###############################################################################
echo "-- Step 3: Missing (in manifest but not on disk)"

MISSING="${REPORT_DIR}/missing_on_disk.txt"
comm -13 "$DISK_LIST" "$MAN_LIST" > "$MISSING" || true
n_missing=$(wc -l < "$MISSING" || echo 0)
echo "Missing: $n_missing (see $MISSING)"
echo

###############################################################################
# Step 4) Enforce naming & directory regex
###############################################################################
echo "-- Step 4: Naming & directory regex"

# 4a) Files that do NOT end with .DAT (e.g., lowercase .dat or other ext)
NON_DAT="${REPORT_DIR}/non_DAT_files.txt"
# Anything not *.DAT (case-sensitive)
find "$ROOT" -type f ! -name '*.DAT' -print | LC_ALL=C sort -u > "$NON_DAT" || true
n_non_dat=$(wc -l < "$NON_DAT" || echo 0)

# 4b) Date folder format: list date dirs that are not 8 digits
BAD_DATE_DIRS="${REPORT_DIR}/bad_date_dirs.txt"
# mindepth 2 / maxdepth 2 => .../{logger}/{date}
find "$ROOT" -mindepth 2 -maxdepth 2 -type d -printf '%p\n' \
  | awk -F'/' '{ if ($NF !~ /^[0-9]{8}$/) print $0 }' \
  | LC_ALL=C sort -u > "$BAD_DATE_DIRS" || true
n_bad_date=$(wc -l < "$BAD_DATE_DIRS" || echo 0)

# 4c) Non 8.3 or lowercase filenames among *.DAT (expects UPPERCASE 8.3)
BAD_FILENAMES="${REPORT_DIR}/bad_DAT_filenames.txt"
# For each *.DAT, print files whose basename is NOT strictly ^[A-Z0-9_]{1,8}\.DAT$
# (Allows underscores; adjust if needed.)
find "$ROOT" -type f -name '*.DAT' -printf '%f\t%p\n' \
  | awk 'BEGIN{IGNORECASE=0} { fname=$1; path=$2; if (fname !~ /^[A-Z0-9_]{1,8}\.DAT$/) print path }' \
  | LC_ALL=C sort -u > "$BAD_FILENAMES" || true
n_bad_names=$(wc -l < "$BAD_FILENAMES" || echo 0)

echo "Non-.DAT files       : $n_non_dat   (see $NON_DAT)"
echo "Bad date directories : $n_bad_date  (see $BAD_DATE_DIRS)"
echo "Bad *.DAT filenames  : $n_bad_names (see $BAD_FILENAMES)"
echo

echo "== Done. Artifacts in: ${REPORT_DIR}/ =="
echo "   - broken_symlinks.txt"
echo "   - orphan_on_disk.txt"
echo "   - missing_on_disk.txt"
echo "   - non_DAT_files.txt"
echo "   - bad_date_dirs.txt"
echo "   - bad_DAT_filenames.txt"
