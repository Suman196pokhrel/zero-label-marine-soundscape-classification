#!/usr/bin/env python3
"""
DAT to WAV Converter with Proper Timestamp Extraction
Based on MATLAB scripts: NL_load_2802_logger_data_new.m and NL_load_2823_logger_data_new.m

Converts CMST Sea Noise Logger .DAT files to .WAV with accurate timestamps.
"""

import argparse
import os
import re
import json
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import glob
from datetime import datetime, timezone, timedelta


# ============================================================================
# Constants
# ============================================================================
COUNTS_TO_VOLTS = 5.0 / 65536.0
HEADER_LINES = 5
RECORD_MARKER = b"Record Marker"

TIMESTAMP_REGEXES = [
    r'(\d{4}/\d{2}/\d{2}\s+\d{2}:\d{2}:\d{2})',            # 2009/04/23 14:03:59
    r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})',            # 2009-04-23 14:03:59 (just in case)
]


FIRST_FINAL_REGEX = re.compile(
    r'^(First Data|Finalised)\s*[-:]\s*(\d{4}/\d{2}/\d{2}\s+\d{2}:\d{2}:\d{2})',
    re.IGNORECASE
)

# ============================================================================
# DAT File Parsing
# ============================================================================

def read_header_lines(data: bytes, n_lines: int = 5) -> Tuple[list, int]:
    """Read first n lines from binary data"""
    lines = []
    pos = 0
    
    for _ in range(n_lines):
        nl_pos = data.find(b'\n', pos)
        if nl_pos == -1:
            line = data[pos:]
            pos = len(data)
        else:
            line = data[pos:nl_pos]
            pos = nl_pos + 1
        
        if line.endswith(b'\r'):
            line = line[:-1]
        
        lines.append(line.decode('utf-8', errors='replace'))
    
    return lines, pos


def parse_sample_rate_and_duration(rate_line: str) -> Tuple[Optional[float], Optional[float]]:
    """Extract sample rate and duration from rate line"""
    duration = None
    duration_match = re.search(r'Duration\s+([0-9]+(?:\.[0-9]*)?)', rate_line, re.IGNORECASE)
    if duration_match:
        duration = float(duration_match.group(1))
    
    rate = None
    rate_match = re.search(r'Rate\s+([0-9]+(?:\.[0-9]*)?)', rate_line, re.IGNORECASE)
    if rate_match:
        rate = float(rate_match.group(1))
    
    return rate, duration


def parse_channel_enables(filter0_line: str, filter1_line: str) -> int:
    """Parse which channels are enabled from filter lines"""
    enabled = []
    
    for line in [filter0_line, filter1_line]:
        if len(line) > 12 and line[12].isdigit():
            enabled.append(1 if int(line[12]) != 0 else 0)
        if len(line) > 17 and line[17].isdigit():
            enabled.append(1 if int(line[17]) != 0 else 0)
    
    num_active = sum(enabled)
    return num_active if num_active in (1, 2, 3, 4) else 1


def find_footer_start(data: bytes, header_end: int) -> int:
    """Find where the footer starts (Record Marker)"""
    pos = data.find(RECORD_MARKER, header_end)
    if pos == -1:
        pos = data.find(RECORD_MARKER)
    return pos if pos != -1 else len(data)


def parse_footer_lines(data: bytes, footer_start: int) -> list:
    """Extract footer lines after Record Marker"""
    footer_data = data[footer_start:]
    lines = []
    
    for line in footer_data.split(b'\n'):
        if line.endswith(b'\r'):
            line = line[:-1]
        decoded = line.decode('utf-8', errors='replace').strip()
        if decoded:
            lines.append(decoded)
    
    return lines


def parse_timestamp_from_footer(footer_lines: list) -> Optional[datetime]:
    """
    Parse timestamp from footer based on MATLAB code:
    Rec_start_timeStr = Header{7}
    Format: "Rec start time: yyyy/mm/dd HH:MM:SS (+xxxx)"
    Position 12:30 = "yyyy/mm/dd HH:MM:SS"
    Position 34:38 = fractional seconds offset
    """
    
    # Look for line with "Rec start time" or "Record start time"
    timestamp_line = None
    for line in footer_lines:
        if 'rec start time' in line.lower() or 'record start time' in line.lower():
            timestamp_line = line
            break
    
    if not timestamp_line:
        return None
    
    # Try to extract datetime string
    # Pattern: yyyy/mm/dd HH:MM:SS
    match = re.search(r'(\d{4}/\d{2}/\d{2}\s+\d{2}:\d{2}:\d{2})', timestamp_line)
    if not match:
        return None
    
    try:
        dt = datetime.strptime(match.group(1), '%Y/%m/%d %H:%M:%S')
        
        # Try to extract fractional seconds offset (position 34:38 in original line)
        # This is a timezone/fractional second correction
        # Format: (+xxxx) where xxxx is a 16-bit value
        offset_match = re.search(r'\(([+-]?\d+)\)', timestamp_line)
        if offset_match:
            try:
                offset_value = int(offset_match.group(1))
                # Convert from 16-bit fractional format to seconds
                # Based on MATLAB: str2double(Rec_start_time_zoneStr)/2^16/24/3600
                offset_seconds = offset_value / (2**16) / 24 / 3600
                dt = dt + timedelta(seconds=offset_seconds)
            except:
                pass
        
        return dt.replace(tzinfo=timezone.utc)
    
    except Exception as e:
        print(f"[warn] Failed to parse timestamp from: {timestamp_line[:50]}... Error: {e}")
        return None



def parse_footer_times(footer_lines: list) -> Tuple[Optional[datetime], Optional[datetime]]:
    """
    Parse 'First Data-YYYY/MM/DD HH:MM:SS' and 'Finalised -YYYY/MM/DD HH:MM:SS'
    from footer lines. Returns (first_data_dt_utc, finalised_dt_utc).
    """
    first_dt = None
    final_dt = None
    for line in footer_lines:
        m = FIRST_FINAL_REGEX.search(line.strip())
        if not m:
            continue
        label = m.group(1).lower()
        ts_str = m.group(2)
        try:
            dt = datetime.strptime(ts_str, '%Y/%m/%d %H:%M:%S').replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        if 'first' in label and first_dt is None:
            first_dt = dt
        elif 'final' in label and final_dt is None:
            final_dt = dt
    return first_dt, final_dt



def convert_dat_file(dat_path: str, output_dir: str, calibration_ref: Optional[str] = None) -> Dict[str, Any]:
    """
    Convert a single DAT file to WAV with robust timestamp extraction from header/footer.

    Expects these helpers/constants in the module:
      - read_initial_text_block(data: bytes, max_bytes: int) -> (str, int)
      - parse_timestamp_from_text(text: str) -> Optional[datetime]
      - find_footer_start(data: bytes, header_end: int) -> int
      - parse_footer_lines(data: bytes, footer_start: int) -> list[str]
      - parse_sample_rate_and_duration(rate_line: str) -> (Optional[float], Optional[float])
      - parse_channel_enables(filter0_line: str, filter1_line: str) -> int
      - COUNTS_TO_VOLTS (float)

    Requires imports at module level:
      - import os, re, json, numpy as np, soundfile as sf
      - from datetime import datetime, timezone, timedelta
      - from typing import Optional, Dict, Any
    """
    print(f"Processing: {os.path.basename(dat_path)}")

    # Read whole file
    with open(dat_path, 'rb') as f:
        data = f.read()

    # --- 1) HEADER: read a generous initial text block and mine it for fields ---
    header_text, header_end_guess = read_initial_text_block(data, max_bytes=32768)
    # Extract timestamp from header text (preferred, some datasets have "Rec start time")
    rec_start_time = parse_timestamp_from_text(header_text)

    # Mine header_text for Rate/Duration and filter/channel info by keywords
    header_lines = [ln.rstrip('\r') for ln in header_text.split('\n') if ln.strip()]

    # Try to assemble a "rate/duration" line from any lines that carry those fields
    rate_candidate = next((l for l in header_lines if re.search(r'\bRate\b', l, re.I) or re.search(r'\bSample\b', l, re.I)), "")
    duration_candidate = next((l for l in header_lines if re.search(r'\bDuration\b', l, re.I)), "")
    combined_rate_line = (rate_candidate + " " + duration_candidate).strip()
    fsample_raw, duration = parse_sample_rate_and_duration(combined_rate_line)

    if fsample_raw is None or duration is None:
        raise ValueError(f"Could not parse sample rate/duration from header text in {dat_path}")

    # Collect any lines that look like filter/channel enables
    filter_lines = [l for l in header_lines if re.search(r'filter', l, re.I)]
    filter0_line = filter_lines[0] if len(filter_lines) > 0 else ""
    filter1_line = filter_lines[1] if len(filter_lines) > 1 else ""
    num_channels = parse_channel_enables(filter0_line, filter1_line)
    if num_channels < 1:
        num_channels = 1

    # Actual per-channel sample rate
    fsample = float(fsample_raw) / num_channels
    # Expected total samples (rounded up)
    expected_num_samples = int(np.ceil(float(duration) * fsample))
    expected_values = expected_num_samples * num_channels

    # --- 2) FOOTER: find 'Record Marker' (if any) and parse timestamps there if needed ---
    footer_start = find_footer_start(data, header_end_guess)
    footer_lines = parse_footer_lines(data, footer_start)

    # If header didn't yield a timestamp, many PAPCA files encode it as:
    #   "First Data-YYYY/MM/DD HH:MM:SS - NNNNN"
    #   "Finalised -YYYY/MM/DD HH:MM:SS - NNNNN"
    first_dt = None
    final_dt = None
    if footer_lines:
        # Try generic parse first (if your helper can catch it)
        if rec_start_time is None:
            rec_start_time = parse_timestamp_from_text("\n".join(footer_lines))

        # Explicitly parse First Data / Finalised formats
        ff_rx = re.compile(r'^(First Data|Finalised)\s*[-:]\s*(\d{4}/\d{2}/\d{2}\s+\d{2}:\d{2}:\d{2})', re.I)
        for line in footer_lines:
            m = ff_rx.search(line.strip())
            if not m:
                continue
            label = m.group(1).lower()
            ts_str = m.group(2)
            try:
                dt = datetime.strptime(ts_str, '%Y/%m/%d %H:%M:%S').replace(tzinfo=timezone.utc)
            except ValueError:
                continue
            if 'first' in label and first_dt is None:
                first_dt = dt
            elif 'final' in label and final_dt is None:
                final_dt = dt

        # Prefer First Data if we still don't have a start time
        if rec_start_time is None and first_dt is not None:
            rec_start_time = first_dt

    # --- 3) PAYLOAD bounds and reading ---
    payload_start = header_end_guess
    payload_end = footer_start if (footer_start is not None and footer_start > 0) else len(data)

    # Compute max values that can be read safely from the payload bytes
    payload_nbytes = max(0, payload_end - payload_start)
    max_values_from_payload = payload_nbytes // 2  # uint16 = 2 bytes
    values_to_read = min(expected_values, max_values_from_payload)

    if values_to_read <= 0:
        raise ValueError(f"No readable audio payload in {dat_path} (payload bytes={payload_nbytes})")

    # Read big-endian uint16 interleaved samples
    audio_data = np.frombuffer(
        data[payload_start: payload_start + values_to_read * 2],
        dtype='>u2',
        count=values_to_read
    )

    # Truncate to full frames of (num_channels)
    if audio_data.size % num_channels != 0:
        full_frames = (audio_data.size // num_channels)
        audio_data = audio_data[: full_frames * num_channels]

    num_samples = audio_data.size // num_channels
    if num_samples == 0:
        raise ValueError(f"Zero samples after channel alignment in {dat_path}")

    # Reshape to [samples, channels]
    audio_data = audio_data.reshape((num_samples, num_channels))

    # --- 4) Sanity checks & conversion to volts ---
    overload = (audio_data.max(axis=0) > 60000).astype(int).tolist()
    if any(overload):
        print(f"  [warn] Overload detected in channels: {[i for i, v in enumerate(overload) if v]}")

    audio_volts = audio_data.astype(np.float64) * float(COUNTS_TO_VOLTS)
    # DC removal per channel
    audio_volts = audio_volts - audio_volts.mean(axis=0, keepdims=True)
    audio_volts = audio_volts.astype(np.float32)

    # --- 5) Write WAV & JSON sidecar ---
    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(dat_path))[0]
    wav_path = os.path.join(output_dir, basename + '.wav')

    sf.write(wav_path, audio_volts, int(round(fsample)), subtype='FLOAT')

    metadata: Dict[str, Any] = {
        'source': os.path.abspath(dat_path),
        'wav_path': os.path.abspath(wav_path),
        'sampling_rate_hz': float(fsample),
        'channels': int(num_channels),
        'num_samples_per_channel': int(num_samples),
        'duration_seconds': float(num_samples / fsample),
        'overload_flags': overload,
        'rec_start_time_utc': rec_start_time.isoformat() if rec_start_time else None,
        # Extra fields to help downstream alignment/debug
        'first_data_time_utc': first_dt.isoformat() if first_dt else None,
        'finalised_time_utc': final_dt.isoformat() if final_dt else None,
        'units': 'volts (DC removed)'
    }

    if calibration_ref and os.path.exists(calibration_ref):
        metadata['calibration_ref'] = os.path.abspath(calibration_ref)

    json_path = os.path.join(output_dir, basename + '.json')
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    return metadata




# ============================================================================
# Calibration File Handling
# ============================================================================

def load_calibration_csv(csv_path: str) -> Dict[str, Any]:
    """Load calibration from CSV file"""
    data = np.loadtxt(csv_path, delimiter=',')
    if data.ndim == 1:
        data = data.reshape(-1, 2)
    
    freq_hz = data[:, 0].tolist()
    db_adc_per_uPa = data[:, 1].tolist()
    
    # Convert ADC dB to Volts dB
    COUNTS_TO_VOLTS_DB = 20.0 * np.log10(COUNTS_TO_VOLTS)
    db_volts_per_uPa = (data[:, 1] + COUNTS_TO_VOLTS_DB).tolist()
    
    return {
        'source': os.path.abspath(csv_path),
        'freq_hz': freq_hz,
        'db_adc_per_uPa': db_adc_per_uPa,
        'db_volts_per_uPa': db_volts_per_uPa
    }


def load_calibration_mat(mat_path: str) -> Dict[str, Any]:
    """Load calibration from MAT file"""
    try:
        from scipy.io import loadmat
    except ImportError:
        raise ImportError("scipy is required to load .mat files. Install with: pip install scipy")
    
    mat_data = loadmat(mat_path)
    
    # Look for frequency and calibration data
    freq = None
    calib_db = None
    
    for key in ['Calibration_frequency', 'frequency', 'freq']:
        if key in mat_data:
            freq = mat_data[key].squeeze()
            break
    
    for key in ['Calibration_PSD', 'psd', 'response_db']:
        if key in mat_data:
            calib_db = mat_data[key].squeeze()
            break
    
    if freq is None or calib_db is None:
        raise ValueError(f"Could not find frequency/calibration data in {mat_path}")
    
    freq = np.asarray(freq, dtype=float)
    calib_db = np.asarray(calib_db, dtype=float)
    
    COUNTS_TO_VOLTS_DB = 20.0 * np.log10(COUNTS_TO_VOLTS)
    db_volts_per_uPa = calib_db + COUNTS_TO_VOLTS_DB
    
    return {
        'source': os.path.abspath(mat_path),
        'freq_hz': freq.tolist(),
        'db_adc_per_uPa': calib_db.tolist(),
        'db_volts_per_uPa': db_volts_per_uPa.tolist()
    }


def save_calibration_json(calib_data: Dict[str, Any], logger_id: str, output_root: str) -> str:
    """Save calibration data to JSON"""
    calib_dir = os.path.join(output_root, '_calibration')
    os.makedirs(calib_dir, exist_ok=True)
    
    json_path = os.path.join(calib_dir, f't{logger_id}.json')
    with open(json_path, 'w') as f:
        json.dump({'logger_id': logger_id, **calib_data}, f, indent=2)
    
    return json_path


def find_calibration_file(calib_root: str, logger_id: str) -> Optional[str]:
    """Find calibration file for a logger"""
    for ext in ['.csv', '.mat']:
        path = os.path.join(calib_root, f't{logger_id}_calibration{ext}')
        if os.path.exists(path):
            return path
    return None


# ============================================================================
# Directory Processing
# ============================================================================

def extract_logger_id(path: str) -> Optional[str]:
    """Extract logger ID from path (e.g., 2802, 2823)"""
    match = re.search(r'PAPCA[/\\](\d{4})', path)
    return match.group(1) if match else None


def find_dat_files(raw_root: str) -> list:
    """Find all DAT files in the raw directory structure"""
    pattern = os.path.join(raw_root, '**', 'raw', '*.DAT')
    return sorted(glob.glob(pattern, recursive=True))


def get_output_dir(dat_path: str, raw_root: str, wav_root: str) -> str:
    """Calculate output directory for WAV file"""
    # Get relative path from raw_root
    rel_path = os.path.relpath(os.path.dirname(dat_path), raw_root)
    
    # Replace 'raw' with 'wav' in the path
    parts = rel_path.split(os.sep)
    if parts and parts[-1].lower() == 'raw':
        parts[-1] = 'wav'
    else:
        parts.append('wav')
    
    return os.path.join(wav_root, *parts)

def read_initial_text_block(data: bytes, max_bytes: int = 16384) -> Tuple[str, int]:
    """
    Try to decode the first chunk of the file as text (header and maybe more).
    Returns (decoded_text, end_pos_of_text).
    We scan until we hit likely binary (long runs of non-ASCII) or reach max_bytes.
    """
    scan = data[:max_bytes]
    # Replace non-text with newlines to not break line splits too badly
    ascii_mask = [(32 <= b <= 126) or b in (9, 10, 13) for b in scan]
    # Find last index where we still have a mostly-text window; heuristic:
    # if there is a long run of non-text bytes, we stop before that.
    # Simple approach: take up to the last newline before a 64+ run of non-text bytes.
    non_text_run = 0
    cut = len(scan)
    for i, ok in enumerate(ascii_mask):
        if not ok:
            non_text_run += 1
            if non_text_run >= 64:
                # backtrack to previous newline if possible
                nl = scan.rfind(b'\n', 0, i)
                cut = nl if nl != -1 else i
                break
        else:
            non_text_run = 0

    text = scan[:cut].decode('utf-8', errors='replace')
    return text, cut



def parse_timestamp_from_text(text: str) -> Optional[datetime]:
    """
    Look for lines like:
      "Rec start time: yyyy/mm/dd HH:MM:SS (+xxxx)"
      "Record start time: yyyy/mm/dd HH:MM:SS"
    Returns UTC datetime if found (no TZ info in files -> treat as UTC unless offset found).
    """
    lower = text.lower()
    if 'rec start time' not in lower and 'record start time' not in lower:
        # Even if these phrases are missing, still try regex (some files omit the label)
        pass

    dt = None
    for rx in TIMESTAMP_REGEXES:
        m = re.search(rx, text)
        if m:
            ts = m.group(1)
            # Try both formats
            for fmt in ('%Y/%m/%d %H:%M:%S', '%Y-%m-%d %H:%M:%S'):
                try:
                    dt = datetime.strptime(ts, fmt)
                    break
                except ValueError:
                    continue
            if dt:
                break

    if not dt:
        return None

    # Optional fractional/offset encoded in parentheses e.g. "(+xxxx)"
    # If present, convert the same way your code describes.
    offset_match = re.search(r'\(([+-]?\d+)\)', text)
    if offset_match:
        try:
            offset_value = int(offset_match.group(1))
            offset_seconds = offset_value / (2**16) / 24 / 3600
            dt = dt + timedelta(seconds=offset_seconds)
        except Exception:
            pass

    # These logger times are effectively UTC; don’t force conversion if you later want local.
    return dt.replace(tzinfo=timezone.utc)

# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Convert CMST Sea Noise Logger DAT files to WAV with timestamps'
    )
    parser.add_argument(
        '--raw-root',
        default='data/raw/PAPCA',
        help='Root directory with DAT files (default: data/raw/PAPCA)'
    )
    parser.add_argument(
        '--wav-root',
        default='data/wav/PAPCA',
        help='Output root for WAV files (default: data/wav/PAPCA)'
    )
    parser.add_argument(
        '--calib-root',
        default=None,
        help='Directory with calibration files (t{logger}_calibration.csv/mat)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of files to process (for testing)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be processed without converting'
    )
    
    args = parser.parse_args()
    
    # Find all DAT files
    print("=" * 80)
    print("FINDING DAT FILES")
    print("=" * 80)
    dat_files = find_dat_files(args.raw_root)
    print(f"Found {len(dat_files)} DAT files in {args.raw_root}")
    
    if args.limit:
        dat_files = dat_files[:args.limit]
        print(f"Limited to {len(dat_files)} files for testing")
    
    if not dat_files:
        print("No DAT files found!")
        return
    
    # Load calibration data
    calibrations = {}
    if args.calib_root:
        print("\n" + "=" * 80)
        print("LOADING CALIBRATION FILES")
        print("=" * 80)
        
        logger_ids = set(extract_logger_id(f) for f in dat_files)
        logger_ids.discard(None)
        
        for logger_id in sorted(logger_ids):
            calib_file = find_calibration_file(args.calib_root, logger_id)
            if calib_file:
                try:
                    if calib_file.endswith('.csv'):
                        calib_data = load_calibration_csv(calib_file)
                    else:
                        calib_data = load_calibration_mat(calib_file)
                    
                    calib_json = save_calibration_json(calib_data, logger_id, args.wav_root)
                    calibrations[logger_id] = calib_json
                    print(f"✓ Logger {logger_id}: {os.path.basename(calib_file)}")
                except Exception as e:
                    print(f"✗ Logger {logger_id}: Failed to load - {e}")
            else:
                print(f"⚠ Logger {logger_id}: No calibration file found")
    
    # Dry run
    if args.dry_run:
        print("\n" + "=" * 80)
        print("DRY RUN - Files to be converted:")
        print("=" * 80)
        for dat_file in dat_files[:10]:  # Show first 10
            logger_id = extract_logger_id(dat_file)
            output_dir = get_output_dir(dat_file, args.raw_root, args.wav_root)
            calib = f" [calib: t{logger_id}.json]" if logger_id in calibrations else ""
            print(f"{os.path.basename(dat_file)} -> {output_dir}{calib}")
        if len(dat_files) > 10:
            print(f"... and {len(dat_files) - 10} more files")
        return
    
    # Convert files
    print("\n" + "=" * 80)
    print("CONVERTING DAT FILES")
    print("=" * 80)
    
    success_count = 0
    error_count = 0
    timestamps_found = 0
    
    for i, dat_file in enumerate(dat_files, 1):
        try:
            logger_id = extract_logger_id(dat_file)
            output_dir = get_output_dir(dat_file, args.raw_root, args.wav_root)
            calib_ref = calibrations.get(logger_id)
            
            metadata = convert_dat_file(dat_file, output_dir, calib_ref)
            
            if metadata['rec_start_time_utc']:
                timestamps_found += 1
            
            success_count += 1
            
            if i % 100 == 0:
                print(f"Progress: {i}/{len(dat_files)} files ({timestamps_found} with timestamps)")
        
        except Exception as e:
            print(f"✗ Error processing {os.path.basename(dat_file)}: {e}")
            error_count += 1
    
    print("\n" + "=" * 80)
    print("CONVERSION SUMMARY")
    print("=" * 80)
    print(f"Total files: {len(dat_files)}")
    print(f"✓ Success: {success_count}")
    print(f"✗ Errors: {error_count}")
    print(f"⏰ Timestamps found: {timestamps_found} ({timestamps_found/len(dat_files)*100:.1f}%)")
    print(f"\nOutput directory: {args.wav_root}")


if __name__ == '__main__':
    main()