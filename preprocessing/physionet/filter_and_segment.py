"""
Physionet Apnea-ECG Signal Preprocessing Script

Apply filtering (45Hz low-pass, 0.5-10Hz bandpass, median) and segment dropping
based on kurtosis and skewness criteria.
"""

import numpy as np
import wfdb
from scipy import signal
from scipy.stats import kurtosis, skew
import pandas as pd
import os
import argparse
from tqdm import tqdm


def lowpass_filter(data, cutoff=45, fs=100, order=5):
    """Apply low-pass Butterworth filter"""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return signal.filtfilt(b, a, data)


def bandpass_filter(data, lowcut=0.5, highcut=10, fs=100, order=5):
    """Apply bandpass Butterworth filter"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, data)


def median_filter(data, kernel_size=5):
    """Apply median filter"""
    return signal.medfilt(data, kernel_size=kernel_size)


def apply_all_filters(data, fs=100):
    """Apply all preprocessing filters"""
    # 45Hz low-pass filter
    data = lowpass_filter(data, cutoff=45, fs=fs)
    
    # 0.5-10Hz bandpass filter
    data = bandpass_filter(data, lowcut=0.5, highcut=10, fs=fs)
    
    # Median filter
    data = median_filter(data, kernel_size=5)
    
    return data


def check_segment_quality(segment):
    """
    Check segment quality based on kurtosis and skewness
    
    Returns:
        bool: True if segment should be kept, False if should be dropped
    """
    kurt = kurtosis(segment)
    skewness = skew(segment)
    
    # Drop segments with extreme kurtosis or skewness
    if abs(kurt) > 10 or abs(skewness) > 5:
        return False
    return True


def load_apnea_labels(record_name, data_dir):
    """Load apnea annotations from Physionet"""
    annotation = wfdb.rdann(os.path.join(data_dir, record_name), 'apn')
    
    # Convert annotations to minute-level labels
    # Annotations are in samples, need to convert to minutes
    labels_dict = {}
    for i, (sample, symbol) in enumerate(zip(annotation.sample, annotation.symbol)):
        minute = sample // (100 * 60)  # 100Hz sampling rate
        labels_dict[minute] = 1 if symbol in ['A', 'H', 'O'] else 0
    
    return labels_dict


def segment_signal(signal_data, labels_dict, segment_length=60, fs=100, drop_bad_segments=True):
    """
    Segment signal into fixed-length segments
    
    Args:
        signal_data: Filtered signal
        labels_dict: Dictionary mapping minute index to label
        segment_length: Segment length in seconds (default: 60)
        fs: Sampling frequency
        drop_bad_segments: Whether to drop low-quality segments
        
    Returns:
        List of (segment, label) tuples
    """
    segment_samples = segment_length * fs
    total_minutes = len(signal_data) // (fs * 60)
    segments = []
    
    for minute in range(total_minutes):
        start_sample = minute * fs * 60
        end_sample = start_sample + segment_samples
        
        if end_sample > len(signal_data):
            break
        
        segment = signal_data[start_sample:end_sample]
        
        # Check segment length
        if len(segment) != segment_samples:
            continue
        
        # Check segment quality
        if drop_bad_segments and not check_segment_quality(segment):
            continue
        
        # Get label (default to 0 if not in labels_dict)
        label = labels_dict.get(minute, 0)
        
        segments.append((segment, label))
    
    return segments


def process_record(record_name, data_dir, output_dir, drop_bad_segments=True):
    """Process a single record"""
    # Load signal
    record = wfdb.rdrecord(os.path.join(data_dir, record_name))
    
    # Assuming ECG is in the first channel
    ecg_signal = record.p_signal[:, 0]
    fs = record.fs
    
    # Apply filters
    filtered_signal = apply_all_filters(ecg_signal, fs=fs)
    
    # Load labels
    labels_dict = load_apnea_labels(record_name, data_dir)
    
    # Segment signal
    segments = segment_signal(filtered_signal, labels_dict, segment_length=60, fs=fs, 
                            drop_bad_segments=drop_bad_segments)
    
    # Save segments
    record_output_dir = os.path.join(output_dir, record_name)
    os.makedirs(record_output_dir, exist_ok=True)
    
    segment_info = []
    for i, (segment, label) in enumerate(segments):
        segment_filename = f"{record_name}_seg{i:04d}_label{label}.npy"
        segment_path = os.path.join(record_output_dir, segment_filename)
        np.save(segment_path, segment)
        
        segment_info.append({
            'record_name': record_name,
            'segment_id': i,
            'label': label,
            'filename': segment_filename
        })
    
    # Save segment info
    info_df = pd.DataFrame(segment_info)
    info_df.to_csv(os.path.join(record_output_dir, 'segment_info.csv'), index=False)
    
    return len(segments)


def main():
    parser = argparse.ArgumentParser(description='Physionet Apnea-ECG signal preprocessing')
    parser.add_argument('--data_dir', required=True, help='Raw data directory')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--no_drop', action='store_true', help='Do not drop bad segments')
    parser.add_argument('--exclude_c05', action='store_true', help='Exclude c05 records')
    args = parser.parse_args()
    
    drop_bad_segments = not args.no_drop
    
    print("=" * 80)
    print("Physionet Apnea-ECG Signal Preprocessing")
    print("=" * 80)
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Drop bad segments: {drop_bad_segments}")
    print(f"Exclude c05: {args.exclude_c05}")
    print("=" * 80)
    
    # Get list of records
    # Physionet naming: a01-a20, b01-b05, c01-c10, x01-x35
    records = []
    for prefix in ['a', 'b', 'c', 'x']:
        if prefix == 'a':
            records.extend([f'{prefix}{i:02d}' for i in range(1, 21)])
        elif prefix == 'b':
            records.extend([f'{prefix}{i:02d}' for i in range(1, 6)])
        elif prefix == 'c':
            records.extend([f'{prefix}{i:02d}' for i in range(1, 11)])
        elif prefix == 'x':
            records.extend([f'{prefix}{i:02d}' for i in range(1, 36)])
    
    # Exclude c05 if requested
    if args.exclude_c05:
        records = [r for r in records if r != 'c05']
    
    # Filter records that exist in data directory
    existing_records = []
    for record in records:
        if os.path.exists(os.path.join(args.data_dir, f'{record}.dat')):
            existing_records.append(record)
    
    print(f"\nFound {len(existing_records)} records to process")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    total_segments = 0
    for record_name in tqdm(existing_records, desc="Processing records"):
        try:
            n_segments = process_record(record_name, args.data_dir, args.output_dir, 
                                       drop_bad_segments)
            total_segments += n_segments
            print(f"  Record {record_name}: {n_segments} segments")
        except Exception as e:
            print(f"  Error processing record {record_name}: {e}")
            continue
    
    print("\n" + "=" * 80)
    print(f"Preprocessing complete!")
    print(f"Total segments generated: {total_segments}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()

