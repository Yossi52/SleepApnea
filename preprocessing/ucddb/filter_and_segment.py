"""
UCDDB Signal Preprocessing Script

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


def lowpass_filter(data, cutoff=45, fs=128, order=5):
    """Apply low-pass Butterworth filter"""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return signal.filtfilt(b, a, data)


def bandpass_filter(data, lowcut=0.5, highcut=10, fs=128, order=5):
    """Apply bandpass Butterworth filter"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, data)


def median_filter(data, kernel_size=5):
    """Apply median filter"""
    return signal.medfilt(data, kernel_size=kernel_size)


def apply_all_filters(data, fs=128):
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


def segment_signal(signal_data, labels, segment_length=60, fs=128, drop_bad_segments=True):
    """
    Segment signal into fixed-length segments
    
    Args:
        signal_data: Filtered signal
        labels: Label DataFrame with minute-level labels
        segment_length: Segment length in seconds (default: 60)
        fs: Sampling frequency
        drop_bad_segments: Whether to drop low-quality segments
        
    Returns:
        List of (segment, label) tuples
    """
    segment_samples = segment_length * fs
    segments = []
    
    for idx, row in labels.iterrows():
        start_sample = int(row['start_time'] * fs)
        end_sample = int(row['end_time'] * fs)
        
        # Check if we have enough samples
        if end_sample > len(signal_data):
            break
        
        segment = signal_data[start_sample:end_sample]
        
        # Check segment length
        if len(segment) != segment_samples:
            continue
        
        # Check segment quality
        if drop_bad_segments and not check_segment_quality(segment):
            continue
        
        segments.append((segment, row['label']))
    
    return segments


def process_subject(subject_id, data_dir, label_dir, output_dir, drop_bad_segments=True):
    """Process a single subject"""
    # Load signal
    record_name = f"ucddb{subject_id}"
    record = wfdb.rdrecord(os.path.join(data_dir, record_name))
    
    # Assuming ECG is in the first channel
    ecg_signal = record.p_signal[:, 0]
    fs = record.fs
    
    # Apply filters
    filtered_signal = apply_all_filters(ecg_signal, fs=fs)
    
    # Load labels
    label_file = os.path.join(label_dir, f"ucddb{subject_id}_label_5sec.csv")
    labels = pd.read_csv(label_file)
    
    # Segment signal
    segments = segment_signal(filtered_signal, labels, segment_length=60, fs=fs, 
                            drop_bad_segments=drop_bad_segments)
    
    # Save segments
    subject_output_dir = os.path.join(output_dir, f"ucddb{subject_id}")
    os.makedirs(subject_output_dir, exist_ok=True)
    
    segment_info = []
    for i, (segment, label) in enumerate(segments):
        segment_filename = f"ucddb{subject_id}_seg{i:04d}_label{label}.npy"
        segment_path = os.path.join(subject_output_dir, segment_filename)
        np.save(segment_path, segment)
        
        segment_info.append({
            'subject_id': subject_id,
            'segment_id': i,
            'label': label,
            'filename': segment_filename
        })
    
    # Save segment info
    info_df = pd.DataFrame(segment_info)
    info_df.to_csv(os.path.join(subject_output_dir, 'segment_info.csv'), index=False)
    
    return len(segments)


def main():
    parser = argparse.ArgumentParser(description='UCDDB signal preprocessing')
    parser.add_argument('--data_dir', required=True, help='Raw data directory')
    parser.add_argument('--label_dir', required=True, help='Label directory')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--no_drop', action='store_true', help='Do not drop bad segments')
    args = parser.parse_args()
    
    drop_bad_segments = not args.no_drop
    
    print("=" * 80)
    print("UCDDB Signal Preprocessing")
    print("=" * 80)
    print(f"Data directory: {args.data_dir}")
    print(f"Label directory: {args.label_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Drop bad segments: {drop_bad_segments}")
    print("=" * 80)
    
    # Get list of subjects
    label_files = sorted([f for f in os.listdir(args.label_dir) if f.endswith('_label_5sec.csv')])
    subject_ids = [f.split('ucddb')[1].split('_')[0] for f in label_files]
    
    print(f"\nFound {len(subject_ids)} subjects to process")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    total_segments = 0
    for subject_id in tqdm(subject_ids, desc="Processing subjects"):
        try:
            n_segments = process_subject(subject_id, args.data_dir, args.label_dir, 
                                        args.output_dir, drop_bad_segments)
            total_segments += n_segments
            print(f"  Subject {subject_id}: {n_segments} segments")
        except Exception as e:
            print(f"  Error processing subject {subject_id}: {e}")
            continue
    
    print("\n" + "=" * 80)
    print(f"Preprocessing complete!")
    print(f"Total segments generated: {total_segments}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()

