"""
UCDDB Labeling Script

Generate 1-minute segment labels (5-second overlap criterion) for UCDDB dataset.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from glob import glob
import os
import argparse


def parse_respiratory_event_file(file_path):
    """Parse respiratory event file (.respevt.txt) into DataFrame"""
    with open(file_path, 'r') as file:
        content = file.read()
    
    lines = content.split('\n')
    
    # Find data start line
    data_start = 0
    for i, line in enumerate(lines):
        if 'Time' in line and 'Type' in line:
            data_start = i + 1
            break
    
    data_lines = [line.strip() for line in lines[data_start:] if line.strip() and line[0].isdigit()]
    
    parsed_data = []
    for line in data_lines:
        parts = [p for p in line.split() if p]
        
        try:
            record = {'Time': parts[0], 'Type': parts[1]}
            current_idx = 2
            
            if len(parts) > current_idx and parts[current_idx] == 'EVENT':
                record['Type'] = 'PB EVENT'
                record['PB/CS'] = 'PB'
                current_idx += 2
            else:
                record['PB/CS'] = None
            
            if len(parts) > current_idx and parts[current_idx].isdigit():
                record['Duration'] = parts[current_idx]
                current_idx += 1
            else:
                record['Duration'] = None
            
            if len(parts) > current_idx and parts[current_idx] != '-':
                try:
                    float(parts[current_idx])
                    record['Low'] = parts[current_idx]
                except ValueError:
                    record['Low'] = None
            else:
                record['Low'] = None
            
            parsed_data.append(record)
        except Exception as e:
            print(f"Error parsing line: {line}")
            continue
    
    df = pd.DataFrame(parsed_data)
    
    # Convert data types
    for col in ['Duration', 'Low']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.time
    
    return df


def time_str_to_seconds(time_str):
    """Convert HH:MM:SS to seconds"""
    hours, minutes, seconds = map(float, time_str.split(':'))
    return hours * 3600 + minutes * 60 + seconds


def convert_time_to_seconds(time_str):
    """Convert HH:MM:SS to integer seconds"""
    h, m, s = map(int, time_str.split(':'))
    return h * 3600 + m * 60 + s


def normalize_seconds(seconds, reference_seconds):
    """Normalize time considering midnight crossing"""
    seconds = float(seconds)
    if seconds < reference_seconds:
        seconds += 24 * 3600
    return seconds


def convert_hours_to_seconds(hours):
    """Convert hours to seconds"""
    return int(hours * 3600)


def label_5sec(subj, meta_dict, subj_detail):
    """
    Label 1-minute segments: 1 if event overlap >= 5 seconds, else 0
    
    Args:
        subj: Subject ID (e.g., '002')
        meta_dict: Dictionary of event metadata
        subj_detail: Subject details DataFrame
        
    Returns:
        DataFrame with minute-level labels
    """
    events_df = meta_dict[subj]
    subj_data = subj_detail[subj_detail['Study Number'].str[-3:] == subj]
    psg_start_time = subj_data['PSG Start Time'].iloc[0]
    study_duration = float(subj_data['Study Duration (hr)'].iloc[0])
    
    psg_start_seconds = convert_time_to_seconds(psg_start_time)
    study_duration_seconds = convert_hours_to_seconds(study_duration)
    max_minutes = int(np.ceil(study_duration_seconds / 60))
    
    # Create 1-minute intervals
    minutes = pd.DataFrame({
        'minute': range(max_minutes),
        'start_time': [i * 60 for i in range(max_minutes)],
        'end_time': [(i + 1) * 60 for i in range(max_minutes)]
    })
    
    base_time = datetime.strptime(psg_start_time, '%H:%M:%S')
    minutes['absolute_time'] = minutes.apply(
        lambda x: (base_time + timedelta(seconds=int(x['start_time']))).strftime('%H:%M:%S'),
        axis=1
    )
    
    # Normalize event times
    normalized_events = events_df.copy()
    normalized_events['start_seconds'] = events_df['start_seconds'].apply(
        lambda x: normalize_seconds(x, psg_start_seconds) - psg_start_seconds
    )
    normalized_events['end_seconds'] = events_df['end_seconds'].apply(
        lambda x: normalize_seconds(x, psg_start_seconds) - psg_start_seconds
    )
    
    def check_events(row):
        events = normalized_events[
            ((normalized_events['start_seconds'] < float(row['end_time'])) & 
             (normalized_events['end_seconds'] > float(row['start_time'])))
        ]
        
        if events.empty:
            return 0
        
        for _, event in events.iterrows():
            overlap = min(float(event['end_seconds']), float(row['end_time'])) - \
                     max(float(event['start_seconds']), float(row['start_time']))
            if overlap >= 5:
                return 1
        return 0
    
    minutes['label'] = minutes.apply(check_events, axis=1)
    minutes = minutes[minutes['start_time'] < study_duration_seconds]
    
    return minutes


def load_ucddb_data(data_dir):
    """Load UCDDB event files and subject details"""
    label_file_list = sorted(glob(os.path.join(data_dir, '*respevt.txt')))
    
    meta_dict = {}
    for file in label_file_list:
        subj_id = file.split('_')[0][-3:]
        df = parse_respiratory_event_file(file)
        df['start_seconds'] = df['Time'].astype(str).apply(time_str_to_seconds)
        df['end_seconds'] = df['start_seconds'] + df['Duration']
        meta_dict[subj_id] = df
    
    subj_detail = pd.read_excel(os.path.join(data_dir, 'SubjectDetails.xls'))
    
    return meta_dict, subj_detail


def generate_labels_for_all_subjects(meta_dict, subj_detail, output_dir):
    """Generate and save 5-second criterion labels for all subjects"""
    os.makedirs(output_dir, exist_ok=True)
    
    lb_5sec = {}
    for k in meta_dict.keys():
        lb_5sec[k] = label_5sec(k, meta_dict, subj_detail)
        
        output_path = os.path.join(output_dir, f'ucddb{k}_label_5sec.csv')
        lb_5sec[k].to_csv(output_path, index=False)
        print(f"Saved: {output_path}")
    
    return lb_5sec


def main():
    parser = argparse.ArgumentParser(description='Generate UCDDB labels')
    parser.add_argument('--data_dir', default='./data', help='Data directory path')
    parser.add_argument('--output_dir', default='./data/labels', help='Output directory path')
    args = parser.parse_args()
    
    print("=" * 80)
    print("UCDDB Labeling Script")
    print("=" * 80)
    
    print("\n[1] Loading data...")
    meta_dict, subj_detail = load_ucddb_data(args.data_dir)
    print(f"   - Loaded {len(meta_dict)} subjects")
    
    print("\n[2] Generating labels...")
    lb_5sec = generate_labels_for_all_subjects(meta_dict, subj_detail, args.output_dir)
    
    print("\n[3] Summary:")
    print("-" * 80)
    print(f"{'Subject':<10} {'Total Segs':<15} {'Event Segs':<20} {'Event Ratio':<15}")
    print("-" * 80)
    
    for subj_id, labels in lb_5sec.items():
        total_segments = len(labels)
        event_segments = labels['label'].sum()
        event_ratio = (event_segments / total_segments) * 100
        print(f"{subj_id:<10} {total_segments:<15} {event_segments:<20} {event_ratio:>6.2f}%")
    
    print("-" * 80)
    print(f"\nLabeling complete! Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

