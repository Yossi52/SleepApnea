"""
Fusion Image Creation Script

Combine six single-method images (GADF, GASF, MTF, RP, Scalogram, Spectrogram)
into a single 6-channel fusion image.
"""

import numpy as np
import os
import argparse
from glob import glob
from tqdm import tqdm


def create_fusion_image(image_paths):
    """
    Create 6-channel fusion image from six single-method images
    
    Args:
        image_paths: Dict with keys ['gadf', 'gasf', 'mtf', 'rp', 'scalogram', 'spectrogram']
        
    Returns:
        Fusion image of shape (H, W, 6)
    """
    # Load all images
    images = {}
    for method in ['gadf', 'gasf', 'mtf', 'rp', 'scalogram', 'spectrogram']:
        img = np.load(image_paths[method])  # Shape: (H, W, 3)
        images[method] = img
    
    # Take only the first channel (R channel) from each RGB image
    # since they're all grayscale images mapped to RGB with the same values
    channels = []
    for method in ['gadf', 'gasf', 'mtf', 'rp', 'scalogram', 'spectrogram']:
        channels.append(images[method][:, :, 0])  # Take R channel
    
    # Stack into 6-channel image
    fusion_image = np.stack(channels, axis=-1)  # Shape: (H, W, 6)
    
    return fusion_image


def get_segment_filenames(base_dir):
    """Get all unique segment filenames"""
    # Get files from any method directory
    method_dir = os.path.join(base_dir, 'gadf')
    if not os.path.exists(method_dir):
        raise ValueError(f"Method directory not found: {method_dir}")
    
    files = [f for f in os.listdir(method_dir) if f.endswith('.npy')]
    return files


def process_segment(segment_filename, input_dir, output_dir):
    """Create fusion image for a single segment"""
    # Build paths for all methods
    image_paths = {}
    for method in ['gadf', 'gasf', 'mtf', 'rp', 'scalogram', 'spectrogram']:
        method_dir = os.path.join(input_dir, method)
        image_paths[method] = os.path.join(method_dir, segment_filename)
        
        # Check if file exists
        if not os.path.exists(image_paths[method]):
            return False
    
    # Create fusion image
    fusion_image = create_fusion_image(image_paths)
    
    # Save fusion image
    output_path = os.path.join(output_dir, segment_filename)
    np.save(output_path, fusion_image)
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Create fusion images')
    parser.add_argument('--input_dir', required=True, 
                       help='Input directory containing subdirs for each method')
    parser.add_argument('--output_dir', required=True, 
                       help='Output directory for fusion images')
    args = parser.parse_args()
    
    print("=" * 80)
    print("Fusion Image Creation")
    print("=" * 80)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 80)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get list of segment filenames
    try:
        segment_filenames = get_segment_filenames(args.input_dir)
        print(f"\nFound {len(segment_filenames)} segments to process")
    except Exception as e:
        print(f"Error: {e}")
        return
    
    # Process all segments
    successful = 0
    failed = 0
    
    for filename in tqdm(segment_filenames, desc="Creating fusion images"):
        try:
            if process_segment(filename, args.input_dir, args.output_dir):
                successful += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\nError processing {filename}: {e}")
            failed += 1
    
    print("\n" + "=" * 80)
    print(f"Fusion image creation complete!")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()

