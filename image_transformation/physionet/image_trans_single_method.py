"""
Image Transformation Script

Transform 1D ECG signals into 2D images using six different methods:
GADF, GASF, MTF, RP, Scalogram, Spectrogram
"""

import numpy as np
from pyts.image import GramianAngularField, MarkovTransitionField, RecurrencePlot
import pywt
from PIL import Image
import matplotlib.pyplot as plt
import os
import argparse
from glob import glob
from tqdm import tqdm
import scipy.signal


def direct_resize(matrix, target_size=(224, 224)):
    """Resize matrix using PIL"""
    img = Image.fromarray(matrix)
    resized = img.resize(target_size, Image.LANCZOS)
    return np.array(resized)
    

def matrix_to_rgb(matrix):
    """Convert single-channel matrix to RGB using viridis colormap"""
    # Normalize
    norm_matrix = (matrix - matrix.min()) / (matrix.max() - matrix.min() + 1e-8)
    
    # Apply viridis colormap
    cm = plt.get_cmap('viridis')
    rgb_img = cm(norm_matrix)[:, :, :3]
    
    # Convert to uint8
    rgb_img = (rgb_img * 255).astype(np.uint8)
    return rgb_img
        
        
def create_images(signal, fs=100, target_size=(224, 224)):
    """Transform signal into six types of images"""
    
    # GADF
    gadf = GramianAngularField(method='d')
    gadf_matrix = gadf.transform(signal.reshape(1, -1))[0]
    
    # GASF
    gasf = GramianAngularField(method='s')
    gasf_matrix = gasf.transform(signal.reshape(1, -1))[0]
    
    # MTF
    mtf = MarkovTransitionField(n_bins=5)
    mtf_matrix = mtf.transform(signal.reshape(1, -1))[0]
    
    # RP
    rp = RecurrencePlot()
    rp_matrix = rp.transform(signal.reshape(1, -1))[0]
    
    # Scalogram
    widths = np.geomspace(0.5, fs//2, num=1500)
    widths = (fs * 1.0) / (2 * widths * np.pi)
    cwtmatr, _ = pywt.cwt(signal, widths, "cmor1.0-1.0")
    scal_matrix = np.flip(np.abs(cwtmatr) ** 2, axis=0)

    # Spectrogram
    win = scipy.signal.windows.blackman(fs // 4, sym=True)
    stft = scipy.signal.ShortTimeFFT(win=win, hop=(fs // 4 // 2), fs=fs, scale_to='psd')
    Sxx = stft.spectrogram(signal)
    spec_matrix = np.flip(Sxx, axis=0)
    
    # Convert to RGB
    gadf_rgb = matrix_to_rgb(gadf_matrix)
    gasf_rgb = matrix_to_rgb(gasf_matrix)
    mtf_rgb = matrix_to_rgb(mtf_matrix)
    rp_rgb = matrix_to_rgb(rp_matrix)
    scal_rgb = matrix_to_rgb(scal_matrix)
    spec_rgb = matrix_to_rgb(spec_matrix)

    # Resize to target size
    gadf_final = direct_resize(gadf_rgb, target_size)
    gasf_final = direct_resize(gasf_rgb, target_size)
    mtf_final = direct_resize(mtf_rgb, target_size)
    rp_final = direct_resize(rp_rgb, target_size)
    scal_final = direct_resize(scal_rgb, target_size)
    spec_final = direct_resize(spec_rgb, target_size)
    
    return {
        'gadf': gadf_final,
        'gasf': gasf_final,
        'mtf': mtf_final,
        'rp': rp_final,
        'scalogram': scal_final,
        'spectrogram': spec_final
    }


def process_segment_file(segment_path, output_dir, fs=100):
    """Process a single segment file and save transformed images"""
    # Load segment
    signal = np.load(segment_path)
    
    # Extract filename info
    basename = os.path.basename(segment_path)
    filename_no_ext = os.path.splitext(basename)[0]
    
    # Create images
    images = create_images(signal, fs=fs, target_size=(224, 224))
    
    # Save each image type
    for method, img in images.items():
        method_dir = os.path.join(output_dir, method)
        os.makedirs(method_dir, exist_ok=True)
        
        save_path = os.path.join(method_dir, f"{filename_no_ext}.npy")
        np.save(save_path, img)


def main():
    parser = argparse.ArgumentParser(description='Transform signals to images')
    parser.add_argument('--input_dir', required=True, help='Input directory with .npy segment files')
    parser.add_argument('--output_dir', required=True, help='Output directory for images')
    parser.add_argument('--fs', type=int, default=100, help='Sampling frequency (default: 100Hz)')
    args = parser.parse_args()
    
    print("=" * 80)
    print("Image Transformation")
    print("=" * 80)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Sampling frequency: {args.fs}Hz")
    print("=" * 80)
    
    # Find all segment files
    segment_files = []
    for root, dirs, files in os.walk(args.input_dir):
        for file in files:
            if file.endswith('.npy') and 'segment_info' not in file:
                segment_files.append(os.path.join(root, file))
    
    print(f"\nFound {len(segment_files)} segment files")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process all segments
    for segment_path in tqdm(segment_files, desc="Transforming segments"):
        try:
            process_segment_file(segment_path, args.output_dir, fs=args.fs)
        except Exception as e:
            print(f"\nError processing {segment_path}: {e}")
            continue
    
    print("\n" + "=" * 80)
    print("Transformation complete!")
    print(f"Output directory: {args.output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()

