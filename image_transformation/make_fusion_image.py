"""Stack the grayscale (PIL luma) version of the six viridis images into one 6-channel image.

Writes <image_dir>/fusion<suffix>/<name>.npy of shape (224, 224, 6), channels in the order
gadf, gasf, mtf, rp, scalogram, spectrogram. Early/late/hybrid fusion select three of them.

Usage:
    python image_transformation/make_fusion_image.py --image_dir IMAGES
"""
import argparse
import os

import numpy as np
from PIL import Image
from tqdm import tqdm

TRANSFORMS = ('gadf', 'gasf', 'mtf', 'rp', 'scalogram', 'spectrogram')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--image_dir', required=True, help='output_dir of image_trans_single_method.py')
    p.add_argument('--suffix', default='_drop')
    args = p.parse_args()

    dirs = [os.path.join(args.image_dir, f'{t}{args.suffix}') for t in TRANSFORMS]
    out = os.path.join(args.image_dir, f'fusion{args.suffix}')
    os.makedirs(out, exist_ok=True)
    for name in tqdm(sorted(f for f in os.listdir(dirs[0]) if f.endswith('.npy'))):
        gray = [np.array(Image.fromarray(np.load(os.path.join(d, name))).convert('L')) for d in dirs]
        np.save(os.path.join(out, name), np.stack(gray, axis=-1))


if __name__ == '__main__':
    main()
