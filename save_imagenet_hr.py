#!/usr/bin/env python3
"""
Download/convert the HuggingFace ImageNet-HR test split into an ImageFolder layout.

Usage:
  python3 scripts/save_imagenet_hr.py --out /data/imagenet_hr

Notes:
 - Requires `datasets` and `Pillow`.
 - If running in a restricted environment, set `HF_HOME` or pass `--cache_dir`.
 - The script is defensive and prints progress; run in foreground to see errors.
"""
import argparse
import os
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=str, default='/data/imagenet_hr')
    p.add_argument('--cache_dir', type=str, default=None)
    args = p.parse_args()

    out_root = args.out
    os.makedirs(out_root, exist_ok=True)

    print('Loading dataset (this may take a while)...')
    ds = load_dataset('antofuller/ImageNet-HR', split='test', cache_dir=args.cache_dir)
    print('Dataset loaded, rows =', len(ds))

    # try to inspect fields
    print('Features:', ds.features)

    # Determine image field name
    img_field = None
    for candidate in ('image','img','img_bytes'):
        if candidate in ds.column_names:
            img_field = candidate
            break
    if img_field is None:
        # fallback: pick first column of type Image or python object
        img_field = 'image' if 'image' in ds.features else ds.column_names[0]
    print('Using image field:', img_field)

    # Determine label field (may be 'labels', 'label', or 'label_text')
    label_field = None
    for candidate in ('label_text','label','labels','synset'):
        if candidate in ds.column_names:
            label_field = candidate
            break
    print('Using label field:', label_field)

    # Save images into ImageFolder-like layout
    saved = 0
    for i, ex in enumerate(tqdm(ds, desc='Saving')):
        try:
            img = ex[img_field]
            # HF `Image` feature returns PIL.Image for many datasets
            if hasattr(img, 'save'):
                pil = img
            else:
                pil = Image.fromarray(img)
        except Exception as e:
            print('Failed to get image for row', i, e)
            continue

        if label_field is not None and label_field in ex:
            lab = ex[label_field]
            # labels field may be a list (e.g., [int]) or scalar
            if isinstance(lab, (list, tuple)) and len(lab) > 0:
                labval = lab[0]
            else:
                labval = lab
            # convert numeric label to string directory
            labstr = str(labval).replace('/', '_')
        else:
            labstr = 'unknown'

        cls_dir = os.path.join(out_root, labstr)
        os.makedirs(cls_dir, exist_ok=True)
        fname = os.path.join(cls_dir, f'{i:08d}.jpg')
        try:
            pil.save(fname, format='JPEG', quality=95)
            saved += 1
        except Exception as e:
            print('Failed to save image', fname, e)

    print('Done. Saved', saved, 'images into', out_root)


if __name__ == '__main__':
    main()

