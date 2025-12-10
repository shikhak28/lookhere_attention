#!/usr/bin/env python3
"""
Map unknown ImageNet-HR files into class folders by matching image content hashes

Usage:
  python3 scripts/map_imagenet_hr_by_hash.py --labeled-root /data/imagenet_hr_labeled --unknown-dir /data/imagenet_hr_labeled/unknown

This script loads the HF ImageNet-HR test split, computes an MD5 hash for each dataset image,
and for each file in the unknown folder computes its MD5 and moves it into the class folder
when the hash matches. This handles cases where filename conventions differ.
"""
import argparse
import os
import hashlib
import shutil
from PIL import Image
from datasets import load_dataset
from io import BytesIO


def md5_from_pil(img):
    bio = BytesIO()
    img.convert('RGB').save(bio, format='PNG')
    return hashlib.md5(bio.getvalue()).hexdigest()


def md5_from_file(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--labeled-root', required=True)
    p.add_argument('--unknown-dir', required=True)
    p.add_argument('--cache-dir', default=None)
    args = p.parse_args()

    labeled_root = args.labeled_root
    unknown_dir = args.unknown_dir

    print('Loading HF ImageNet-HR dataset...')
    ds = load_dataset('antofuller/ImageNet-HR', split='test', cache_dir=args.cache_dir)
    print('Rows:', len(ds))

    # build hash -> label map for dataset images
    print('Computing hashes for HF dataset images...')
    hash_map = {}
    for i, ex in enumerate(ds):
        img = ex.get('image')
        if img is None:
            continue
        try:
            h = md5_from_pil(img)
        except Exception:
            # fallback: convert array
            try:
                pil = Image.fromarray(img)
                h = md5_from_pil(pil)
            except Exception:
                continue
        labels = ex.get('labels') or ex.get('label') or None
        if isinstance(labels, (list, tuple)) and len(labels) > 0:
            lab = labels[0]
        else:
            lab = labels if labels is not None else 'unknown'
        hash_map[h] = str(lab)
        if (i+1) % 500 == 0:
            print('  processed', i+1)

    print('HF hash map size:', len(hash_map))

    # Now iterate unknown files and match
    moved = 0
    unmatched = []
    for name in os.listdir(unknown_dir):
        src = os.path.join(unknown_dir, name)
        if not os.path.isfile(src):
            continue
        try:
            h = md5_from_file(src)
        except Exception as e:
            print('Failed hash for', src, e)
            unmatched.append(name)
            continue
        lab = hash_map.get(h)
        if lab is None:
            unmatched.append(name)
            continue
        dst_dir = os.path.join(labeled_root, str(lab))
        os.makedirs(dst_dir, exist_ok=True)
        # choose destination filename as original base name without unknown_ prefix
        basename = name
        if basename.startswith('unknown_'):
            basename = basename[len('unknown_'):]
        dst = os.path.join(dst_dir, basename)
        if os.path.exists(dst):
            # skip and remove source
            os.remove(src)
            continue
        shutil.move(src, dst)
        moved += 1

    print('Moved by hash:', moved)
    print('Unmatched unknown files:', len(unmatched))
    if len(unmatched) > 0:
        print('Examples:', unmatched[:20])

    total_labeled = sum(len(files) for _, _, files in os.walk(labeled_root))
    print('Total files under labeled root now:', total_labeled)


if __name__ == '__main__':
    main()

