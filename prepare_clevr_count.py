"""
Builds VTAB-1k clevr_count in the same layout as the other datasets under data/vtab-1k:
  <out>/clevr_count/{train800,val200,train800val200,test}.txt
  <out>/clevr_count/images/<split>/<idx:06d>.jpg

Splits and labels follow task_adaptation/data/clevr.py (task='count_all'):
  train800 = train[:800], val200 = train[63000:63200] (90% train / 10% val boundary),
  test = validation (15000 images), label = number of objects - 3 (8 classes).

Needs tensorflow + tensorflow_datasets (preinstalled on Colab). The tfds download is ~18 GB
and preparing it needs ~50 GB of disk, so run it on a Colab CPU runtime, not a GPU one.
Usage: python prepare_clevr_count.py --out /content/vtab-1k --tfds_dir /content/tfds
"""
import os
from argparse import ArgumentParser

import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image

TRAIN_SPLIT_PERCENT = 90


def main():
    parser = ArgumentParser()
    parser.add_argument('--out', type=str, default='./data/vtab-1k')
    parser.add_argument('--tfds_dir', type=str, default='./data/vtab/clevr')
    args = parser.parse_args()
    tf.config.set_visible_devices([], 'GPU')

    builder = tfds.builder('clevr:3.*.*', data_dir=os.path.join(args.tfds_dir, 'tfrecord'))
    builder.download_and_prepare(
        download_dir=os.path.join(args.tfds_dir, 'raw'),
        download_config=tfds.download.DownloadConfig(extract_dir=os.path.join(args.tfds_dir, 'extracted')))

    n_train = (TRAIN_SPLIT_PERCENT * builder.info.splits['train'].num_examples) // 100
    splits = {
        'train800': 'train[:800]',
        'val200': f'train[{n_train}:{n_train + 200}]',
        'train800val200': f'train[:800]+train[{n_train}:{n_train + 200}]',
        'test': 'validation',
    }

    root = os.path.join(args.out, 'clevr_count')
    for split_name, tfds_split in splits.items():
        os.makedirs(os.path.join(root, 'images', split_name), exist_ok=True)
        data = builder.as_dataset(split=tfds_split, shuffle_files=False)
        counts = [0] * 8
        with open(os.path.join(root, f'{split_name}.txt'), 'w') as f:
            for i, item in enumerate(tfds.as_numpy(data)):
                label = int(item['objects']['size'].shape[0]) - 3
                counts[label] += 1
                image_path = f'images/{split_name}/{i:06d}.jpg'
                Image.fromarray(item['image'][..., :3]).save(os.path.join(root, image_path))
                f.write(f'{image_path} {label}\n')
        print(f'{split_name}: {i + 1} images, label counts {counts}')


if __name__ == '__main__':
    main()
