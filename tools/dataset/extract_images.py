import argparse
import os
import os.path as osp
import tempfile
import zipfile

import cv2
import mmcv
from PIL import Image
import PIL


def parse_args():
    parser = argparse.ArgumentParser(
        description='Extract Dataset Images from Zip File')
    parser.add_argument(
        'dataset', help='Dataset name')
    parser.add_argument(
        'train_images_path', help='the train images of the dataset')
    parser.add_argument(
        'test_images_path', help='the test images of the dataset')

    parser.add_argument('--tmp_dir', help='path of the temporary directory')
    parser.add_argument('-o', '--out_dir', help='output path')
    args = parser.parse_args()
    return args


def main():

    args = parse_args()
    dataset_name = args.dataset
    train_images_path = args.train_images_path
    test_images_path = args.test_images_path

    if args.out_dir is None:
        out_dir = osp.join('/content/pl_mmpose/data', dataset_name)
    else:
        out_dir = args.out_dir

    print('Making directories...')
    mmcv.mkdir_or_exist(out_dir)
    mmcv.mkdir_or_exist(osp.join(out_dir, 'images'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'test_images'))

    with tempfile.TemporaryDirectory(dir=args.tmp_dir) as tmp_dir:
        print('Extracting train images.zip...')
        zip_file = zipfile.ZipFile(train_images_path)
        zip_file.extractall(tmp_dir)

        print('Generating image training dataset...')
        now_dir = osp.join(tmp_dir, 'images')
        for img_name in sorted(os.listdir(now_dir)):
            img = Image.open(osp.join(now_dir, img_name))
            img = img.save(osp.join(out_dir, 'images',
                                    osp.splitext(img_name)[0] + '.jpg'))

    with tempfile.TemporaryDirectory(dir=args.tmp_dir) as tmp_dir:
        print('Extracting test images.zip...')
        zip_file = zipfile.ZipFile(test_images_path)
        zip_file.extractall(tmp_dir)

        print('Generating image testing dataset...')
        now_dir = osp.join(tmp_dir, 'test_images')
        for img_name in sorted(os.listdir(now_dir)):
            img = Image.open(osp.join(now_dir, img_name))
            img = img.save(osp.join(out_dir, 'test_images',
                                    osp.splitext(img_name)[0] + '.jpg'))

        print('Removing the temporary files...')


print('Done!')

if __name__ == '__main__':
    main()