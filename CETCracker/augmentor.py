import argparse

import Augmentor
import os


def get_distortion_pipeline(path, num):
    p = Augmentor.Pipeline(path)
    p.zoom(probability=0.5, min_factor=1.05, max_factor=1.05)
    p.random_distortion(probability=1, grid_width=2, grid_height=2, magnitude=2)
    p.sample(num)
    return p


def get_skew_tilt_pipeline(path, num):
    p = Augmentor.Pipeline(path)
    # p.zoom(probability=0.5, min_factor=1.05, max_factor=1.05)
    # p.random_distortion(probability=1, grid_width=6, grid_height=2, magnitude=3)
    p.skew_tilt(probability=0.5,magnitude=0.02)
    p.skew_left_right(probability=0.5,magnitude=0.02)
    p.skew_top_bottom(probability=0.5, magnitude=0.02)
    p.skew_corner(probability=0.5, magnitude=0.02)
    p.sample(num)
    return p


def get_rotate_pipeline(path, num):
    p = Augmentor.Pipeline(path)
    # p.zoom(probability=0.5, min_factor=1.05, max_factor=1.05)
    # p.random_distortion(probability=1, grid_width=6, grid_height=2, magnitude=3)
    p.rotate(probability=1,max_left_rotation=1,max_right_rotation=1)
    p.sample(num)
    return p


def rename(path):
    os.chdir(path)
    cnt = 1
    for filename in os.listdir('.'):
        if filename.endswith('.jpg'):
            os.rename(filename, filename[filename.find('.')-4:filename.find('.')] + str(cnt) + '.jpg')
            cnt += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='generate more data in a dir \'$basepath_output\' in input path')
    parser.add_argument('--base', required=True, type=str, help='the path of original dataset')
    parser.add_argument('--multiple', required=True, type=float, help='the dataset size after augmentation')
    args = parser.parse_args()

    path = args.base
    times = args.multiple

    num = int(len(os.listdir(path)) * times)
    p = get_distortion_pipeline(path, num)
    rename(os.path.join(path, 'output'))
