import os
import random
import shutil
import argparse


train_dir = r'train/'
test_dir = r'test/'

if __name__ == "__main__":
    parser = argparse.ArgumentParser( # usage='shuffle --train_dir=DIR_TRAIN --test_dir=DIR_TEST --ratio=TEST/TOTAL',
                                     prefix_chars='-', conflict_handler='error', add_help=True,
                                     description='small util to move samples in train_dir into a test_dir')
    parser.add_argument('--train_dir',type=str, help='where dataset store', required=True)
    parser.add_argument('--test_dir', type=str, help='where test set store', required=True)
    parser.add_argument('-r', '--ratio', type=float, default=0.3, help='test set size / dataset size')
    parser.add_argument('--path_prefix', type=str, default='./', help='the prefix of all the path')
    args = parser.parse_args()
    print(args)

    train_dir = os.path.join(args.path_prefix, args.train_dir)
    test_dir = os.path.join(args.path_prefix, args.test_dir)

    all_samples = []
    test_samples = []
    for base, dirs, filenames in os.walk(train_dir):
        for filename in filenames:
            all_samples.append(filename)
    print(all_samples)
    print(len(all_samples))
    indexes = random.sample(range(len(all_samples)), int(len(all_samples) * args.ratio))
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)
    for i in indexes:
        shutil.move(train_dir + all_samples[i], test_dir + all_samples[i])
