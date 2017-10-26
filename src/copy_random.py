import os
import shutil
import random
import glob
from slice_audio_classes import ensure_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(add_help=True,
                                     description="Generate chunks of audio so we have a dataset")
    parser.add_argument('sound_folder', type=str, help='source folder')
    parser.add_argument('dest_folder', type=str, help='destination folder')
    args = parser.parse_args()

    files_list = list(glob.glob(os.path.join(args.sound_folder, "*")))

    random.shuffle(files_list)

    ensure_dir(args.dest_folder)
    for i in range(15000):
        shutil.copy(files_list[i], args.dest_folder)


    print("done")
