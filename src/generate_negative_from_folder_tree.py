#!/usr/bin/env python3

from slice_audio_classes import ensure_dir
import shutil
import os
import random
from pydub import AudioSegment

def scan_folder_non_parallel(rootdir, func):
    results = []
    for subdir, dirs, files in os.walk(rootdir):
        for file_name in files:
            results += func(rootdir, subdir, file_name)

    return results


def handle_archive_file(rootdir, subdir, file_name):
    full_name = os.path.join(subdir, file_name)
    return_value = []

    if full_name.endswith(".mp3") or full_name.endswith(".wav"):
        return_value.append(full_name)


    return return_value


if __name__ == "__main__":


    import argparse

    parser = argparse.ArgumentParser(add_help=True,
                                     description="Slice all the data")

    parser.add_argument('in_folder', type=str, help='output_wav folder)')
    parser.add_argument('out_folder', type=str, help='output_wav folder)')

    args = parser.parse_args()
    folder = args.in_folder
    out_folder = args.out_folder

    count = 0

    try:
        shutil.rmtree(out_folder)
    except FileNotFoundError:
        pass
    ensure_dir(out_folder)

    sound_files = scan_folder_non_parallel(folder, handle_archive_file)
    random.shuffle(sound_files)

    for full_name in sound_files:
        print(full_name)
        # Load audio
        sound = AudioSegment.from_file(full_name)

        length = sound.duration_seconds

        start_second = random.uniform(0, length - 1)
        end_second = start_second + 1

        class_name = "negative"

        even_cut_from_track = sound[start_second * 1000:end_second * 1000]

        out_path = os.path.join(out_folder, class_name, os.path.basename(full_name) + str(count) + ".wav")
        ensure_dir(os.path.dirname(out_path))
        even_cut_from_track.export(out_path, format="wav")
