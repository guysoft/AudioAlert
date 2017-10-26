#!/usr/bin/env python3

import xml.etree.ElementTree as ET
import glob
import os
from pydub import AudioSegment
from pydub.utils import make_chunks
import shutil
import numpy as np
import random

class_id_dict = {"2": "glass",
                 "3": "gunshot",
                 "4": "scream"
                 }

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

def get_xml(wav_path):
    xml_path =  os.path.dirname(os.path.dirname(wav_path))
    wav_path = os.path.basename(wav_path)[:-4]
    return os.path.join(xml_path, wav_path.split("_")[0] + ".xml")


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

    #folder = "/home/guy/workspace/datahack/audio/AudioAlert/src/dataset/MIVIA_DB4_dist/training"
    #out_folder = "/tmp/out"

    try:
        shutil.rmtree(out_folder)
    except FileNotFoundError:
        pass
    ensure_dir(out_folder)

    sounds_folder = os.path.join(folder, "sounds")


    glass = []
    scream = []
    gun = []

    for wav_path in glob.glob(os.path.join(sounds_folder, "*.wav")):

        # Load audio
        sound = AudioSegment.from_file(wav_path)


        # Load xml for this audio
        xml_path = get_xml(wav_path)
        tree = ET.parse(xml_path)
        xml_root = tree.getroot()

        sound_array = np.array(sound.get_array_of_samples())
        mask = np.ones(len(sound_array), dtype=bool)

        for event in xml_root.find("events"):
            start_second = float(event.find("STARTSECOND").text)
            end_second = float(event.find("ENDSECOND").text)

            mask[int(start_second* sound.frame_rate):int( end_second * sound.frame_rate)] = False

        new_array = sound_array[mask]

        sound_clean = AudioSegment(
            new_array.tobytes(),
            frame_rate=sound.frame_rate,
            sample_width=new_array.dtype.itemsize,
            channels=1
        )

        chunks = make_chunks(sound_clean, 1 * 1000)

        class_name = "negative"
        for chunk in chunks:
            out_path = os.path.join(out_folder, class_name, os.path.basename(wav_path) + str(count) + ".wav")
            ensure_dir(os.path.dirname(out_path))
            chunk.export(out_path, format="wav")
            count += 1

    print("done")