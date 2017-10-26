#!/usr/bin/env python3
from pydub.utils import make_chunks
from pydub import AudioSegment
import random
import sys


def get_random_sample(filepath):
    sound = AudioSegment.from_file(filepath)
    start = 0
    volume = 100.0

    length = sound.duration_seconds
    playchunk = sound[start * 1000.0:(start + length) * 1000.0] - (60 - (60 * (volume / 100.0)))
    chunk_length = 5 # Length in seconds for output
    chunks = make_chunks(playchunk, chunk_length * 1000)
    random.shuffle(chunks)
    for chunk in chunks:
        chunk.export("mashup.wav", format="wav")
        sys.exit()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(add_help=True,
                                     description="Generate chunks of audio so we have a dataset")
    parser.add_argument('audio_file', type=str, help='The Path to the audio file (mp3, wav and more supported)')
    args = parser.parse_args()

    get_random_sample(args.audio_file)