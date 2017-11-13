import os
import glob
import numpy as np
import tensorflow as tf
from pydub import AudioSegment

from neural_model import AudioNet_1D


def load_sound(f, sample_rate):
    s1 = AudioSegment.from_file(f)
    s1 = s1.set_frame_rate(sample_rate)
    s1.set_channels(1)
    array_in = np.array(s1.get_array_of_samples())
    array_in = array_in / max(array_in)
    return array_in


# Initialize model
sample_rate = 32000
target_file_for_evaluation = r"C:\Projects\DataHack\AudioAlarm\AudioAlert\7060.wav"
array_in = load_sound(target_file_for_evaluation, sample_rate)
window_size = len(array_in)
number_of_classes = 4
model = AudioNet_1D(window_size, number_of_classes)
model_ckpt = r'C:\Projects\DataHack\AudioAlarm\AudioAlert\DataHack\nets\go_1031_epoch_159.ckpt'
model.load_model(model_ckpt)

print(model.eval_track(array_in))

