import sys
import tensorflow as tf
import os
from pydub import AudioSegment

import sounddevice as sd
import numpy as np

#from model.neural_model import AudioNet_1D
from neural_model import AudioNet_1D


def load_sound(f, lenght):
    # Output is normalized
    s1 = AudioSegment.from_file(f)
    s1 = s1.set_frame_rate(sample_rate)
    s1 = s1.set_channels(1)
    array_in = np.array(s1.get_array_of_samples())  # / 32768
    array_in = np.nan_to_num(array_in / max(abs(array_in)))
    return array_in

print("Initialize model")
# Initialize model
sample_rate = 44100
window_size = sample_rate // 4
# Start running operations on the Graph.
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
#sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
print("loading session")
sess = tf.Session()
print("loading net")
model = AudioNet_1D(window_size, 4, sess=sess)

# check_point_path = r'D:\GIT\AudioAlert\DataHack\vio\checkpoint_67.ckpt'
check_point_path =  os.path.abspath(os.path.join(os.path.dirname(__file__), "wetransfer", "checkpoint_67.ckpt"))
print("Using network at: " + str(check_point_path))
#check_point_path = "/home/guy/workspace/wave_detector/ruslan/wetransfer/checkpoint_67.ckpt"
print("Loading Model")
model.load_model(check_point_path)
print("Loaded Model")
tag = 'model_vio'
import pickle

# val_list = pickle.load(open("val_set.pkl", 'rb'))

val_list = [sys.argv[1], sys.argv[1], sys.argv[1]]

res = []
classes_set = ['up', 'down', 'silence', 'unknown']

fs = 44100
duration = 0.25  # seconds
import os

correct = 0
total = 0
import time

for f in val_list:
    dataset_path =  os.path.abspath(os.path.join(os.path.dirname(__file__), "test"))

    # myrecording = load_sound(os.path.join(dataset_path, f), duration * duration)
    myrecording = load_sound(f, duration * duration)
    myrecording_ = myrecording.flatten() / np.max(np.abs(myrecording))
    vec_in = np.zeros([int(fs * duration)])
    if len(myrecording_) > len(vec_in):
        myrecording_ = myrecording_[:len(vec_in)]
    print(len(myrecording_) ,len(vec_in))
    vec_in[:len(myrecording_)] += myrecording_
    ts = time.time()
    vec = model.eval_track(vec_in)
    print(f)
    print(vec)
    # if classes_set[np.argmax(vec)] in f:
    #    correct += 1
    print(classes_set[np.argmax(vec)])
    total += 1
    print(correct / total)
    print('time: %f' % (time.time() - ts))
