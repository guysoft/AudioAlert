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
array_in = load_sound(r"C:\Projects\DataHack\AudioAlarm\AudioAlert\00003_5.wav10946.wav", sample_rate)
array_in = load_sound(r"C:\Projects\DataHack\AudioAlarm\AudioAlert\7060.wav", sample_rate)
window_size = len(array_in)

model = AudioNet_1D(window_size, 4)
model.load_model(r'C:\Projects\DataHack\AudioAlarm\AudioAlert\DataHack\nets\go_1031_epoch_159.ckpt')

print(model.eval_track(array_in))

# model.load_model(r'C:\Projects\DataHack\AudioAlarm\AudioAlert\DataHack\nets\go_best500_epoch_4.ckpt')
for f in glob.glob(r'E:\AudioAlarm\dataset\train\glass\*.wav'):
    # s1 = AudioSegment.from_file(r"C:\Projects\DataHack\AudioAlarm\AudioAlert\7060.wav")
    # s1 = AudioSegment.from_file(r"C:\Projects\DataHack\AudioAlarm\AudioAlert\00003_5.wav10946.wav")
    # s1 = AudioSegment.from_file(r"C:\Projects\DataHack\AudioAlarm\AudioAlert\00002_1.wav35864.wav")
    # s1 = AudioSegment.from_file(r"E:\AudioAlarm\dataset\train\gunshot\00002_3.wav13687.wav")
    array_in = load_sound(f, sample_rate)
    window_size = len(array_in)
    in_a = np.zeros(window_size)
    # in_a[80000:120000]=array_in[80000:120000]
    in_a = array_in

    print(f)
    print(model.eval_track(in_a))


    # num_epochs = 1000
    # batch_size = 1
    # Get the number of training/validation steps per epoch
    # logit = model.net_cls_logit
    # y = model.net_cls
    # w = model.in_sound_target
    # x = model.in_sound
    # keep_prob = model.keep_prob
    # dropout_rate = 1.0


    # Initalize the data generator seperately for the training and validation set
    # with tf.Session() as sess:
    #     # Initialize all variables
    #     # sess.run(tf.global_variables_initializer())
    #     saver.restore(sess, r'C:\Projects\DataHack\AudioAlarm\AudioAlert\DataHack\nets\go_best500_epoch_4.ckpt')
    #     # Add the model graph to TensorBoard
    #     res = sess.run(y, feed_dict={x:np.expand_dims( in_a,axis=0),
    #                                         keep_prob: dropout_rate
    #                                         })
    #     print(res)
