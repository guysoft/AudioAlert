from datetime import datetime
import tensorflow as tf
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))

from neural_model import AudioNet_1D
from datafeeder import DataFeeder
import os

# Initialize model
model = AudioNet_1D(32000, 3)
beta = 1E-8
gamma = 1E-10
learning_rate = 1E-5

sample_rate = 32000

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)


filewriter_path = os.path.join(DIR, "DataHack", "nets", "TB")
checkpoint_path = os.path.join(DIR, "DataHack", "nets")

checkpoint_path = os.path.join(DIR, "DataHack", "nets")
train_path = os.path.join(DIR, "..", "dataset", "train")
val_path = os.path.join(DIR, "..", "dataset", "val")

ensure_dir(filewriter_path)
ensure_dir(checkpoint_path)
ensure_dir(train_path)


num_epochs = 100
batch_size = 200
window_size = 1 * sample_rate
# Get the number of training/validation steps per epoch
val_batches_per_epoch = 50
train_batches_per_epoch = 50
display_step = 20
# List of trainable variables of the layers we want to train
var_list = tf.trainable_variables()
# #L2 norm
reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
logit = model.net_cls_logit
y = model.net_cls
w = model.in_sound_target
x = model.in_sound
keep_prob = model.keep_prob
dropout_rate = 0.5
# Op for calculating the loss
with tf.name_scope("loss"):
    clean_loss = tf.nn.l2_loss(model.in_sound_target - model.net_splt) / window_size
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=y)) + beta * sum(
        reg_losses) + gamma * clean_loss

# Train op
with tf.name_scope("train"):
    # Get gradients of all trainable variables
    gradients = tf.gradients(loss, var_list)
    gradients = list(zip(gradients, var_list))

    # Create optimizer and apply gradient descent to the trainable variables
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.apply_gradients(grads_and_vars=gradients)

# Add gradients to summary
for gradient, var in gradients:
    tf.summary.histogram(var.name + '/gradient', gradient)

# Add the variables we train to the summary
for var in var_list:
    tf.summary.histogram(var.name, var)

# Add the loss to summary
tf.summary.scalar('total_loss', loss)

# Evaluation op: Accuracy of the model
with tf.name_scope("accuracy"):
    correct_pred = tf.equal(tf.argmax(model.net_cls_logit, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Add the accuracy to the summary
tf.summary.scalar('accuracy', accuracy)
tf.summary.scalar('clean_loss', clean_loss)

au_train_summery = tf.summary.audio("train", model.in_sound, sample_rate, 3)
au_clean_summery = tf.summary.audio("clean", model.in_sound, sample_rate, 3)
au_pred_summery = tf.summary.audio("predicted", model.net_splt, sample_rate, 3)

# Merge all summaries together
merged_summary = tf.summary.merge_all()

accuracy_train = tf.summary.scalar('accuracy_train', accuracy)
clean_loss_train = tf.summary.scalar('clean_loss_train', clean_loss)
merged_summary_train = tf.summary.merge([accuracy_train, clean_loss_train])

# Add the accuracy to the summary
val_acc = tf.summary.scalar('Validation Accuracy', accuracy)

# Initialize the FileWriter
writer = tf.summary.FileWriter(filewriter_path)

# Initialize an saver for store model checkpoints
saver = tf.train.Saver()

# Initalize the data generator seperately for the training and validation set
train_generator = DataFeeder(train_path)
print('val')
val_generator = DataFeeder(val_path)

# Start Tensorflow session
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
# with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
with tf.Session() as sess:
    # Initialize all variables
    sess.run(tf.global_variables_initializer())
    # Add the model graph to TensorBoard
    writer.add_graph(sess.graph)
    print("{} Start training...".format(datetime.now()))
    print("{} Open Tensorboard --logdir {}".format(datetime.now(),
                                                   filewriter_path))

    # Loop over number of epochs
    for epoch in range(num_epochs):

        print("{} Epoch number: {}".format(datetime.now(), epoch + 1))

        step = 1
        test_count = 0

        while step < train_batches_per_epoch:

            # Get a batch of images and labels
            batch_xs, batch_clean, batch_ys = train_generator.generate_next_set(batch_size, window_size)
            # And run the training op
            out, train_summ = sess.run([train_op, merged_summary_train], feed_dict={x: batch_xs,
                                                                                    y: batch_ys,
                                                                                    w: batch_clean,
                                                                                    keep_prob: dropout_rate
                                                                                    })
            writer.add_summary(train_summ, epoch * train_batches_per_epoch + step)

            # Generate summary with the current batch of data and write to file
            if step % display_step == 0:
                batch_tx, batch_tclean, batch_ty = val_generator.generate_next_set(batch_size, window_size)
                merged_summary_res = sess.run(merged_summary, feed_dict={x: batch_tx,
                                                    y: batch_ty,
                                                    w: batch_tclean,
                                                    keep_prob: dropout_rate
                                                    })
                writer.add_summary(merged_summary_res, epoch * train_batches_per_epoch + step)
                test_count += 1

            step += 1

        print("{} Saving checkpoint of model...".format(datetime.now()))

        # save checkpoint of the model
        checkpoint_name = os.path.join(checkpoint_path, 'fix_epoch_' + str(epoch + 1) + '.ckpt')
        save_path = saver.save(sess, checkpoint_name)

        print("{} Model checkpoint saved at {}".format(datetime.now(), checkpoint_name))
