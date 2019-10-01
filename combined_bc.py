import numpy as np
import tensorflow as tf
from glob import glob
from cv_bridge import CvBridge
import json
from resnet_model import Model
import cv2
import os
from sys import stdout
import signal
import sys


"""
/robot/joint_states
/kinect2/sd/image_color_rect/compressed
/kinect2/sd/image_depth_rect/compressed
/usb_cam/image_raw/compressed
/user_control
"""

def image2gt(msg):
    """
    Convert an img message back into the string message it came from,
    load the json message and then return gt information as well as
    information on if this should be counted in the dataset (if if
    the control information is valid or not.
    """
    return msg[0], msg[1], msg[2]
    img = CvBridge().imgmsg_to_cv2(msg)

    img = np.reshape(img, (-1,))
    string = ""
    for i in range(img.shape[0]):
        string = string + str(chr(img[i]))

    json_msg = json.loads(string)
    valid = json_msg[u'user_info'][u'engaged'] and json_msg[u'user_info'][u'valid'] 
    

    return np.concatenate([ np.ravel(json_msg[u'controller_info'][u'action']),
        np.ravel(json_msg[u'user_info'][u'grasp'])]), msg.header.stamp, valid

class DataReader:

    def __init__(self, npz_file_base, image=False, proprio=False, fused=False, train_percentage=0.9, img_size=(224,224),
                 filter_fn=lambda x: 'pose' in x or 'control' in x or 'usb' in x):

        np_files = glob(npz_file_base + '*.np*')
        np_files = [
            f for f in np_files if filter_fn(f)
        ]
        self.fused = fused
        self.image = image
        self.proprio = proprio
        self.img_size = img_size
        self.np_files = np_files
        self.train_percentage = train_percentage
        self.prepare_dataset()
        
    def prepare_dataset(self):
        dataset = {}
        base_gt = np.array([0, -1.18, 0, 2.18, 0, 0.57, 1.5708])

        not_valid = []
        val = None
        rm = None
        for file in self.np_files:
            if 'control' in file:
                val = file
            if self.proprio and 'usb' in file:
                rm = file
            if self.image and 'pose' in file:
                rm = file

        self.np_files.remove(val)
        self.np_files.insert(0,val)

        if rm is not None:
            self.np_files.remove(rm)
        print(self.np_files)

        for f in self.np_files:

            buff = []
            data = np.load(f, allow_pickle=True)

            # loop over time
            prev_time = -1
            prev_gt = -1
            cnt = 0
            for i in range(data.shape[0]):
                if 'control' in f:
                    gt, time, valid = image2gt(data[i])

                    
                    if prev_time == -1 or time.to_sec() - prev_time.to_sec() > 1.5: #determine if the start of an episode
                        #gt = gt - base_gt
                        cnt += 1
                    else:
                        gt = gt
                        #gt = gt - prev_gt

                    prev_time = time
                    prev_gt = gt

                    if not valid:
                        not_valid.append(i)
                        continue

                    buff.append(gt)
                    k = 'gt'

                elif 'pose' in f: # this is the image
                    if i in not_valid:
                        continue

                    pose = data[i]
                    buff.append(
                        pose
                    )
                    k = 'low_dim_obs'

                else: # this is the image
                    if i in not_valid:
                        continue

                    # img = CvBridge().compressed_imgmsg_to_cv2(data[i])
                    img = data[i]
                    buff.append(
                        # cv2.resize(img, (224,224))
                        img
                    )
                    k = 'obs'


            dataset[k] = np.array(buff)

        dataset_size = dataset[list(dataset.keys())[0]].shape[0]
        inds = np.arange(dataset_size)
        np.random.shuffle(inds)
        self.train_inds = inds[:int(dataset_size * self.train_percentage)]
        self.valid_inds = inds[int(dataset_size * self.train_percentage):]
        self.dataset = dataset

        self.gt_shape = np.squeeze(dataset['gt'][0]).shape
        
        if self.image or self.fused:
            self.obs_shape = np.squeeze(dataset['obs'][0]).shape
            obs_flat = np.reshape(self.dataset['obs'][self.train_inds], (-1, 3))
            self.obs_mean = np.mean(obs_flat)
            try:
                self.obs_std = np.std(obs_flat)
            except:
                self.obs_std = np.std(obs_flat[:100000000,:])

        if self.proprio or self.fused:
            self.low_dim_obs_shape = np.squeeze(dataset['low_dim_obs'][0]).shape

    def iterator(self, mode, batch_size):
        assert mode in ['train', 'valid']

        if mode == 'train':
            inds = self.train_inds
        else:
            inds = self.valid_inds

        for i in range(0, inds.shape[0], batch_size):
            obs = None
            low_dim_obs = None
            end = min(i+batch_size, inds.shape[0])
            if self.image or self.fused:
                # obs = (self.dataset['obs'][i:end] - self.obs_mean) / self.obs_std
                obs = (self.dataset['obs'][i:end] - 127) / 127

            
            if self.proprio or self.fused:
                low_dim_obs = self.dataset['low_dim_obs'][i:end]

            yield low_dim_obs, obs, self.dataset['gt'][i:end]


class Network:

    def __init__(self, dataset_location, img_size=(224,224), batch_size=32, training=True, image=False, proprio=False, fused = False):
        self.image = image
        self.proprio = proprio
        self.fused = fused

        # Ensure exactly one is True
        assert image or proprio or fused, 'Define the inputs to the Network' # At least one is true
        if image and proprio: # if more than one is true, use fused
            self.fused = True

        if self.fused:  # if fused, make sure everything else is false, so there is only one that is true
            self.proprio = False
            self.image=False

        self.dataset = DataReader(dataset_location, train_percentage=0.9, img_size=img_size, image=self.image, proprio=self.proprio, fused=self.fused)
        self.batch_size = batch_size
        self.training = training

        with tf.device('/gpu:0'):
            self.build_network()

    def build_network(self):
        
        self.gt_placeholder = tf.placeholder(tf.float32, shape=[None , self.dataset.gt_shape[0]])

        if self.fused:
            self.obs_placeholder = tf.placeholder(tf.float32, shape=[None, self.dataset.obs_shape[0], self.dataset.obs_shape[1], self.dataset.obs_shape[2]])
            self.obs_placeholder_low_dim = tf.placeholder(tf.float32, shape=[None, self.dataset.low_dim_obs_shape[0]])
            network = Model(
                resnet_size=34,
                bottleneck=False,
                num_classes=10,
                num_filters=64,
                kernel_size=7,
                conv_stride=2,
                first_pool_size=3,
                first_pool_stride=2,
                block_sizes=[3, 4, 6, 3],
                block_strides=[1, 2, 2, 2],
                resnet_version=2,
                data_format='channels_last',
                dtype=tf.float32,
            )
            # Make a network for the images
            hid = network(self.obs_placeholder, True)

            # Make a network for proprioception
            hid2 = tf.layers.dense(self.obs_placeholder_low_dim, 64, activation=tf.nn.relu)
            hid2 = tf.layers.dense(hid2, 64, activation=tf.nn.relu)
            hid2 = tf.layers.dense(hid2, 10, activation=tf.nn.tanh)

            # Combine the two networks
            hid = tf.concat([hid, hid2], axis=-1)
            hid = tf.layers.dense(hid, 64, activation=tf.nn.relu)
            hid = tf.layers.dense(hid, 64, activation=tf.nn.relu)
            hid = tf.layers.dense(hid, self.gt_placeholder.shape[-1], activation=tf.nn.tanh)

            self.outputs = tf.identity(hid)

        if self.image:
            self.obs_placeholder = tf.placeholder(tf.float32, shape=[None, self.dataset.obs_shape[0], self.dataset.obs_shape[1], self.dataset.obs_shape[2]])
            network = Model(
                resnet_size=34,
                bottleneck=False,
                num_classes=self.gt_placeholder.shape[-1],
                num_filters=64,
                kernel_size=7,
                conv_stride=2,
                first_pool_size=3,
                first_pool_stride=2,
                block_sizes=[3, 4, 6, 3],
                block_strides=[1, 2, 2, 2],
                resnet_version=2,
                data_format='channels_last',
                dtype=tf.float32,
            )
            self.outputs = network(self.obs_placeholder, True)
            
        if self.proprio:
            self.obs_placeholder_low_dim = tf.placeholder(tf.float32, shape=[None, self.dataset.low_dim_obs_shape[0]])
            hid = tf.layers.dense(self.obs_placeholder_low_dim, 64, activation=tf.nn.relu)
            hid = tf.layers.dense(hid, 64, activation=tf.nn.relu)
            hid = tf.layers.dense(hid, self.gt_placeholder.shape[-1], activation=tf.nn.tanh) # scale between +-1
            self.outputs = tf.identity(hid)

        joint_loss = tf.reduce_mean(
            tf.reduce_sum(
                tf.abs(self.outputs[:,:-1] - self.gt_placeholder[:,:-1]), axis=-1
            )
        )
        self.loss = joint_loss

        grasp_loss = tf.reduce_mean(
            tf.abs(self.outputs[:,-1] - self.gt_placeholder[:,-1])
        )

        self.loss += grasp_loss
        
        optim = tf.train.AdamOptimizer(learning_rate=5.e-5)
        self.train_op = optim.minimize(self.loss)

        with tf.device('/cpu:0'):
            tf.summary.scalar('joint_loss', joint_loss)

            tf.summary.scalar('grasp_loss', grasp_loss)

            tf.summary.scalar('loss', self.loss)

            if not os.path.exists('./logs'):
                os.mkdir('./logs')

            n_runs = len(glob('./logs/*'))
            os.mkdir('./logs/run_' + str(n_runs))

            self.saver = tf.train.Saver()
            self.saver_best = tf.train.Saver(tf.global_variables(), max_to_keep=1)
            self.writer = tf.summary.FileWriter('./logs/run_' + str(n_runs))
            self.summary_op = tf.summary.merge_all()

    def eval(self, ob, sess):
        if self.image:
            img = ob[0]
            # scaled_img = (img - self.dataset.obs_mean) / self.dataset.obs_std
            scaled_img = (img - 127) / 127
            scaled_img = np.maximum(np.minimum(scaled_img, 5), -5)
            return sess.run(self.outputs, feed_dict={
                self.obs_placeholder: scaled_img
                })

        elif self.proprio:
            return sess.run(self.outputs, feed_dict={
                self.obs_placeholder_low_dim: np.reshape(ob[1],(1,7))
                })

        elif self.fused:
            img = ob[0]
            scaled_img = (img - self.dataset.obs_mean) / self.dataset.obs_std
            scaled_img = np.maximum(np.minimum(scaled_img, 5), -5)
            return sess.run(self.outputs, feed_dict={
                self.obs_placeholder: scaled_img,
                self.obs_placeholder_low_dim: np.reshape(ob[1], (1,7))
                })


    def train(self, restore_path):
        with tf.Session() as sess:
            cntr = 0
            best_valid_loss = 1000
            sess.run(tf.global_variables_initializer())
            if restore_path is not None:
                self.saver.restore(sess, restore_path)
            while True:

                for train_batch in self.dataset.iterator('train', self.batch_size):
                    if self.proprio:
                        summ, loss, _ = sess.run([self.summary_op, self.loss, self.train_op],
                                                 feed_dict={
                                                     self.obs_placeholder_low_dim: train_batch[0],
                                                     self.gt_placeholder: train_batch[2]
                                                 }
                        )
                    elif self.image:
                        summ, loss, _ = sess.run([self.summary_op, self.loss, self.train_op],
                                                 feed_dict={
                                                     self.obs_placeholder: train_batch[1],
                                                     self.gt_placeholder: train_batch[2]
                                                 }
                        )
                    elif self.fused:
                        summ, loss, _ = sess.run([self.summary_op, self.loss, self.train_op],
                                                 feed_dict={
                                                     self.obs_placeholder_low_dim: train_batch[0],
                                                     self.obs_placeholder: train_batch[1],
                                                     self.gt_placeholder: train_batch[2]
                                                 }
                        )
                    self.writer.add_summary(summ)
                    self.writer.flush()

                            # stdout.write("\r%d safenet is intervening" % self.count)
                    stdout.write('\rTrain loss (' + str(cntr) + '): ' + str(float(int(loss * 10000)) / 10000.))
                    stdout.flush()

                valid_loss_sum, valid_loss_count = 0., 0.
                for valid_batch in self.dataset.iterator('valid', self.batch_size):
                    if self.proprio:
                        loss = sess.run(self.loss,
                                        feed_dict={
                                            self.obs_placeholder_low_dim: train_batch[0],
                                            self.gt_placeholder: train_batch[2]
                                        }
                        )
                    elif self.image:
                        loss = sess.run(self.loss,
                                        feed_dict={
                                            self.obs_placeholder: train_batch[1],
                                            self.gt_placeholder: train_batch[2]
                                        }
                        )
                    elif self.fused:
                        loss = sess.run(self.loss,
                                        feed_dict={
                                            self.obs_placeholder_low_dim: train_batch[0],
                                            self.obs_placeholder: train_batch[1],
                                            self.gt_placeholder: train_batch[2]
                                        }
                        )

                    valid_loss_sum += loss
                    valid_loss_count += 1.
                loss = valid_loss_sum / valid_loss_count
                if loss < best_valid_loss:
                    best_valid_loss = loss
                    self.saver_best.save(sess, './checkpoints/best_' + str(loss) + '.ckpt', global_step = cntr)
                    
                print('\tvalidation complete with loss: ' + str(float(int(loss * 10000)) / 10000.))

                if cntr % 1000 == 500 and self.proprio:
                    self.saver.save(sess, './checkpoints/ckpt.ckpt', global_step=cntr)
                elif cntr % 100 == 50 and self.image:
                    self.saver.save(sess, './checkpoints/ckpt.ckpt', global_step=cntr)
                elif cntr % 100 == 50 and self.fused:
                    self.saver.save(sess, './checkpoints/ckpt.ckpt', global_step=cntr)
                cntr += 1

def test_rollout(net, sess):

    from perls_robot_interface_ros.sawyer_gym_env import RoboTurkSawyerEnv
    import time

    ctrl_freq = 5.
    env = RoboTurkSawyerEnv()

    obs = env.reset()

    a = raw_input('Ready')

    grasp = False
    prev_time = time.time()
    for _ in range(5000):
        # import matplotlib.pyplot as plt
        #print(obs.shape)
        #plt.imshow(obs[0])

        obs = env.get_observation()
        action = net.eval(obs, sess)

        if np.ravel(action)[-1] > 0.5:
            action = np.concatenate([np.ravel(action)[:-1], [1]])
            grasp = True
        else:
            action = np.concatenate([np.ravel(action)[:-1], [0]])
        # print(time.time() - s)
        #exit()
        #plt.show()    

        env.step(action)

        if np.ravel(action)[-1] == 0:
            env.unwrapped.robot.robot_arm.open_gripper()
        else:
            env.unwrapped.robot.robot_arm.close_gripper()

        # Enforce Control Frequency
        while(1/(time.time()-prev_time) > ctrl_freq+.01):
            NotImplemented
        print("control rate: {} hz".format(1/(time.time()-prev_time)))
        prev_time=time.time()


def signal_handler(signal, frame):
    """
    Exits on ctrl+C
    """
    sys.exit(0)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_rollout', action='store_true')
    parser.add_argument('--restore_path', type=str)
    parser.add_argument('--proprio', action='store_true')
    parser.add_argument('--image', action='store_true')
    parser.add_argument('--fused', action='store_true')
    args = parser.parse_args()

    signal.signal(signal.SIGINT, signal_handler)  # Handles ctrl+C

    if args.test_rollout:
        with tf.Session() as sess:
            net = Network('../processed_bc_data/', training=False, proprio=args.proprio, image=args.image, fused=args.fused)

            sess.run(tf.global_variables_initializer())
            net.saver.restore(sess, args.restore_path)

            test_rollout(net, sess)
            exit() 

    net = Network('../processed_bc_data/', proprio=args.proprio, image=args.image, fused=args.fused)
    
    net.train(args.restore_path)
