import sys

try:
    from perls_robot_interface_ros import SawyerInterface
    # from RobotTeleop.controllers import OpSpaceController
    # from RobotTeleop.configs import RealSawyerDemoServerConfig
    # from RobotTeleop.robots import RealSawyerRobot
    from RobotTeleop import make_robot, make_controller, make_config
    print('imported')
except ImportError:
    from RobotTeleop import make_config
    import cv2
    print(sys.version_info)
    from batchRL.algo import *
    from batchRL.config import *
    import torch
    from gama import GAMA
    try:
        from surreal.main.rollout import restore_agent, restore_model, restore_config
        from surreal.env.wrapper import RobosuiteWrapper
        from surreal.main.ppo_configs import *
    except ImportError:
        print('Unable to import surreal. Hopefully you dont need it!')
    print('Hopefully this is py3')

from array import array

import numpy as np
import time
import argparse
import subprocess
import signal
import time

try:
    from .python_bridge import PythonBridge
    
except:
    from python_bridge import PythonBridge
from video_capture import VideoCapture


RESET = -1000
SUCCESS = -10000
GET_POSE = -51515
GET_PROPRIO = -9999
CLOSE = -10

PORT_HIGH = 7500
PORT_LOW = 7000

class RealSawyerLift(object):

    def __init__(self, control_freq=20, horizon=1000, camera_height=256,grip_thresh=.25,
                 camera_width=256, use_image=False, use_proprio=False,
                 port1=None, joint_proprio=True, do_sleep=True):

        self.joint_proprio = joint_proprio
        self.ctrl_freq = control_freq
        self.t = horizon
        self.use_image = use_image
        self.use_proprio = use_proprio
        self.do_sleep = do_sleep

        assert self.use_image or self.use_proprio, 'One of use_image and use_proprio must be set'
        # self.controller = controller
        self.cam_dim = (camera_height, camera_width)

        self.control_timestep = 1. / self.ctrl_freq
        self.is_py2 = sys.version_info[0] < 3
        self.gripper_closed = 0 # 0 = open 1 = closed
        self.timestep = 0
        self.done = False
        self.has_object_obs = False
        self.data_size = 96000

        self.grip_thresh=grip_thresh
        self.sawyer_interface = None
        self.camera = None

        #np.random.seed(int(time.time()))
        self.port1 = port1 #np.random.choice(range(PORT_LOW, PORT_HIGH))
        self.port2 = self.port1 + 1
        self.debounce = 0  # make sure the gripper doesnt toggle too quickly

        self.config = make_config('RealSawyerDemoServerConfig')
        self.config.infer_settings()

        self.controller = self.config.controller.mode


        if self.is_py2:
            self.bridge = PythonBridge(self.port1, self.port2, self.data_size)
            self._setup_robot()
            self._recv_loop()
        else:
            self.bridge = PythonBridge(self.port2, self.port1, self.data_size)
            self._setup_camera()

    def signal_handler(self, signal, frame):
        """
        Exits on ctrl+C
        """
        if self.is_py2:
            self.sawyer_interface.open_gripper()
            if self.controller=='opspace':
                self.osc_controller.reset_flag.publish(True)

    def send(self, data):
        #self.close()
        #exit()
        self.bridge.send(data)

    def recv(self):
        data = self.bridge.recieve()
        action = array('f')
        action.fromstring(data)
        data = np.array(action)

        # print('Recieved', data)
        if data[0] == RESET and data[3] == RESET and data[-1] == RESET:
            self.reset()
        elif data[0] == SUCCESS and data[3] == SUCCESS and data[-1] == SUCCESS:
            return True
        elif data[0] == GET_POSE and data[3] == GET_POSE and data[-1] == GET_POSE:
            pose = self.sawyer_interface.ee_pose
            pose = self.sawyer_interface.pose_endpoint_transform(pose, des_endpoint='right_hand',
                                                                 curr_endpoint='right_gripper_tip')
            pose.append(self.sawyer_interface._gripper.get_position())
            self.send(array('f', pose))
        elif data[0] == GET_PROPRIO and data[3] == GET_PROPRIO and data[-1] == GET_PROPRIO:
            joint_pos = np.array(self.sawyer_interface.q)
            joint_vel = np.array(self.sawyer_interface.dq)

            gripper_pos = np.array(self.sawyer_interface._gripper.get_position())

            ee_pose = self.sawyer_interface.ee_pose
            ee_pose = np.array(self.sawyer_interface.pose_endpoint_transform(ee_pose, des_endpoint='right_hand',
                                                                    curr_endpoint='right_gripper_tip'))

            print(joint_pos.shape, joint_vel.shape, gripper_pos.shape, ee_pose.shape)
            pose = np.concatenate([
                np.sin(joint_pos), np.cos(joint_pos), joint_vel, [-gripper_pos], [gripper_pos], ee_pose
            ])
            print(joint_pos)
            #print(pose)
            self.send(array('f', pose))
        elif data[0] == CLOSE and data[3] == CLOSE and data[-1] == CLOSE:
            self.close()
            exit()
        else:
            self._apply_action(data)

    def _recv_loop(self):
        while True:
            self.recv()

    def _setup_camera(self):
        self.camera = VideoCapture(0, self.cam_dim)

    def _setup_robot(self):
        self.osc_robot = make_robot(self.config.robot.type, config=self.config)
        self.osc_controller = make_controller(self.config.controller.type, robot=self.osc_robot, config=self.config)
        self.osc_controller.reset()
        self.osc_controller.sync_state()

        self.sawyer_interface = self.osc_robot.robot_arm

        signal.signal(signal.SIGINT, self.signal_handler)  # Handles ctrl+C

    def reset(self):
        self.timestep = 0

        if self.is_py2:
            self.sawyer_interface.open_gripper()
            self.sawyer_interface.reset()
            time.sleep(1)
            # Successful Reset
            self.send(array('f', np.array([SUCCESS] * self.dof).tolist()))
            return
        else:
            # Request a reset
            self.send(array('f', np.array([RESET] * self.dof).tolist()))
            _ = self.recv()

        return self._get_observation()

    #def _process_image(self, img):
    #    h, w, _ = img.shape
    #    new_w = int( float(w) / float(h) * self.cam_dim[0])
    #    new_shape = (new_w, self.cam_dim[0])
    #
    #    resized = cv2.resize(img, new_shape, cv2.INTER_AREA)
    #
    #    center = new_w // 2
    #    left = center - self.cam_dim[1] // 2 # assume for now that this is multiple of 2
    #    right = center + self.cam_dim[1] // 2
    #
    #    img = resized[:,left:right,:]
    #    img = np.array([img])
    #    img = np.transpose(img, (0, 3, 1, 2))
    #    return img

    def _get_image(self):
        ret, frame = self.camera.read()
        if not ret:
            raise RuntimeError('camera read failed')
        return frame

    def _get_proprio(self):
        if self.joint_proprio:
            self.send(array('f', np.array([GET_PROPRIO]*self.dof).tolist()))
        else:
            self.send(array('f', np.array([GET_POSE]*self.dof).tolist()))
        data = self.bridge.recieve()
        proprio = array('f')
        proprio.fromstring(data)
        return np.array(proprio)

    def _get_observation(self):
        start = time.time()
        di = {}

        if self.use_image:
            img = self._get_image()
            di['image'] = img
        if self.use_proprio:
            di['proprio'] = self._get_proprio()
        # print('Observation retrieval time: {}'.format(time.time()-start))
        return di

    def _pre_action(self, action):

        if self.controller == 'velocity':
            # TODO: there is linear interpolation being done between the velocity limits
            #       in sim, do we have something similar to that on the real robot?
            action = np.clip(action, -0.3, 0.3)

        return action

    def _apply_action(self, action):
        if self.is_py2:
            if self.controller == 'velocity':
                self.sawyer_interface.dq = action[:7]

            elif self.controller == 'opspace':
                self.osc_controller.send_action(action[:-1])

            if self.debounce:
                self.debounce-=1
            else:
                if self.gripper_closed:
                    if abs(action[-1]) < 1-self.grip_thresh:
                        self.sawyer_interface.open_gripper()
                        self.gripper_closed=0
                        self.debounce=5
                        print('open gripper')
                else:
                    if abs(action[-1])>self.grip_thresh:
                        self.sawyer_interface.close_gripper()
                        self.gripper_closed=1
                        self.debounce=5
                        print('close gripper')

        else:
            # send to py2
            self.send(bytes(array('f', action.tolist())))

    @property
    def action_spec(self):
        low = np.ones(self.dof) * -1.
        high = np.ones(self.dof) * 1.
        return low, high

    @property
    def dof(self):
        return 8

    def step(self, action):

        if self.done:
            raise RuntimeError('Env is terminated')

        start = time.time()

        self.timestep += 1
        action = self._pre_action(action)

        self._apply_action(action)

        if self.do_sleep:
            end_time = start + self.control_timestep
            while time.time() < end_time:
                pass

        return self._get_observation(), 0., False, {}

    def render(self):
        cv2.imshow('Observation', self._get_image())
        cv2.waitKey(1)

    def observation_spec(self):
        observation = self._get_observation()
        return observation

    def close(self):
        if self.is_py2:
            self.sawyer_interface.open_gripper()
            self.sawyer_interface.reset()
        else:
            self.camera.release()
            self.send(array('f', np.array([CLOSE] * self.dof).tolist()))



class RCANRealSawyerLift(RealSawyerLift):

    def __init__(self, rcan_restore_func, rcan_kwargs, env_kwargs, no_rcan=False):

        rcan_kwargs = {} if not rcan_kwargs else rcan_kwargs
        self.rcan = rcan_restore_func(**rcan_kwargs)
        self.no_rcan = no_rcan

        env_kwargs = {} if not env_kwargs else env_kwargs

        try:
            super(self).__init__(do_sleep=False, **env_kwargs)
        except TypeError:
            super(RCANRealSawyerLift, self).__init__(do_sleep=False,**env_kwargs)

    def _process_rcan(self, img):

        if not self.no_rcan:
            img = self.rcan.forward(torch.Tensor(img).cuda()).contiguous().detach().cpu().numpy()
            img = np.transpose(img[0, :3, ...], (1,2,0))
            img = cv2.resize(img, (84, 84))
            img = np.transpose(img, (2, 0, 1))
        return img  # 1x3x84x84

    def step(self, action):
        start = time.time()
        try:
            obs, rew, done, info = super(self).step(action)
        except TypeError:
            obs, rew, done, info = super(RCANRealSawyerLift, self).step(action)

        img = obs['image']
        if not self.no_rcan:
            img = (self._process_rcan(img) + 1.) / 2.

        if 'proprio' in obs:
            di = {'pixel': {'camera0': img}, 'low_dim': {'flat_inputs': obs['proprio']}}
        else:
            di = {'pixel': {'camera0': img}}

        # do the sleeping here to make rcan part of the control frequency
        end_time = start + self.control_timestep
        while time.time() < end_time:
            pass

        return di, rew, done, info

    def reset(self):
        d = super(RCANRealSawyerLift, self).reset()

        img = d['image']
        img = self._process_rcan(img)

        if 'proprio' in d:
            di = {'pixel': {'camera0': img}, 'low_dim': {'flat_inputs': d['proprio']}}
        else:
            di = {'pixel': {'camera0': img}}

        return di


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--proprio', action='store_true', help='use --proprio if proprioception observations will be used')
    parser.add_argument('--image', action='store_true', help='use --image if image observations will be used')
    parser.add_argument('--port1', type=int, required=True, help='The first port number to use for PythonBridge')
    parser.add_argument('--mode', type=str, required=True, choices={'penv', 'py36'})
    parser.add_argument('--algo', type=str, choices={'bc', 'lmp', 'rcan', 'gama'}, help='choose which algorithm to use (py36)')
    parser.add_argument('--policy', type=str, help='path to trained policy (py36)')

    parser.add_argument('--batch', type=str, default='/home/robot/Desktop/processed_bc_data/output.hdf5',
                            help='Default is output.hdf5 in processed_bc_data (py36)')

    parser.add_argument('--ctrl_freq', type=int, default=10,
                            help='Set the control frequency. (py36)')
    args = parser.parse_args()


    env_kwargs = {}
    rcan_kwargs = {}
    policy_kwargs = {}

    class DummyRCAN:
        def __init__(self, *args, **kwargs):
            pass

        def forward(self, x):
            return x

    class DummyPolicy:
        def __init__(self, *args, **kwargs):
            pass

        def forward(self, x):
            return np.concatenate([np.random.randn(7), [1]])

    def restore_rcan(*args, **kwargs):
        return torch.load('/home/robot/andrewk/pytorch-RCAN/eval_net_G.pth').eval()
        #return torch.load('/home/robot/andrewk/pytorch-RCAN/adpativeG.pth').eval()


    def load_model(*args, **kwargs):
        model = restore_model('./policy_ckpts/joint_vel_NEW_rnd_pos_rot_black', 'learner.64000.ckpt') # '../jonathan/', 'learner.19000.ckpt')
        configs = restore_config('./policy_ckpts/joint_vel_NEW_rnd_pos_rot_black/config.yml') # ../jonathan/config.yml') #
        session_config, learner_config, env_config = \
            configs.session_config, configs.learner_config, configs.env_config
        agent = restore_agent(PPOAgent, learner_config, env_config, session_config, False)
        agent.model.load_state_dict(model)
        agent.agent_mode = 'eval_deterministic'
        return agent
        #return DummyPolicy()


    def py3main():
        if args.algo == "bc":
            assert args.policy is not None, 'Need to specify the path to the policy.'
            config = BCConfig()
            config.train.data = args.batch
            policy = BehavioralCloningAlgo(config)
            policy._prepare_for_test_rollouts(args.policy)
            env = RealSawyerLift(control_freq=args.ctrl_freq, use_image=args.image,
                                 use_proprio=args.proprio)

        elif args.algo == "lmp":
            assert args.policy is not None, 'Need to specify the path to the policy.'
            config = LMPConfig()
            config.train.data=args.batch
            policy = LatentMotorPlansAlgo(config)
            policy._prepare_for_test_rollouts(args.policy)

            env = RealSawyerLift(control_freq=args.ctrl_freq, use_image=args.image,
                                 use_proprio=args.proprio)

            obs_seq, _, _, _, _, _, _ = policy.memory.get_trajectory_at_index(-1)
            goal_inds = list(range(len(obs_seq)))[policy.config.train.seq_length :: policy.config.train.seq_length]
            goal_inds.append(-1)

        elif args.algo =='gama':
            assert args.policy is not None, 'Need to specify the path to the policy.'
            env = RCANRealSawyerLift(restore_rcan, rcan_kwargs, env_kwargs={'use_image':args.image,
                                                                            'use_proprio':args.proprio,
                                                                            'port1': args.port1,
                                                                            'control_freq': args.ctrl_freq},
                                                                            no_rcan=True)

            policy = GAMA()

        else:
            env = RCANRealSawyerLift(restore_rcan, rcan_kwargs, env_kwargs={'use_image':args.image,
                                                                            'use_proprio':args.proprio,
                                                                            'port1': args.port1,
                                                                            'control_freq': args.ctrl_freq})

            config = PPO_DEFAULT_ENV_CONFIG
            config.pixel_input = True
            #env = RobosuiteWrapper(env, config)
            policy = load_model(policy_kwargs)

        obs = env.reset()

        run_loop = True
        while run_loop:
            if args.algo=='bc':
                action = policy.forward(obs['proprio'])
                num_actions = np.shape(action)[0]/7
                for i in range(int(num_actions)):
                    ac = action[i*7:(i+1)*7]
                    obs, rew, done, info = env.step(ac)
                    print('action: {}'.format(ac))

            elif args.algo=='lmp':
                for index in goal_inds:
                    goal = obs_seq[index]
                    for i in range(policy.config.train.seq_length):
                        # start = time.time()
                        t_obs = torch.from_numpy(np.expand_dims(obs['proprio'], axis=0)).float().to(policy.device)
                        t_goal = torch.from_numpy(np.expand_dims(goal, axis=0)).float().to(policy.device)

                        policy.start_control_loop(t_obs,t_goal)
                        action = policy.act(obs=t_obs, goals=t_goal).cpu().detach().numpy().squeeze()
                        print('action: {}'.format(action))
                        obs, rew, done, info = env.step(action)
                        # print('loop time: {}'.format(time.time()-start))

                run_loop = False

            elif args.algo == 'gama':
                start = time.time()

                o = obs['pixel']['camera0'][0,:,::-1].copy() #np.flip(obs, 0).copy()

                processed = o.copy() #env._get_observation()
                processed = np.transpose(processed, (1,2,0))

                processed = processed.astype(np.uint8) #* (255. / 2.) + (255. / 2.)

                cv2.imshow('obs', processed)
                cv2.waitKey(1)

                a = policy.act(obs) #{'pixel': {'camera0': obs}})

                obs, rew, done, info = env.step(a )

            else:

                start = time.time()

                o = obs['pixel']['camera0'][:,::-1].copy() * 255. #np.flip(obs, 0).copy()

                processed = o.copy() #env._get_observation()
                processed = np.transpose(processed, (1,2,0))
                processed = processed.astype(np.uint8) #* (255. / 2.) + (255. / 2.)

                cv2.imshow('obs', processed)
                cv2.waitKey(1)

                start0 = time.time()
                a = policy.act(obs) #{'pixel': {'camera0': obs}})
                print('policy time', time.time() - start0)

                start2 = time.time()
                obs, rew, done, info = env.step(a / 2.)
                print('step time', time.time() - start2)
                print('time', time.time() - start)

        env.close()

    def py2main():
        # joint proprio must be set to true here if using surreal.
        RealSawyerLift(grip_thresh=.1, port1=args.port1,
                       joint_proprio=args.algo not in ['bc', 'lmp'], use_image=args.image, use_proprio=args.proprio)

    py36_location = '/home/robot/virtual_envs/py36/bin/python'
    penv_location = '/home/robot/virtual_envs/perlsenv/bin/python'

    if args.mode == 'py36':
        assert args.algo is not None, 'Specify which algorithm you are using'
        py3main()
    elif args.mode == 'penv':
        py2main()
