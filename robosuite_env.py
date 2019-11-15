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


RESET = -1000
SUCCESS = -10000
GET_POSE = -51515
CLOSE = -10

PORT_HIGH = 7500
PORT_LOW = 7000

class RealSawyerLift(object):

    def __init__(self, control_freq=10, horizon=1000, camera_height=256,
                 camera_width=256):

        self.ctrl_freq = control_freq
        self.t = horizon
        # self.controller = controller
        self.cam_dim = (camera_height, camera_width)

        self.control_timestep = 1. / self.ctrl_freq
        self.is_py2 = sys.version_info[0] < 3
        self.gripper_state = 0 # 0 == open 1 == closed
        self.timestep = 0
        self.done = False
        self.has_object_obs = False
        self.data_size = 96000

        self.sawyer_interface = None
        self.camera = None

        np.random.seed(time.time())
        self.port1 = np.random.choice(range(PORT_LOW, PORT_HIGH))
        self.port2 = self.port1 + 1

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
        if self.is_py2 and self.controller=='opspace':
            self.osc_controller.reset_flag.publish(True)


    def send(self, data):
        # print('DATA SIZE ***************', sys.getsizeof(data))
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
            pose = self.sawyer_interface.pose_endpoint_transform(pose, des_endpoint='right_hand', curr_endpoint='right_gripper_tip')
            pose.append(self.sawyer_interface._gripper.get_position())
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
        self.camera = cv2.VideoCapture(0)

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
            self.sawyer_interface.reset()
            self.sawyer_interface.open_gripper()
            # Successful Reset
            self.send(array('f', np.array([SUCCESS] * self.dof).tolist()))
            return
        else:
            # Request a reset
            self.send(array('f', np.array([RESET] * self.dof).tolist()))
            _ = self.recv()

        return self._get_observation()

    def _process_image(self, img):
        h, w, _ = img.shape
        new_w = int( float(w) / float(h) * self.cam_dim[0])
        new_shape = (new_w, self.cam_dim[0])

        resized = cv2.resize(img, new_shape, cv2.INTER_AREA)

        center = new_w // 2
        left = center - self.cam_dim[1] // 2 # assume for now that this is multiple of 2
        right = center + self.cam_dim[1] // 2

        img = resized[:,left:right,:]
        img = np.array([img])
        img = np.transpose(img, (0, 3, 1, 2))
        return img

    def _get_image(self):
        for _ in range(30):
            ret, frame = self.camera.read()
        if not ret:
            raise RuntimeError('camera read failed')
        return frame

    def _get_proprio(self):
        self.send(array('f', np.array([GET_POSE]*self.dof).tolist()))
        data = self.bridge.recieve()
        proprio = array('f')
        proprio.fromstring(data)
        return np.array(proprio)

    def _get_observation(self):
        start = time.time()
        di = {}
        # img = self._get_image()
        # di['image'] = self._process_image(img)
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
                self.sawyer_interface.dq = action[:-1]

            elif self.controller == 'opspace':
                self.osc_controller.send_action(action[:-1])

            if action[-1] > 0.25 and not self.gripper_state:
                self.sawyer_interface.close_gripper()
                self.gripper_state = 1
            elif not action[-1] < 0.25 and self.gripper_state:
                self.sawyer_interface.open_gripper()
                self.gripper_state = 0

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

        self.timestep += 1
        action = self._pre_action(action)

        self._apply_action(action)

        end_time = time.time() + self.control_timestep
        while time.time() < end_time:
            NotImplemented

        return self._get_observation(), 0., False, {}

    def render(self):
        cv2.imshow('Observation', self._get_image())
        cv2.waitKey(1)

    def observation_spec(self):
        observation = self._get_observation()
        return observation

    def close(self):
        if self.is_py2:
            self.sawyer_interface.reset()
        else:
            self.camera.release()
            self.send(array('f', np.array([CLOSE] * self.dof).tolist()))



class RCANRealSawyerLift(RealSawyerLift):

    def __init__(self, rcan_restore_func, rcan_kwargs, env_kwargs):

        rcan_kwargs = {} if not rcan_kwargs else rcan_kwargs
        self.rcan = rcan_restore_func(**rcan_kwargs)

        env_kwargs = {} if not env_kwargs else env_kwargs

        try:
            super(self).__init__(**env_kwargs)
        except TypeError:
            super(RCANRealSawyerLift, self).__init__(**env_kwargs)

    def _process_rcan(self, img):
        print('img shape', img.shape)

        #img = self.rcan.forward(torch.Tensor(img).cuda()).contiguous().detach().cpu().numpy()
        print('rcan img shape', img.shape)
        img = np.transpose(img[0, :3, ...], (1,2,0))
        print('1', img.shape)
        img = cv2.resize(img, (84, 84))
        print('2', img.shape)
        img = np.transpose(img, (2, 0, 1))
        print (img.shape)
        return img  # 1x3x84x84

    def step(self, action):
        try:
            obs, rew, done, info = super(self).step(action)
        except TypeError:
            obs, rew, done, info = super(RCANRealSawyerLift, self).step(action)

        img = obs['image']
        img = self._process_rcan(img)

        return img, rew, done, info

    def reset(self):
        d = super(RCANRealSawyerLift, self).reset()

        img = d['image']
        img = self._process_rcan(img)

        return img


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str)
    parser.add_argument('--algo', type=str)
    parser.add_argument('--policy', type=str)
    parser.add_argument('--batch', type=str, default='/home/robot/Desktop/processed_bc_data/output.hdf5')
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
        #return DummyRCAN()

    def load_model(*args, **kwargs):
        model = restore_model('.', 'learner.19000.ckpt')
        configs = restore_config('./config.yml')
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
            env = RealSawyerLift()


        elif args.algo == "lmp":
            assert args.policy is not None, 'Need to specify the path to the policy.'
            config = LMPConfig()
            config.train.data=args.batch
            policy = LatentMotorPlansAlgo(config)
            policy._prepare_for_test_rollouts(args.policy)
            env = RealSawyerLift()

            goal_inds = list(range(len(states_seq)))[self.config.train.seq_length :: self.config.train.seq_length]

        else:
            env = RCANRealSawyerLift(restore_rcan, rcan_kwargs, env_kwargs)

            config = PPO_DEFAULT_ENV_CONFIG
            config.pixel_input = True
            #env = RobosuiteWrapper(env, config)

            policy = load_model(policy_kwargs)

        obs = env.reset()
        # print(obs)
        while True:
            #for n in range(20):
            if args.algo=='bc':
                obs, rew, done, info = env.step(policy.forward(obs['proprio']))

            elif args.algo=='lmp':
                obs, rew, done, info = env.step(policy.forward(obs['proprio']))

            else:

                print(obs)
                obs = obs[:,::-1].copy() #* (255. / 2.) + (255. / 2.)#np.flip(obs, 0).copy()
                #obs = cv2.cvtColor(obs, cv2.COLOR_BGR2RGB)

                processed = obs.copy() #env._get_observation()
                processed = np.transpose(processed, (1,2,0))

                #processed = env._process_rcan(raw_obs['image']).numpy()
                processed = processed.astype(np.uint8) #* (255. / 2.) + (255. / 2.)
                #processed = processed - (127.)
                #processed /= 127.
                #processed = processed.astype(np.uint8)
                #processed = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)
                cv2.imshow('obs', processed)
                cv2.waitKey(0)


                obs = torch.Tensor(obs)
                a = policy.act({'pixel': {'camera0': obs}})

                print(a)

                """
                str_a = ''
                for aa in a:
                    str_a += '{:0.3f},'.format(aa)
                str_a = str_a[:-1]
                print(str_a)
                """
                obs, rew, done, info = env.step(a)

                """
                raw_obs = env._get_observation()
                processed = env._process_rcan(raw_obs['image']).numpy()
                processed = np.transpose(processed, (1,2,0)) * (255. / 2.) + (255. / 2.)
                processed = processed.astype(np.uint8)
                processed = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)
                cv2.imshow('obs', processed)
                cv2.waitKey(1)
                """
            #cv2.imshow('obs2', obs['pixel']['camera0'])
            #cv2.waitKey()

        env.close()

    def py2main():
        RealSawyerLift() #restore_rcan, rcan_kwargs, env_kwargs)

    py36_location = '/home/robot/virtual_envs/py36/bin/python'
    penv_location = '/home/robot/virtual_envs/perlsenv/bin/python'

    assert args.mode is not None, 'Specify the mode, either py36 or penv'
    assert args.algo is not None, 'Specify which algorithm you are using'
    if args.mode == 'py36':
        py3main()
    elif args.mode == 'penv':
        py2main()
