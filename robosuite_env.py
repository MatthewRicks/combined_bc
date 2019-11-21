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

try:
    from .python_bridge import PythonBridge
except:
    from python_bridge import PythonBridge
    

class RealSawyerLift(object):

    def __init__(self, control_freq=10, horizon=1000, camera_height=256,grip_thresh=.25,
                 camera_width=256):

        self.ctrl_freq = control_freq
        self.t = horizon
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

        self.debounce = 0  # make sure the gripper doesnt toggle too quickly

        self.port1 = 7020
        self.port2 = 7021

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
        if data[0] == -1000 and data[3] == -1000 and data[-1] == -1000:
            self.reset()
        elif data[0] == -10000 and data[3] == -10000 and data[-1] == -10000:
            return True
        elif data[0] == -51515 and data[3] == -51515 and data[-1] == -51515:
            pose = self.sawyer_interface.ee_pose
            pose = self.sawyer_interface.pose_endpoint_transform(pose, des_endpoint='right_hand', curr_endpoint='right_gripper_tip')
            pose.append(self.sawyer_interface._gripper.get_position())
            self.send(array('f', pose))
        elif data[0] == -10 and data[3] == -10 and data[-1] == -10:
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
            self.sawyer_interface.open_gripper()
            self.sawyer_interface.reset()
            time.sleep(1)
            # Successful Reset
            self.send(array('f', np.array([-10000.] * self.dof).tolist()))
            return
        else:
            # Request a reset
            self.send(array('f', np.array([-1000.] * self.dof).tolist()))
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
        self.send(array('f', np.array([-51515.]*self.dof).tolist()))
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

        self.timestep += 1
        action = self._pre_action(action)

        self._apply_action(action)
        # print(action)
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
            self.sawyer_interface.open_gripper()
            self.sawyer_interface.reset()
        else:
            self.camera.release()
            self.send(array('f', np.array([-10.] * self.dof).tolist()))



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
    parser.add_argument('--mode', type=str, required=True, choices={'penv', 'py36'})
    parser.add_argument('--algo', type=str, choices={'bc', 'lmp', 'rcan'}, help='choose which algorithm to use (py36)')
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
            env = RealSawyerLift(control_freq=args.ctrl_freq)


        elif args.algo == "lmp":
            assert args.policy is not None, 'Need to specify the path to the policy.'
            config = LMPConfig()
            config.train.data=args.batch
            policy = LatentMotorPlansAlgo(config)
            policy._prepare_for_test_rollouts(args.policy)

            obs_seq, _, _, _, _, _, _ = policy.memory.get_trajectory_at_index(-1)
            goal_inds = list(range(len(obs_seq)))[policy.config.train.seq_length :: policy.config.train.seq_length]
            goal_inds.append(-1)
            env = RealSawyerLift(control_freq=args.ctrl_freq)

        else:
            env = RCANRealSawyerLift(restore_rcan, rcan_kwargs, env_kwargs)

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
        RealSawyerLift(grip_thresh=.1) #restore_rcan, rcan_kwargs, env_kwargs)

    py36_location = '/home/robot/virtual_envs/py36/bin/python'
    penv_location = '/home/robot/virtual_envs/perlsenv/bin/python'
    
    if args.mode == 'py36':
        assert args.algo is not None, 'Specify which algorithm you are using'
        py3main()
    elif args.mode == 'penv':
        py2main()
    
