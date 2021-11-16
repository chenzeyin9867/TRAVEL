import os
import gym
import numpy as np
import torch
from gym.spaces.box import Box
from envs.distributions import FixedNormal
import random
import matplotlib.pyplot as plt
import math
OBS_NORM = False
REWARD_SCALE = False
VELOCITY = 1.4 / 60.0
HEIGHT, WIDTH = 20 , 20
HEIGHT_ALL, WIDTH_ALL = 20.0, 20.0
FRAME_RATE = 60
PI = np.pi
STEP_LOW = int(0.5 / VELOCITY)
STEP_HIGH = int(3.5 / VELOCITY)
DELTA_X = (WIDTH_ALL - WIDTH) / 2.0
DELTA_Y = (HEIGHT_ALL - HEIGHT) / 2.0
PENALTY = torch.Tensor([10.0])
OBSERVATION_SPACE = 13


class PassiveHapticsEnv(object):
    def __init__(self, gamma, num_frame_stack, path, random=False, eval=False):
        self.distance = []
        self.num_frame_stack = num_frame_stack  # num of frames stcked before input to the MLP
        self.gamma = gamma                      # γ used in PPO
        self.observation_space = Box(-1., 1., (num_frame_stack * OBSERVATION_SPACE,))
        self.action_space = Box(-1.0, 1.0, (3,))
        self.obs = []                   # current stacked infos send to the networks

        self.x_t_p = WIDTH  / 2
        self.y_t_p = HEIGHT / 2
        self.o_t_p = -PI    / 4         # target's orientation in PE

        self.time_step = 0              # count of the time step
        self.direction_change_num = 0
        self.delta_direction_per_iter = 0
        self.reward = 0.0
        self.path_cnt = 0               # path index for current iteration

        self.r1, self.r2, self.r3 = [], [], []

        if not eval:
            print("Loading the training dataset")
            self.path_file = np.load(os.path.join(path, 'train.npy'), allow_pickle=True)
        else:
            print("Loading the eval dataset")
            self.path_file = np.load(os.path.join(path, 'eval.npy'),  allow_pickle=True)
        self.v_path = self.path_file[self.path_cnt]
        self.v_step_pointer = 0         # the v_path counter


    def get_obs(self):
        # Scale the physical and virtual space into the same square box inside [-1,1]
        # return [(self.x_p+DELTA_X-(WIDTH_ALL)/2)/(WIDTH_ALL/2), (self.y_p+DELTA_Y-(HEIGHT_ALL/2))/(HEIGHT_ALL/2), (self.o_p) / (PI),
        #         (self.x_v-(WIDTH_ALL)/2)/(WIDTH_ALL/2), (self.y_v-(HEIGHT_ALL/2))/(HEIGHT_ALL/2), (self.o_v) / (PI),
        #         # (self.x_t_p+DELTA_X-(WIDTH_ALL)/2)/(WIDTH_ALL/2), (self.y_t_p+DELTA_Y-(HEIGHT_ALL)/2)/(HEIGHT_ALL/2), (self.obj_d_p)/(PI),
        #         # (self.x_t_v-(WIDTH_ALL/2))/(WIDTH_ALL/2), (self.y_t_v-(HEIGHT_ALL/2))/(HEIGHT_ALL/2), (self.obj_d) / (PI)
        #         self.delta_direction_per_iter
        #         ]

        state = [ self.x_p,  self.y_p,   self.o_p,             # physical observation
                  self.x_v,  self.y_v,   self.o_v,             # virtual  observation
                  WIDTH,     HEIGHT,     self.o_t_p,                 # physical space info
                  WIDTH_ALL, HEIGHT_ALL, self.o_t_v,                   # virtual  space info
                  self.delta_direction_per_iter                        # eye direction             
                ]
        
        # print(state)
        return state 

    
    def reset(self):
        '''
        Reset the state when one simulation path ends.
        '''
        self.obs = []
        # Update the path
        self.v_path = self.path_file[self.path_cnt]
        self.path_cnt += 1                                   # next v_path
        self.path_cnt = self.path_cnt % len(self.path_file)  # using the training data iteratively

        # Zero the path pointer
        self.v_step_pointer = 0
        self.time_step = 0
        self.direction_change_num = 0
        self.delta_direction_per_iter = 0
        seed = random.randint(1, 4)
        self.x_v, self.y_v, self.o_v, self.delta_direction_per_iter  = self.v_path[self.v_step_pointer]                                      # initialize the user in virtual space
        self.x_p, self.y_p, self.o_p = initialize(seed)
        self.x_t_v, self.y_t_v, self.o_t_v, _ = self.v_path[-1]                     # initialize the target prop in virtual space
        self.obs.extend(10 * self.get_obs())
        self.reward = 0.0
        
        return toTensor(self.obs)

    def step(self, action):
        # Step forward for K times using the same action based on action-repeat strategy.
        gt, gr, gc = split(action)
        k = random.randint(5, 15)       # action repetition
        reward = torch.Tensor([0.0])    # initial reward for this step period
        for ep in range(k):             # for every iter, get the virtual path info, and steering
            self.vPathUpdate()
            signal = self.physical_step(gt, gr, gc)  # steering the physical env using the actions
            self.time_step += 1
            
            if not signal: # Collide the boundary 
                reward -= PENALTY
                break
            
            if self.v_step_pointer >= len(self.v_path) - 1: # Ends the v_path
                break
                                                        
            elif ep == 0:                                                      
                reward = self.get_reward()                  # Only compute reward once due to the action repeat strategy

        obs = self.obs[OBSERVATION_SPACE:]                  # update the observation after k steps 
        obs.extend(self.get_obs())
        self.obs = obs
        obs = toTensor(obs)
        
        ret_reward = reward    # The returen reward for this action
        self.reward += reward  # Total reward

        if not signal:         # reset the env when leave the tracking space
            bad_mask = 1
            self.reset()
            return obs, ret_reward, [1], [bad_mask]
        elif signal and self.v_step_pointer >= len(self.v_path) - 1: # successfully end one episode, get the final reward
            ret_reward += self.final_reward()
            self.reset()
            return obs, ret_reward, [1], [0]
        else:
            return obs, ret_reward, [0], [0]

    def vPathUpdate(self):
        self.x_v, self.y_v, self.o_v, self.delta_direction_per_iter = \
            self.v_path[self.v_step_pointer]  # unpack the next timestep virtual value
        self.v_step_pointer += 1

    def physical_step(self, gt, gr, gc):
        delta_dis = VELOCITY / gt
        delta_curvature = gc * delta_dis
        delta_rotation = self.delta_direction_per_iter / gr
        self.x_p = self.x_p + torch.cos(torch.Tensor([self.o_p])) * delta_dis
        self.y_p = self.y_p + torch.sin(torch.Tensor([self.o_p])) * delta_dis
        if outbound(self.x_p, self.y_p):
            return False
        self.o_p = norm(self.o_p + delta_curvature + delta_rotation)
        return True

    '''
    when collide with the bundary in eval type, the user just reset instead of end the episode
    '''
    def physical_step_eval(self, gt, gr, gc):
        delta_curvature = gc * (VELOCITY / gt)
        delta_rotation = self.delta_direction_per_iter / gr
        delta_angle = 0
        if abs(delta_curvature) > abs(delta_rotation):
            delta_angle = delta_curvature
        else:
            delta_angle = delta_rotation
        # print(gt, gr, gc, delta_curvature, delta_rotation)
        self.o_p = norm(self.o_p + delta_curvature + delta_rotation)
        # self.o_p = norm(self.o_p + delta_angle)
        delta_dis = VELOCITY / gt
        tmp_x = self.x_p + torch.cos(self.o_p) * delta_dis
        tmp_y = self.y_p + torch.sin(self.o_p) * delta_dis
        if outbound(tmp_x, tmp_y):
            self.o_p = norm(self.o_p + PI)
            return False
        else:
            self.x_p = tmp_x
            self.y_p = tmp_y
            return True


    """
    when in eval mode, initialize the user's postion
    """
    def init_eval_state(self, ind, evalType=0, physical_pos=np.array([0.0,0.0])):
        # print(ind)
        if evalType == 2:
            m = (int(ind / 10)) % 4
            n = (ind % 10)/10.0
            if m == 0:
                self.x_p = 0
                self.y_p = HEIGHT * n
                self.o_p = 0
            elif m == 1:
                self.x_p = WIDTH * n
                self.y_p = HEIGHT
                self.o_p = -PI/2
            elif m == 2:
                self.x_p = WIDTH
                self.y_p = HEIGHT * n
                self.o_p = -PI
            elif m == 3:
                self.x_p = WIDTH * n
                self.y_p = 0
                self.o_p = PI / 2



    def compute_dir_to_obj(self):
        vec = np.array([WIDTH/2-self.x_p, HEIGHT/2-self.y_p])
        if vec[0]==0:
            if vec[1] > 0:
                theta = PI/2
            else:
                theta = -PI/2
        else:
            tan = vec[1] / vec[0]
            theta = np.arctan(tan)
            if vec[0] < 0 and tan >= 0:
                theta = norm(theta + PI)
            elif tan < 0 and vec[0] < 0:
                theta = norm(theta + PI)
        return theta

    def step_specific_path(self, actor_critic, ind, evalType=1):
        collide = 0
        std1 = []
        std2 = []
        std3 = []
        x_l = []
        y_l = []
        gt_l = []
        gr_l = []
        gc_l = []
        self.v_path = self.path_file[ind]
        self.v_step_pointer = 0

        self.x_v, self.y_v, self.o_v, self.delta_direction_per_iter = self.v_path[self.v_step_pointer]
        self.init_eval_state(ind, evalType)
        self.x_t_v, self.y_t_v, self.o_t_v, _ = self.v_path[-1]     # Initialize the object's orientation in VE
        i = self.v_step_pointer
        assign = False
        while i < len(self.v_path):
            with torch.no_grad():
                _value, action_mean, action_log_std = actor_critic.act(
                    torch.Tensor(self.obs).unsqueeze(0))
                dist = FixedNormal(action_mean, action_log_std)
                std1.append(action_log_std[0][0].item())
                std2.append(action_log_std[0][1].item())
                std3.append(action_log_std[0][2].item())
                action = dist.mode()
            gt, gr, gc = split(action)
            gt_l.append(gt.item())
            gr_l.append(gr.item())
            gc_l.append(gc.item())
            for m in range(10):
                if i > len(self.v_path) - 1:
                    signal = False
                    self.reward += self.final_reward()
                    break
                signal = self.physical_step(gt, gr, gc)
                self.vPathUpdate()
                x_l.append(self.x_p)
                y_l.append(self.y_p)
                i += 1
                if not signal:
                    self.reward -= PENALTY
                    collide += 1
                    break
                elif m == 0:
                    self.reward += self.get_reward()
            if not signal:
                break
            init_obs = self.obs[OBSERVATION_SPACE:]
            init_obs.extend(self.get_obs())
            self.obs = init_obs
        vx_l = self.v_path[:, 0]
        vy_l = self.v_path[:, 1] 
        final_dis = math.sqrt((self.x_p - self.x_t_p) * (self.x_p  - self.x_t_p) +
                              (self.y_p - self.y_t_p) * (self.y_p  - self.y_t_p))                  
        return self.reward, final_dis, self.err_angle(), gt_l, gr_l, gc_l, x_l, y_l, vx_l, vy_l, std1, std2, std3, collide

    def err_angle(self):
        # return abs(self.o_x - self.o_t_p) * 180.0 / PI
        # return abs(self.get_reward_angle())*PI
        return abs(delta_angle_norm(self.o_p - self.o_t_p)) * 180.0 / PI

    def final_reward(self):
    
        # r = 10 * (1 - math.pow(distance(self.x_p/WIDTH, self.y_p/HEIGHT, self.x_t_p/WIDTH, self.y_t_p/HEIGHT), 1))
        # r = 10 * (2 - math.pow(distance(self.x_p/WIDTH, self.y_p/HEIGHT, self.x_t_p/WIDTH, self.y_t_p/HEIGHT), 1) - self.err_angle()/ PI)
        # print(0.1*distance(self.x_p, self.y_p, self.x_t_p, self.y_t_p), self.err_angle()/PI)
        # return r 
        r = torch.Tensor([0.0])
        return r

    def step_specific_path_nosrl(self, ind, evalType=2, physical_pos=np.array([0.0,0.0])):
        x_l = []
        y_l = []
        collide = 0
        self.v_path = self.path_file[ind]
        self.v_step_pointer = 0
        self.x_v, self.y_v, self.o_v, self.delta_direction_per_iter = self.v_path[
            self.v_step_pointer]
        self.init_eval_state(ind, evalType, physical_pos)
        self.x_t_v, self.y_t_v, self.o_t_v, _ = self.v_path[-1]     # Initialize the object's orientation in VE

        self.obs.extend(10 * self.get_obs())
        i = self.v_step_pointer

        while i < len(self.v_path):
            gr, gt, gc = torch.Tensor([1.0]), torch.Tensor([1.0]), torch.Tensor([0.0])
            for m in range(10):
                if i > len(self.v_path) - 1:
                    signal = False
                    self.reward += self.final_reward()
                    break
                signal = self.physical_step(gt, gr, gc)
                x_l.append(self.x_p)
                y_l.append(self.y_p)
                self.vPathUpdate()
                i += 1
                if not signal:
                    self.reward -= PENALTY
                    collide += 1
                    break
                elif m == 0:
                    self.reward += self.get_reward()
            if not signal:
                break
        vx_l = self.v_path[:, 0]
        vy_l = self.v_path[:, 1] 
        return self.reward, math.sqrt(
            (self.x_p  - self.x_t_p) * (self.x_p - self.x_t_p) +
            (self.y_p  - self.y_t_p) * (self.y_p - self.y_t_p)), self.err_angle(), x_l, y_l, vx_l, vy_l, collide

    def get_reward(self):
        # d_wall = min(self.x_p/WIDTH, (WIDTH-self.x_p)/WIDTH, (self.y_p)/HEIGHT, (HEIGHT-self.y_p)/HEIGHT)
        # r2 = self.get_reward_distance()
        r1 = self.get_reward_wall()
        # r3 = self.get_reward_angle()
        return r1
 

    def print(self):
        print("physical:", self.x_p, " ", self.y_p, " ", self.o_p)
        print("virtual:", self.x_v, " ", self.y_v, " ", self.o_v)

    def get_reward_distance(self):
        d1_ratio = distance((self.x_p+DELTA_X)/WIDTH_ALL, (self.y_p+DELTA_Y)/HEIGHT_ALL, (self.x_t_p+DELTA_X)/WIDTH_ALL, (self.y_t_p+DELTA_Y)/HEIGHT_ALL)
        d2_ratio = distance(self.x_v/WIDTH_ALL, self.y_v/HEIGHT_ALL, self.x_t_v/WIDTH_ALL, self.y_t_v/WIDTH_ALL)
        # delta_distance_ratio = abs(d1_ratio-d2_ratio) * np.exp(-3*d2_ratio)
        delta_distance_ratio = abs(d1_ratio-d2_ratio)
        # max_d = max(abs(self.x_p-self.x_t_p)/WIDTH, abs(self.y_p-self.y_t_p)/HEIGHT)
        return toTensor(delta_distance_ratio)


    def get_reward_wall(self):
        # Reward from the boundary based on SRL
        x, y = self.x_p / WIDTH, self.y_p / HEIGHT  # normalize to [0,1]
        cos, sin = torch.cos(self.o_p), torch.sin(self.o_p)
        if sin == 0 and cos > 0:
            r = 1 - x + min(y, 1 - y)
        elif sin == 0 and cos < 0:
            r = x + min(y, 1 - y)
        elif cos == 0 and sin > 0:
            r = 1 - y + min(x, 1 - x)
        elif cos == 0 and sin < 0:
            r = y + min(x, 1 - x)
        else:
            a = sin / cos
            b = y - a * x
            min_len1 = min_length_direction(x, y, a, b, cos)  # distance along the walking direction
            a_ = -1 / a
            b_ = y - a_ * x
            min_len2 = min_length(x, y, a_, b_)  # distance vertical to 
            r = min_len1 + min_len2
        return r 

    
    # This method compute the angle error between the person and the target
    def get_reward_angle(self):
        # return self.err_angle() / (PI)
        vec1 = np.array([self.x_t_p-self.x_p, self.y_t_p -self.y_p])
        vec2 = np.array([self.x_t_v - self.x_v, self.y_t_v - self.y_v])

        # vec1 = np.array([np.cos(self.obj_d_p), np.sin(self.obj_d_p)])
        # vec2 = np.array([np.cos(self.obj_d),   np.sin(self.obj_d)])

        vec3 = np.array([np.cos(self.o_p), np.sin(self.o_p)])
        vec4 = np.array([np.cos(self.o_v), np.sin(self.o_v)])
        vec1 = normaliztion(vec1)
        vec2 = normaliztion(vec2)
        ang1 = np.arccos(np.clip(np.dot(vec1, vec3), -1.0, 1.0))
        ang2 = np.arccos(np.clip(np.dot(vec2, vec4), -1.0, 1.0))
        # num1 = np.dot(vec1, vec3)
        # num2 = np.dot(vec2, vec4)
        if np.cross(vec1, vec3) * np.cross(vec2, vec4) < 0:
            ang = delta_angle_norm(ang1 + ang2)
        else:
            ang = delta_angle_norm(ang1 - ang2)

        return abs(ang)/PI



    '''
    scale the angle into 0-pi
    '''
def delta_angle_norm(x):
    if x >= PI:
        x = 2 * PI - x
    elif x <= -PI:
        x = x + 2 * PI
    return x

def normaliztion(x):
    if x[0]*x[0] + x[1] * x[1] == 0:
        return x
    return x /np.sqrt(x[0]*x[0] + x[1] * x[1])

def toTensor(x):
    return torch.Tensor([x])

def toPI(x):
    return x * PI / 180.0

"""
This is the rdw environment
"""


def split(action):
    a, b, c = action[0]
    a = 1.060 + 0.2   * a
    b = 1.145 + 0.345 * b
    c = 0.13 * c
    return a, b, c



def initialize(seed):
    xt, yt, dt = 0, 0, 0
    if seed == 1:
        xt = 0
        yt = random.random() * HEIGHT
        # dt = random.random() * PI - PI / 2.
        dt = 0
    elif seed == 2:
        xt = random.random() * WIDTH
        yt = HEIGHT
        # dt = -random.random() * PI
        dt = -PI / 2.0
    elif seed == 3:
        xt = WIDTH
        yt = random.random() * HEIGHT
        # dt = random.randint(0, 1) * 1.5 * PI + random.random() * PI / 2. - PI
        dt = -PI
    elif seed == 4:
        xt = random.random() * WIDTH
        yt = 0
        # dt = random.random() * PI
        dt = PI / 2

    return xt, yt, dt




def norm(theta):
    if theta < -PI:
        theta = theta + 2 * PI
    elif theta > PI:
        theta = theta - 2 * PI
    return theta


def outbound(x, y):
    if x <= 0 or x >= WIDTH or y <= 0 or y >= HEIGHT:
        return True
    else:
        return False


def min_length_direction(x, y, a, b, cos):  # cause the heading has the direction
    p1 = torch.Tensor([0, b])
    # p2 = np.array([1,a+b])
    p2 = torch.Tensor([1, a + b])
    # p3 = np.array([-b/a,0])
    p3 = torch.Tensor([-b / a, 0])
    # p4 =np.array([(1-b)/a,1])
    p4 = torch.Tensor([(1 - b) / a, 1.])
    p = torch.cat((p1, p2, p3, p4))
    # p = np.concatenate((p1,p2,p3,p4),axis=0)
    p = p.reshape((4, 2))
    p = p[p[:, 0].argsort(), :]
    if cos > 0:
        c, d = p[2]
    else:
        c, d = p[1]
    len = distance(x, y, c, d)
    # len = min(distance(x, y, c, d), distance(x, y, e, f))
    return len


def min_length(x, y, a, b):  # min length of the line y = ax+b with intersection with the bounding box of [0,1]
    p1 = torch.Tensor([0, b])
    # p2 = np.array([1,a+b])
    p2 = torch.Tensor([1, a + b])
    # p3 = np.array([-b/a,0])
    p3 = torch.Tensor([-b / a, 0])
    # p4 =np.array([(1-b)/a,1])
    p4 = torch.Tensor([(1 - b) / a, 1.])
    p = torch.cat((p1, p2, p3, p4))
    # p = np.concatenate((p1,p2,p3,p4),axis=0)
    p = p.reshape((4, 2))
    p = p[p[:, 0].argsort(), :]
    c, d = p[1]
    e, f = p[2]
    return min(distance(x, y, c, d), distance(x, y, e, f))


def distance(x, y, a, b):
    # return np.sqrt(np.square(x-a)+np.square(y-b))
    # return torch.sqrt((x - a).pow(2) + (y - b).pow(2))
    return math.sqrt((x - a) * (x - a) + (y - b) * (y - b))


def toTensor(x):
    return torch.Tensor(x)
class RunningStats:

    def __init__(self):
        self.n = 0
        self.old_m = 0
        self.new_m = 0
        self.old_s = 0
        self.new_s = 0

    def clear(self):
        self.n = 0

    def push(self, x):
        self.n += 1

        if self.n == 1:
            self.old_m = self.new_m = x
            self.old_s = 0
        else:
            self.new_m = self.old_m + (x - self.old_m) / self.n
            self.new_s = self.old_s + (x - self.old_m) * (x - self.new_m)

            self.old_m = self.new_m
            self.old_s = self.new_s

    def mean(self):
        return self.new_m if self.n else 0.0

    def variance(self):
        return self.new_s / (self.n - 1) if self.n > 1 else torch.Tensor([0.0])
