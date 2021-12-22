import os
import gym
import numpy as np
import torch
from gym.spaces.box import Box
from envs.distributions import FixedNormal
from math import pi
import random
import matplotlib.pyplot as plt
import math
from running_mean_std import RunningMeanStd

REWARD_SCALE = False
VELOCITY = 1.4 / 60.0
HEIGHT, WIDTH = 10 , 10
HEIGHT_ALL, WIDTH_ALL = 10.0, 10.0
FRAME_RATE = 60
PI = np.pi
STEP_LOW = int(0.5 / VELOCITY)
STEP_HIGH = int(3.5 / VELOCITY)
DELTA_X = (WIDTH_ALL - WIDTH) / 2.0
DELTA_Y = (HEIGHT_ALL - HEIGHT) / 2.0
PENALTY = torch.Tensor([1.0])
OBSERVATION_SPACE = 6


class PassiveHapticsEnv(object):
    def __init__(self, gamma, num_frame_stack, path, random=False, eval=False, obs_norm = False):
        self.configure_space()
        self.distance = []
        self.action_repeat = 10
        self.num_frame_stack = num_frame_stack  # num of frames stcked before input to the MLP
        self.gamma = gamma                      # γ used in PPO
        self.observation_space = Box(-1., 1., (num_frame_stack * OBSERVATION_SPACE,))
        self.action_space = Box(-1.0, 1.0, (3,))
        self.obs = []                   # current stacked infos send to the networks
        self.obs_norm = obs_norm

        self.time_step = 0              # count of the time step
        self.direction_change_num = 0
        self.delta_direction_per_iter = 0
        self.reward = 0.0
        self.path_cnt = 0               # path index for current iteration

        self.r1, self.r2, self.r3 = [], [], []

        if not eval:
            # print("Loading the training dataset")
            self.path_file = np.load(os.path.join(path, 'train.npy'), allow_pickle=True)
        else:
            # print("Loading the eval dataset")
            self.path_file = np.load(os.path.join(path, 'eval.npy'),  allow_pickle=True)
        self.v_path = self.path_file[self.path_cnt]
        self.v_step_pointer = 0         # the v_path counter


    # configure the physical and virtual space
    def configure_space(self):
        # virtual space configure
        
        self.v_list = [obj(2.5, 2.5, 0.5), obj(7.5,2.5, 0.5), obj(2.5,7.5, 0.5), obj(7.5, 7.5, 0.5)]
        # physical space configure
        # p_height, p_width = 10.0, 10.0
        self.p_list = [obj(3.0, 3.0, 0.5), obj(6.0,3.0, 0.5), obj(5.0, 7.0, 0.5)]
        # virtual avator configure, x, y, orientation
        self.pos_list = [[0, HEIGHT_ALL/2, 0], [WIDTH_ALL/2, 0, pi/2], 
                        [WIDTH_ALL, HEIGHT_ALL/2, pi], [WIDTH_ALL/2, HEIGHT_ALL, -pi/2]]
    

    def get_obs(self):

        state = [ self.x_p/WIDTH,        self.y_p/HEIGHT,       (self.o_p + PI)   /(2*PI),                     # physical observation
                  self.x_v/WIDTH_ALL,    self.y_v/HEIGHT_ALL,   (self.o_v + PI)   /(2*PI),                     # virtual  observation
                #   self.delta_direction_per_iter                        # eye direction             
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
        seed = random.randint(0, 3)
        self.x_v, self.y_v, self.o_v, self.delta_direction_per_iter, _ = self.v_path[self.v_step_pointer]                                      # initialize the user in virtual space
        self.x_p, self.y_p, self.o_p = self.pos_list[seed]
        self.obs.extend(self.num_frame_stack * self.get_obs())
        self.reward = 0.0
        self.touch, self.target_p = -1, -1
        
        return toTensor(self.obs)

    def step(self, action):
        # Step forward for K times using the same action based on action-repeat strategy.
        gt, gr, gc = split(action)
        k = self.action_repeat          # action repetition
        reward = torch.Tensor([0.0])    # initial reward for this step period
        for ep in range(k):             # for every iter, get the virtual path info, and steering
            self.vPathUpdate()
            signal = self.physical_step(gt, gr, gc)  # steering the physical env using the actions
            self.time_step += 1
            
            if not signal: # Collide the boundary 
                reward = -PENALTY
                break
            
            if self.v_step_pointer >= len(self.v_path) - 1: # Ends the v_path
                break
            if self.touch != -1:
                reward, _ = self.final_reward()                            
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
            self.reset()
            return obs, ret_reward, [1], [0]
        else:
            return obs, ret_reward, [0], [0]

    def vPathUpdate(self):
        self.x_v, self.y_v, self.o_v, self.delta_direction_per_iter, self.touch = \
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
            # self.o_p = norm(self.o_p + PI)
            self.o_p = self.compute_dir_to_obj()
            return False
        else:
            self.x_p = tmp_x
            self.y_p = tmp_y
            return True


    """
    when in eval mode, initialize the user's postion
    """
    def init_eval_state(self, ind, evalType=0):
        # print(ind)
        if evalType == 0:
            m = (int(ind / 5)) % 4
            n = (4 + ind % 22)/30.0 + 0.05
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
        # self.o_p = self.compute_dir_to_obj()


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

    def step_specific_path(self, actor_critic, running_mean_std, ind, gain=True):
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

        self.x_v, self.y_v, self.o_v, self.delta_direction_per_iter, _ = self.v_path[self.v_step_pointer]
        seed = ind % 4
        self.x_p, self.y_p, self.o_p = self.pos_list[seed]
        
        i = self.v_step_pointer
        touch_cnt = 0
        gap_dis = 0.0
        with torch.no_grad():
            while i < len(self.v_path):
                if gain:
                    
                    obs = torch.Tensor(self.obs).unsqueeze(0)
                    if self.obs_norm:
                        obs = running_mean_std.process(obs)
                    _value, action_mean, action_log_std = actor_critic.act(
                        obs)
                    dist = FixedNormal(action_mean, action_log_std)
                    action = dist.mode()

                    gt, gr, gc = split(action)
                    std1.append(action_log_std[0].item())
                    std2.append(action_log_std[1].item())
                    std3.append(action_log_std[2].item())
                    gt_l.append(gt.item())
                    gr_l.append(gr.item())
                    gc_l.append(gc.item())
                else:
                    gr, gt, gc = torch.Tensor([1.0]), torch.Tensor([1.0]), torch.Tensor([0.0])
                    
                    
                for m in range(self.action_repeat):
                    if i > len(self.v_path) - 1:
                        signal = False
                        # self.reward += self.final_reward()
                        break
                    signal = self.physical_step(gt, gr, gc)
                    # signal = self.physical_step_eval(gt, gr, gc)
                    self.vPathUpdate()
                    x_l.append(self.x_p)
                    y_l.append(self.y_p)
                    i += 1
                    if not signal:
                        self.reward -= PENALTY
                        collide += 1
                        # signal = True
                        break
                    if self.touch != -1:
                        r, gap = self.final_reward()
                        self.reward += r
                        gap_dis += gap 
                        touch_cnt += 1
                    elif m == 0:
                        self.reward += self.get_reward()
                if not signal:
                    break
                init_obs = self.obs[OBSERVATION_SPACE:]
                init_obs.extend(self.get_obs())
                self.obs = init_obs
        vx_l = self.v_path[:, 0]
        vy_l = self.v_path[:, 1] 
        ave_gap_dis = gap_dis # average pde
            
        return self.reward, ave_gap_dis, touch_cnt, 1.0, gt_l, gr_l, gc_l, x_l, y_l, vx_l, vy_l, std1, std2, std3, collide

    def err_angle(self):
        # return abs(self.o_x - self.o_t_p) * 180.0 / PI
        # return abs(self.get_reward_angle())*PI
        return abs(delta_angle_norm(self.o_p - self.o_t_p)) * 180.0 / PI

    def final_reward(self):
        x_t, y_t  = self.p_list[self.target_p].x, self.p_list[self.target_p].y
        # r = 10 * (1 - math.pow(distance(self.x_p/WIDTH, self.y_p/HEIGHT, self.x_t_p/WIDTH, self.y_t_p/HEIGHT), 1))
        gap_dis = distance(self.x_p/WIDTH, self.y_p/HEIGHT, x_t/WIDTH, y_t/HEIGHT) - self.p_list[self.target_p].r/WIDTH 
        r = (1 - gap_dis)
        # print(distance(self.x_p/WIDTH, self.y_p/HEIGHT, self.x_t_p/WIDTH, self.y_t_p/HEIGHT), 0.5 * self.err_angle()/180)
        # return r 
        # r = torch.Tensor([0.0])
        return r, gap_dis

    def get_reward(self):
        # d_wall = min(self.x_p/WIDTH, (WIDTH-self.x_p)/WIDTH, (self.y_p)/HEIGHT, (HEIGHT-self.y_p)/HEIGHT)
        # r_d = self.get_reward_distance()
        r_avoid = self.get_reward_wall()
        # r_p = self.get_reward_position_orientation()
        
        r_align = self.compute_map_function()
        
        
        # print("%3f" % (r3))
        # self.r1.append(r_avo)
        # self.r2.append(r_p)
        # if(len(self.r1) == 1000):
            # print("rd:%.4f  rp:%.4f" % (np.mean(self.r1), np.mean(self.r2)))    
        return (1.0 + 0.3 * r_avoid - r_align)/ 50
        # return torch.Tensor([0.0])

    def compute_map_function(self):
        # compute each (V_obj, X_obj) pair's propotional to the final reward
        min_gap = 10
        for i in range(len(self.v_list)):
            for j in range(len(self.p_list)):
                d = self.get_reward_distance(i, j)
                o = self.get_reward_orientation(i, j)
                if (3 * o + d) < min_gap:
                    min_gap = 3 * o + d
                    # m_p, m_v = j, i
                    self.target_p = j # current physical obj
        return 3 * o + d
                    
                
        # The d smaller, the propotion larger
        
                
                

    def print(self):
        print("physical:", self.x_p, " ", self.y_p, " ", self.o_p)
        print("virtual:", self.x_v, " ", self.y_v, " ",  self.o_v)

    def get_reward_distance(self, v, p):
        d1 = distance(self.x_p/WIDTH,     self.y_p/HEIGHT,     self.p_list[p].x/WIDTH,     self.p_list[p].y/  HEIGHT)
        d2 = distance(self.x_v/WIDTH_ALL, self.y_v/HEIGHT_ALL, self.v_list[v].x/WIDTH_ALL, self.v_list[v].y/  WIDTH_ALL)
        return toTensor(abs(d1 - d2))


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
    def get_reward_orientation(self, v, p):
        # return self.err_angle() / (PI)
        obj_p_x, obj_p_y = self.p_list[p].x, self.p_list[p].y
        obj_v_x, obj_v_y = self.v_list[v].x, self.v_list[v].y

        ang1 = np.arctan2((obj_p_y - self.y_p), (obj_p_x - self.x_p)) - self.o_p
        ang2 = np.arctan2((obj_v_y - self.y_v), (obj_v_x - self.x_v)) - self.o_v
        # ang_orientation = delta_angle_norm(ang3 + ang4) if np.cross(vec3, vec5) * np.cross(vec4, vec6) < 0 else delta_angle_norm(ang3-ang4)
        # return abs(ang_pos)/PI, abs(ang_orientation)/PI
        return abs(delta_angle_norm(ang1 - ang2))/PI
    # This method scale the orientation error
    # def get_reward_orientation(self):
        


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
    # a = 1.060 + 0.2   * a
    # b = 1.145 + 0.345 * b
    # c = 0.13 * c
    a = 1 + 0.5 * a
    b = 1 + 0.5 * b
    c = 0.5 * c 
    return a, b, c

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

class obj:
    def __init__(self, x, y, r):
        self.x = x
        self.y = y
        self.r = r