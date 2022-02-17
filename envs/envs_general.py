import os
from site import USER_SITE
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
VELOCITY = 1.0 / 50.0
HEIGHT, WIDTH = 8 , 8
HEIGHT_ALL, WIDTH_ALL = 10.0, 10.0
FRAME_RATE = 50
PI = np.pi
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
        # self.v_list = [obj(5.0, 5.0, 0.5), obj(15, 5, 0.5), obj(5, 15, 0.5), obj(15, 15, 0.5)]
        self.v_list = [obj(1.0, 1.0, 0.5)]
        # self.v_list = [obj(2.5, 2.5, 0.5), obj(7.5,2.5, 0.5), obj(2.5,7.5, 0.5), obj(7.5, 7.5, 0.5)]
        # self.v_list = [obj(2.0, 2.0, 0.5), obj(8.0,2.0, 0.5), obj(2.0,8.0, 0.5), obj(8.0, 8.0, 0.5)]
        # self.v_list = [obj(10.0, 10.0, 0.5)]
    
        # physical space configure
        # p_height, p_width = 10.0, 10.0
        # self.p_list = [obj(4.0, 4.0, 0.5), obj(12.0, 4.0, 0.5), obj(8.0, 12.0, 0.5)]
        # self.p_list = [obj(1.5, 4.0, 0.5), obj(6.5, 6.0, 0.5), obj(4, 1.5, 0.5)]
        self.p_list = [obj(7.0, 1.0, 0.5)]
        self.obstacle_list = [obstacle("square", 3.0, 3.0, 2.0)]
        # virtual avator configure, x, y, orientation
        self.pos_list = [[0, HEIGHT_ALL/2, 0], [WIDTH_ALL/2, 0, pi/2], 
                        [WIDTH_ALL, HEIGHT_ALL/2, pi], [WIDTH_ALL/2, HEIGHT_ALL, -pi/2]]
    

    def get_obs(self):

        state = [ (self.x_p - (WIDTH)/2) / (WIDTH/2),        (self.y_p-(HEIGHT/2))/(HEIGHT/2),       (self.o_p) /(PI),                     # physical observation
                  (self.x_v -(WIDTH_ALL/2))/(WIDTH_ALL/2),   (self.y_v - (HEIGHT_ALL/2))/(HEIGHT_ALL/2),   (self.o_v)   /(PI),                     # virtual  observation
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
        self.x_p, self.y_p, self.o_p = WIDTH/ 2, HEIGHT / 2, random.random() * pi * 2 - pi
        self.obs.extend(self.num_frame_stack * self.get_obs())
        self.reward = 0.0
        self.touch, self.target_p = -1, -1
        self.p_prop_list = []
        
        return toTensor(self.obs)

    def step(self, action):
        # Step forward for K times using the same action based on action-repeat strategy.
        gt, gr, gc = split(action)
        k = self.action_repeat          # action repetition
        reward = torch.Tensor([0.0])    # initial reward for this step period
        for ep in range(k):             # for every iter, get the virtual path info, and steering
            self.vPathUpdate()
            signal = self.physical_step(gt, gr, gc)  # steering the physical env using the actions
            # signal = self.physical_step_eval(gt, gr, gc)
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
            self.o_p = norm(self.o_p + PI)
            # self.o_p = self.compute_dir_to_obj()
            self.obs.extend(self.num_frame_stack * self.get_obs())
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
        return WIDTH / 2, HEIGHT / 2, (ind % 100)/ 100 * 2 * pi - pi
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
        pde_list = []
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
        self.x_p, self.y_p, self.o_p = self.init_eval_state(ind)
        # self.init_eval_state(ind)
        
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
                    
                signal = True
                for m in range(self.action_repeat):
                    if i == len(self.v_path):
                        signal = False
                        # self.reward += self.final_reward()
                        break
                    # signal = self.physical_step(gt, gr, gc)
                    if not signal:
                        break
                    signal = self.physical_step_eval(gt, gr, gc)
                    if i < len(self.v_path):
                        self.vPathUpdate()
                    i += 1
                    x_l.append(self.x_p)
                    y_l.append(self.y_p)
                    if not signal:
                        self.reward -= PENALTY
                        collide += 1
                        signal = True
                        break
                    if self.touch != -1:
                        r, gap = self.final_reward()
                        self.reward += r
                        gap_dis += gap 
                        pde_list.append(gap)
                        touch_cnt += 1
                    elif m == 0:
                        self.reward += self.get_reward()
                if not signal:
                    break
                init_obs = self.obs[-OBSERVATION_SPACE * (self.num_frame_stack - 1):]
                init_obs.extend(self.get_obs())
                self.obs = init_obs
        vx_l = self.v_path[:, 0]
        vy_l = self.v_path[:, 1] 
        ave_gap_dis = gap_dis # average pde
            
        return self.reward, ave_gap_dis, pde_list, 1, touch_cnt, gt_l, gr_l, gc_l, x_l, y_l, vx_l, vy_l, std1, std2, std3, collide

    def err_angle(self):
        # return abs(self.o_x - self.o_t_p) * 180.0 / PI
        # return abs(self.get_reward_angle())*PI
        return abs(delta_angle_norm(self.o_p - self.o_t_p)) * 180.0 / PI

    def final_reward(self):
        p_target = self.p_prop_list[int(self.touch)]
        x_t, y_t  = self.p_list[p_target].x, self.p_list[p_target].y
        # r = 10 * (1 - math.pow(distance(self.x_p/WIDTH, self.y_p/HEIGHT, self.x_t_p/WIDTH, self.y_t_p/HEIGHT), 1))
        gap_dis = distance(self.x_p/WIDTH, self.y_p/HEIGHT, x_t/WIDTH, y_t/HEIGHT)
        # gap_ori = abs(delta_angle_norm(np.arctan2(y_t - self.y_p, x_t - self.x_p) - self.o_p )) / pi
        # r = (1 - gap_dis - gap_ori * 0.2)
        # r = (1 - gap_dis)
        # r = 
        # print(distance(self.x_p/WIDTH, self.y_p/HEIGHT, self.x_t_p/WIDTH, self.y_t_p/HEIGHT), 0.5 * self.err_angle()/180)
        # return r 
        r = torch.Tensor([0.0])
        # print(gap_dis,"  ", gap_ori)
        
        # return the distance to the nearest object
        ret_distance = self.nearest_obj()
        
        return r, ret_distance

    def nearest_obj(self):
        min_dis = 1000
        for i in range(len(self.p_list)):
            x_t, y_t = self.p_list[i].x, self.p_list[i].y
            min_dis = min(min_dis, distance(x_t, y_t, self.x_p, self.y_p))
        return min_dis

    def get_reward(self):
        # d_wall = min(self.x_p/WIDTH, (WIDTH-self.x_p)/WIDTH, (self.y_p)/HEIGHT, (HEIGHT-self.y_p)/HEIGHT)
        # r_d = self.get_reward_distance()
        # r_avoid = self.get_reward_wall()
        # r_p = self.get_reward_position_orientation()
        
        r_align = self.compute_map_function()
        
        
        # print("%3f" % (r3))
        # self.r1.append(r_avoid)
        # self.r2.append(r_align)
        # if(len(self.r1) == 5000):
            # print("rd:%.4f  rp:%.4f" % (np.mean(self.r1), np.mean(self.r2)))    
        return (0.5 - r_align)/10
        # return torch.Tensor([0.0])

    def compute_map_function(self):
        # step1: predict the possibility to each virtual object
        # using c.exp() to represent the possibility
        dis_obj_v = []
        ori_obj_v = []
        unalign_obj_v = []
        for i in range(len(self.v_list)):
            dis_obj_v.append(distance(self.x_v/WIDTH_ALL, self.y_v/HEIGHT_ALL, self.v_list[i].x/WIDTH_ALL, self.v_list[i].y/HEIGHT_ALL)) # dis from v_avator to each v_obj
            ori_obj_v.append(abs(delta_angle_norm(self.o_v - np.arctan2(self.v_list[i].y - self.y_v, self.v_list[i].x - self.x_v))))
            unalign_obj_v.append(torch.exp(toTensor(-(dis_obj_v[i] + 3 * ori_obj_v[i]))))
        
        # possibility to goto each V_obj 
        sum_possibility = sum(unalign_obj_v)
        possibility_list = [x / sum_possibility.item() for x in unalign_obj_v]
        
        # choose physical prop for each virt prop
        self.p_prop_list = [0 for _ in range(len(self.v_list))]
        penalty = 0
        for i in range(len(self.v_list)):
            unalign_min = 100
            p_obj_ind = 0
            for j in range(len(self.p_list)):
                d = self.get_reward_distance(i, j)
                o = self.get_reward_orientation(i, j)
                intersect = self.get_reward_intersect(j)
                # print(o, d)
                if (o + 3 * d + intersect) < unalign_min:
                    unalign_min = o + 3 * d + intersect
                    p_obj_ind = j
            self.p_prop_list[i] = p_obj_ind
            penalty += unalign_min * possibility_list[i]
        
        return penalty
        
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
        return r / np.sqrt(2)

    
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
    
    
    def get_reward_intersect(self, j):
        prop = self.p_list[j]
        if self.isIntersectProp(prop):
            return 1.0
        else:
            return 0.0
        

    def isIntersectProp(self, prop):
        flag = False
        for obstacle in self.obstacle_list:
            if self.isIntersectObstacle(self, prop, obstacle):
                return True
        return False
            
    def isIntersectObstacle(self, prop, obstacle):
        # whether two linesegment intersect
        if obstacle.m_name == "square": # test each edge
            heading_seg = Segment(Point(self.x_p, self.y_p), Point(prop.x, prop.y))
            left_b  = Point(obstacle.x - obstacle.r, obstacle.y - obstacle.r)
            left_u  = Point(obstacle.x - obstacle.r, obstacle.y + obstacle.r)
            right_u = Point(obstacle.x + obstacle.r, obstacle.y + obstacle.r)
            right_b = Point(obstacle.x + obstacle.r, obstacle.y - obstacle.r)
            # Four segment
            l_seg = Segment(left_u, left_b)
            r_seg = Segment(right_u, right_b)
            u_seg = Segment(left_u, right_u)
            b_seg = Segment(left_b, right_b)
            return heading_seg.is_cross(l_seg) or heading_seg.is_cross(r_seg) or heading_seg.is_cross(u_seg) or heading_seg.is_cross(b_seg)
        
    

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
    b = 0.955 + 0.285 * b
    c = 0.13 * c
    # a = 1 + 0.5 * a
    # b = 1 + 0.5 * b
    # c = 0.5 * c 
    return a, b, c

def norm(theta):
    if theta < -PI:
        theta = theta + 2 * PI
    elif theta > PI:
        theta = theta - 2 * PI
    return theta


def outbound(x, y):
    # if x <= 0 or x >= WIDTH or y <= 0 or y >= HEIGHT or (x >= 6  and x <= 10 and y >= 0 and y <= 4):
    #     return True
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
        

# Obstacle in environment, have shape of rectangle and circle
class obstacle:
    def __init__(self, m_type, x, y, r):
        m_name = m_type
        m_x = x
        m_y = y
        m_r = r


def multiply(v1, v2):  
    """ 
    计算两个向量的叉积 
    """  
    return v1.x*v2.y - v2.x*v1.y  
  
  
class Point:  
    def __init__(self, x, y):  
        self.x = x  
        self.y = y  
  
    def __sub__(self, other):  
        """ 
        重载减法运算，计算两点之差，实际上返回的表示该点到另一点的向量 
        :param other: 另一个点 
        :return: 该点到另一点的向量 
        """  
        return Point(self.x-other.x, self.y-other.y)  
  
  
class Segment:  
    def __init__(self, point1, point2):  
        self.point1 = point1  
        self.point2 = point2  
  
    def straddle(self, another_segment):  
        """ 
        :param another_segment: 另一条线段 
        :return: 如果另一条线段跨立该线段，返回True；否则返回False 
        """  
        v1 = another_segment.point1 - self.point1  
        v2 = another_segment.point2 - self.point1  
        vm = self.point2 - self.point1  
        if multiply(v1, vm) * multiply(v2, vm) <= 0:  
            return True  
        else:  
            return False  
  
    def is_cross(self, another_segment):  
        """ 
        :param another_segment: 另一条线段 
        :return: 如果两条线段相互跨立，则相交；否则不相交 
        """  
        if self.straddle(another_segment) and another_segment.straddle(self):  
            return True  
        else:  
            return False  
  
    def cross(self, another_segment):  
        if self.is_cross(another_segment):  
            print('两线段相交.')  
        else:  
            print('两线段不相交.')  
        plt.plot([self.point1.x, self.point2.x], [self.point1.y, self.point2.y])  
        plt.plot([another_segment.point1.x, another_segment.point2.x],  
                 [another_segment.point1.y, another_segment.point2.y])  
        plt.show()  

