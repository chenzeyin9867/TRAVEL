# script used to generate multipule-target in virtual and physical space
from math import pi
import numpy as np
import argparse
import os
import sys
import matplotlib.pyplot as plt
import random
from tqdm import trange
from random import choice

# from dataset_gen import Epoch

# target object in virtual and physical space
# info includes x, y and radius of this object
class obj:
    def __init__(self, x, y, r):
        self.x = x
        self.y = y
        self.r = r
# virtual space configure
# v_height, v_width = 12.0, 17.0
v_height, v_width = 8.0, 8.0
# v_height, v_width = 10.0, 10.0
# v_obj_list = [obj(10.0, 10.0, 0.5)]
# v_obj_list = [obj(2.0, 2.0, 0.5), obj(8.0, 2.0, 0.5), obj(2.0, 9.0, 0.5), obj(8.0, 9.0, 0.5)]
# v_obj_list = [obj(2.5, 2.5, 0.5), obj(7.5, 2.5, 0.5), obj(2.5, 7.5, 0.5), obj(7.5, 7.5, 0.5)]
# v_obj_list = [obj(5.0, 5.0, 0.5), obj(15, 5, 0.5), obj(5, 15, 0.5), obj(15, 15, 0.5)]
v_obj_list = [obj(1.0, 1.0, 0.5)]
# v_obj_list = [obj(3.0, 2.0, 0.5), obj(3.0, 10.0, 0.5), obj(14.0, 2.0, 0.5)]
# physical space configure
p_height, p_width = 10.0, 10.0
p_obj_list = [obj(3.0, 3.0, 0.5), obj(7.0,3.0, 0.5), obj(5.0, 7.0, 0.5)]


# virtual avator configure, x, y, orientation
pos_list = [[0, v_height/2, 0], [v_width/2, 0, pi/2], [v_width, v_height/2, pi], [v_width/2, v_height, -pi/2]]



velocity = 1.0 / 50.0
frame_rate = 50
step_low = int(0.5 / velocity)
step_high = int(3.5 / velocity)


parser = argparse.ArgumentParser(description="training path generation.")
parser.add_argument('--mode', type=str, default='train', help='training or evaluation')
args = parser.parse_args()
print(args)
mode = args.mode

def init_avator(i):
    # seed = np.random.randint(0, 4)
    # print("Epocd:", i, " seed:", seed)
    # x, y, o = pos_list[1]
    x, y, o  = v_width / 2, v_height / 2, random.random() * pi * 2 - pi
    return x, y, o

# normalize the theta into [-PI,PI]
def norm(theta):
    if theta <= -pi:
        theta = theta + 2 * pi
    elif theta > pi:
        theta = theta - 2 * pi
    return theta

def outbound(x, y):
    return x < 0 or x > v_width or y < 0 or y > v_height

if __name__ == '__main__':
    result = []
    len_ = []
    dir = os.path.join("./waypoint/h8w8_1", "h" + str(int(v_height))+'w'+str(int(v_width)))
    if not os.path.exists(dir):
        os.makedirs(dir)    
    pathnum = 500 if mode == 'eval' else 50000
    Epoch = 0
    # while Epoch < pathnum:
    #     if Epoch % 100 == 0:
    #         print(Epoch)
    for Epoch in trange(pathnum):
        x, y, o = init_avator(Epoch)
        x_s, y_s, o_s = x, y, o
        x_list ,y_list ,o_list, o_delta, tourch_list = [], [], [], [], []
        obj_set = [i for i in range(len(v_obj_list)) ]
        # x_list.append(x)
        # y_list.append(y)
        # o_list.append(o)
        # o_delta.append(0)
        tourch_list.append(-1)
        iter = 0
        delta_direction_per_iter = 0
        num_change_direction = 0
        turn_flag = np.random.randint(step_low, step_high)
        current_obj = -1
        collide = 0
        choose_obj = 0
        for t in range(3600):
            if turn_flag == 0:
                # print("Len:", len(obj_set))
                if len(obj_set) == 0:
                    break
                rd = random.randint(0,1)
                # print(rd)
                if rd != 1:
                    turn_flag = np.random.randint(step_low, step_high)
                    delta_direction = np.clip(random.normalvariate(0, 45), -180, 180)
                    delta_direction = np.random.random() * 2 * pi - pi
                    delta_direction_per_iter = 1.5 * pi / 180.0
                    if delta_direction < 0:
                        delta_direction_per_iter = - delta_direction_per_iter
                    num_change_direction = int(abs(delta_direction / delta_direction_per_iter))
                    current_obj = -1
                    # x_list.append(x + )
                else:
                    # choose a unselected obj
                    obj_ind = choice(obj_set)
                    # print(obj_ind)
                    obj_set.remove(obj_ind)
                    
                    tar_x, tar_y = v_obj_list[obj_ind].x, v_obj_list[obj_ind].y
                    delta_direction = norm(np.arctan2(tar_y - y, tar_x - x) - o)
                    random_radius = 0.5
                    # num_change_direction = (delta_direction * random_radius / velocity) if random_radius != 0 else 1
                    # delta_direction_per_iter = delta_direction / num_change_direction
                    
                    delta_direction_per_iter = 1.5 * pi / 180.0
                    if delta_direction < 0:
                        delta_direction_per_iter = -delta_direction_per_iter
                    num_change_direction = int(delta_direction / delta_direction_per_iter) + 1000
                    # dis = np.sqrt((tar_x-x)*(tar_x-x) + (tar_y - y) * (tar_y - y))
                    turn_flag = 100000
                    current_obj = obj_ind
                
            if num_change_direction > 0:
                o = norm(o + delta_direction_per_iter)
                num_change_direction = num_change_direction - 1
                if current_obj != -1:
                    tar_x, tar_y = v_obj_list[current_obj].x, v_obj_list[current_obj].y
                    if abs(norm(np.arctan2(tar_y - y, tar_x - x) - o)) < (pi / 180):
                        num_change_direction = 0
                        dis = np.sqrt((tar_x-x)*(tar_x-x) + (tar_y - y) * (tar_y - y))
                        turn_flag = int(dis / velocity)
                        # continue
            else:
                turn_flag = turn_flag - 1
                delta_direction_per_iter = 0
            x = x + velocity * np.cos(o)
            y = y + velocity * np.sin(o)
            if outbound(x, y):
                break
            x_list.append(x)
            y_list.append(y)
            o_list.append(o)
            o_delta.append(delta_direction_per_iter)
            if current_obj != -1 and turn_flag == 0:
                tourch_list.append(current_obj)
                # break
            else:
                tourch_list.append(-1)
        if collide == 1:
            continue
        Epoch += 1
        x_np = np.array(x_list)
        y_np = np.array(y_list)
        o_np = np.array(o_list)
        o_delta_np = np.array(o_delta)
        touch_np = np.array(tourch_list)
        stack_data = np.stack((x_np, y_np, o_np, o_delta_np, touch_np), axis=-1)
        result.append(stack_data)
        len_.append(t)
        #
        plt.figure(figsize=(5,5))
        if Epoch < 50 and mode == 'eval':
            plt.axis('scaled')
            plt.xlim(0.0, v_width)
            plt.ylim(0.0, v_height)
            plt.yticks([])
            plt.xticks([])
            # dst = str(Epoch) + '.pdf'
            dst1 = str(Epoch) + '.png'
            plt.scatter(x_s, y_s, s = 50, color='b', label="origin")
            # plt.scatter(x_list, y_list, s=0.5, c=[t for t in range(len(x_list))], cmap="Reds", alpha = 0.2)
            plt.scatter(x_list, y_list, s=0.5, cmap="Reds", alpha = 0.2)
            for obj_ind in range(len(v_obj_list)):
                plt.scatter(v_obj_list[obj_ind].x, v_obj_list[obj_ind].y, color='gold')
                plt.scatter(v_obj_list[obj_ind].x, v_obj_list[obj_ind].y, s=200, marker="o", label='target object', edgecolors='orange', linewidths=0.5, )
            ax = plt.gca()
            ax.spines['bottom'].set_linewidth(1.5)
            ax.spines['left'].set_linewidth(1.5)
            ax.spines['right'].set_linewidth(1.5)
            ax.spines['top'].set_linewidth(1.5)
            plt.legend()
            plot_dir = os.path.join(dir, "visualize")
            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir)
            plt.savefig(os.path.join(plot_dir, dst1))
            plt.close()

    save_np = np.array(result)
    np.save(os.path.join(dir, "train.npy" if mode=="train" else "eval.npy"), save_np)
    print(np.mean(len_), np.std(len_))
    sys.exit() 


    
