import math
import os
import matplotlib.pyplot as plt
from tqdm import trange
from envs.envs_general import *

def drawPath(vx_l, vy_l, x_none_l, y_none_l, x_l, y_l, env, args, epoch):
    '''
        :vx, vy: virtual path
        :x_, y_: redirected path without gains
        :x,  y : redirected path using   phrl
    '''
    draw = args.draw
    name = args.env_name
    for t in range(len(vx_l)):
        vx, vy, x_, y_, x, y = vx_l[t], vy_l[t], x_none_l[t], y_none_l[t], x_l[t], y_l[t]
        if draw and t < 100 and epoch % 500 == 0:
                plt.figure(1, figsize=(10, 5))
                plt_srl = plt.subplot(1, 2, 2)
                plt_none = plt.subplot(1, 2, 1)
                plt_none.set_title('virtual')
                plt_srl.set_title('physical')
                plt_srl.axis('scaled')
                plt_srl.axis([0.0, WIDTH, 0.0, HEIGHT])
                plt_none.axis('scaled')
                plt_none.axis([0.0, WIDTH_ALL, 0.0, HEIGHT_ALL])

                plt_srl.scatter(x,   y,  label='SRL',     s=1, c='r')
                plt_srl.scatter(x_,  y_, label="None",    s=1, c='g')
                plt_none.scatter(vx, vy, label="virtual", s=1, c='b')
                plt_srl.legend()
                
                for obj_ind in range(len(env.v_list)):
                    plt_none.scatter(env.v_list[obj_ind].x, env.v_list[obj_ind].y, color='gold')
                    plt_none.scatter(env.v_list[obj_ind].x, env.v_list[obj_ind].y, s=200, marker="o", label='target object', edgecolors='orange', linewidths=0.5, )
                for obj_ind in range(len(env.p_list)):
                    plt_srl.scatter(env.p_list[obj_ind].x, env.p_list[obj_ind].y, color='gold')
                    plt_srl.scatter(env.p_list[obj_ind].x, env.p_list[obj_ind].y, s=200, marker="o", label='target object', edgecolors='orange', linewidths=0.5, )
   
                
 
                if not os.path.exists('./plot_result/%s/ep_%d' % (name, epoch)):
                    os.makedirs('./plot_result/%s/ep_%d' % (name, epoch))
                plt.savefig('./plot_result/%s/ep_%d/%d.png' % (name, epoch, t))
                plt.clf()
                plt.cla()