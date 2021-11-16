from os import POSIX_FADV_WILLNEED, popen
import numpy as np
import torch
import time
from envs import myutils
import argparse
# from envs.envs import *
from envs.envs_general import *
from tqdm import trange
from matplotlib.backends.backend_pdf import PdfPages

import numpy as np



def phrlEvaluate(actor_critic, epoch, **kwargs):
    gamma, stack_frame_num, path = kwargs['gamma'], kwargs['stack_frame'], kwargs['data']
    env = PassiveHapticsEnv(gamma,  stack_frame_num, path,  eval=True)
    reward      = 0
    collide     = 0
    pde         = 0     # Physical Distance    Error
    poe         = 0     # Physical Orientation Error
    std_list1   = []
    std_list2   = []
    std_list3   = []
    gt_list     = []
    gr_list     = []
    gc_list     = []
    num         = kwargs['test_frames']
    draw        = kwargs['draw']
    evalType    = kwargs['path_type']
    env.reset()
    for t in trange(0, num):
        env.reset()
        if actor_critic != None:
            ret, pe, oe, gt, gr, gc, x, y, vx, vy, std1, std2, std3, c = env.step_specific_path(actor_critic, t, evalType)
            std_list1.extend(std1)
            std_list2.extend(std2)
            std_list3.extend(std3)
            gt_list.extend(gt)
            gr_list.extend(gr)
            gc_list.extend(gc)

        else: # None
            ret, pe, oe, x, y, vx, vy,c = env.step_specific_path_nosrl(t, evalType)    
  
        reward += ret
        collide += c     
        pde += pe
        poe += oe # orientation error

        if draw and t < num / 10.0:
            plt.figure(1, figsize=(10, 5))
            title = "SRL" if actor_critic else "None"
            plt_srl = plt.subplot(1, 2, 2)
            plt_none = plt.subplot(1, 2, 1)
            plt_none.set_title('virtual')
            plt_srl.set_title('physical')
            plt_srl.axis('scaled')
            plt_srl.axis([0.0, WIDTH, 0.0, HEIGHT])
            plt_none.axis('scaled')
            plt_none.axis([0.0, WIDTH_ALL, 0.0, HEIGHT_ALL])

            plt_srl.scatter(np.array(x), np.array(y), label='SRL', s=1, c='r')
            plt_none.scatter(np.array(vx), np.array(vy), s=1, c='b')
            plt_srl.legend()
            plt_srl.scatter(env.x_t_p, env.y_t_p, s=10)
            plt_none.scatter(env.x_t_v, env.y_t_v, s= 10)
            # plt_noSRL.scatter(WIDTH / 2.0, HEIGHT / 2.0, s=10)
            name = {1: "same_edge", 2: "random", 3: "shuffle"}
            if epoch is None:
                if not os.path.exists('./plot_result/general'):
                    os.makedirs('./plot_result/general')
                plt.savefig('./plot_result/general/' + name[evalType] + '_' + str(t) + '.png')
                plt.clf()
                plt.cla()
            elif epoch % 100 == 0:
                if not os.path.exists('./plot_result/%s/ep_%d' % (kwargs['env_name'], epoch)):
                    os.makedirs('./plot_result/%s/ep_%d' % (kwargs['env_name'], epoch))
                plt.savefig('./plot_result/%s/ep_%d/%s_%d.png' % (kwargs['env_name'], epoch, title, t))
            plt.clf()
            plt.cla()

    reward = reward / num
    pde    = pde    / num
    poe    = poe    / num
    std1   = np.mean(std_list1).item()
    std2   = np.mean(std_list2).item()
    std3   = np.mean(std_list3).item()
    gt     = np.mean(gt_list).item()
    gr     = np.mean(gr_list).item()
    gc     = np.mean(gc_list).item()
    return reward, pde, poe, collide, std1, std2, std3, gt, gr, gc 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=10,
        help='log interval, one log per n updates (default: 10)')
    parser.add_argument(
        '--env-name',
        default='PongNoFrameskip-v4',
        help='environment to train on (default: PongNoFrameskip-v4)')
    parser.add_argument(
        '--load-dir',
        default='./trained_models/',
        help='directory to save agent logs (default: ./trained_models/)')
    parser.add_argument(
        '--non-det',
        action='store_true',
        default=False,
        help='whether to use a non-deterministic policy')
    parser.add_argument(
        '--load_param',
        default='trained_models/ppo/18_rdw_731.pt',
        type=str,
        help='the pth file you need to load'
    )
    args = parser.parse_args()
    param_name = os.path.join(args.load_param)
    print('loading the ————', param_name)

    actor_critic = torch.load(param_name)
    num = 5000
    # draw = True
    draw = False
    print('running ', num, " paths:")
    # reward, r_none, distance_physical, dis_nosrl, angle_srl, angle_none, flag, r_1, r_2,std_list1, std_list2, std_list3, gt_list, gr_list, gc_list, collide, collide_ = PassiveHapticRdwEvaluate(actor_critic, args.seed,
    #                                      1, 0.99, None, None, 10, None, 0, args.env_name, False, num, draw, evalType=3)
    # print("TYPE1:")
    # print("SRL:\tdistance:{:.2f}\t{:2f}|{:.2f}|{:.2f}\tanglr:{:.2f}\t".format(distance_physical/num, r_1[0], r_1[int(num/2)], r_1[num-1], angle_srl.item()/num))
    # print("None:\tdistance:{:.2f}\t{:2f}|{:.2f}|{:.2f}\tanglr:{:.2f}\t".format(dis_nosrl/num, r_2[0], r_2[int(num/2)], r_2[num-1], angle_none.item()/num))

    # reward, r_none, distance_physical, dis_nosrl, angle_srl, angle_none, flag, r_1, r_2,std_list1, std_list2, std_list3, gt_list, gr_list, gc_list, collide, collide_ = PassiveHapticRdwEvaluate(actor_critic, args.seed,
    #                                      1, 0.99, None, None, 10, None, 0, args.env_name, False, num, draw, evalType=1)
    # print("TYPE2:")
    # print("SRL:\tdistance:{:.2f}\t{:2f}|{:.2f}|{:.2f}\tanglr:{:.2f}\t".format(distance_physical/num, r_1[0], r_1[int(num/2)], r_1[num-1], angle_srl.item()/num))
    # print("None:\tdistance:{:.2f}\t{:2f}|{:.2f}|{:.2f}\tanglr:{:.2f}\t".format(dis_nosrl/num, r_2[0], r_2[int(num/2)], r_2[num-1], angle_none.item()/num))


    reward, r_none, distance_physical, dis_nosrl, angle_srl, angle_none, flag, r_1, r_2, r_3, std_list1, std_list2, std_list3, gt_list, gr_list, gc_list, collide, collide_, collide_frank = PassiveHapticRdwEvaluateFrank(actor_critic, args.seed,
                                         1, 0.99, None, None, 10, None, 0, args.env_name, False, num, draw, evalType=2)
    len = len(r_1)
    left = int(len/4)
    right = int(len*3/4)
    print("TYPE3:")
    print("SRL:\tdistance:{:.2f}\t{:2f}|{:.2f}|{:.2f}\tanglr:{:.2f}\tIQR:{:.2f}".format(distance_physical/num, r_1[0], r_1[int(num/2)], r_1[num-1], angle_srl.item()/num,r_1[right]-r_1[left] ), "\tcollide", collide)
    print("None:\tdistance:{:.2f}\t{:2f}|{:.2f}|{:.2f}\tanglr:{:.2f}\tIQR:{:.2f}".format(dis_nosrl/num, r_2[0], r_2[int(num/2)], r_2[num-1], angle_none.item()/num, r_2[right]-r_2[left]), "\tcollide", collide_)
    print("FRANK:\tdistance:{:.2f}\t{:2f}|{:.2f}|{:.2f}\t\t\tIQR:{:.2f}".format(0.0, r_3[0], r_3[int(num/2)], r_3[num-1], r_3[right]-r_3[left]), "\tcollide", collide_frank)