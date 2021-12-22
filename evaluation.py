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
from running_mean_std import RunningMeanStd
import numpy as np

def phrlEvaluate(actor_critic, running_mean_std, epoch, **kwargs):
    gamma, stack_frame_num, path = kwargs['gamma'], kwargs['stack_frame'], kwargs['data']
    env = PassiveHapticsEnv(gamma,  stack_frame_num, path,  eval=True)
    touch_cnt   = 0
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
    x_l, y_l, vx_l, vy_l = [], [], [], []
    
    env.reset()
    for t in range(0, num):
        env.reset()
        if actor_critic != None:
            ret, pe, oe, t_cnt, gt, gr, gc, x, y, vx, vy, std1, std2, std3, c = env.step_specific_path(actor_critic, running_mean_std, t, gain=True)
            std_list1.extend(std1)
            std_list2.extend(std2)
            std_list3.extend(std3)
            gt_list.extend(gt)
            gr_list.extend(gr)
            gc_list.extend(gc)

        else: # None
            ret, pe, oe, t_cnt, _, _, _ ,x, y, vx, vy, _, _, _, c = env.step_specific_path(None, None, t, gain=False)    
  
        reward += ret
        collide += c     
        pde += pe
        poe += oe # orientation error
        touch_cnt += t_cnt
        x_l.append(x)
        y_l.append(y)
        vx_l.append(vx)
        vy_l.append(vy)
        
        
        

    reward = (reward / num).item()
    pde    = (pde    / touch_cnt)
    poe    = (poe    / num)
    std1   = np.mean(std_list1).item()
    std2   = np.mean(std_list2).item()
    std3   = np.mean(std_list3).item()
    gt     = np.mean(gt_list).item()
    gr     = np.mean(gr_list).item()
    gc     = np.mean(gc_list).item()
    
    rets ={
        "reward": reward, "pde" : pde,   "poe" : poe,    "collide": collide,
        "std1"  : std1,   "std2": std2,  "std3": std3,
        "gt"    : gt,     "gr"  : gr ,   "gc"  : gc,
        "x"     : x_l,    "y"   : y_l,   "vx"  : vx_l,    "vy"     : vy_l   
    }
    
    return rets


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