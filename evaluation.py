from os import POSIX_FADV_WILLNEED, popen
import numpy as np
from numpy.lib.function_base import median
import torch.optim as optim
import torch
import time
from envs import myutils
import argparse
# from envs.envs import *
from envs.envs_general import *
from tqdm import trange
from matplotlib.backends.backend_pdf import PdfPages
from running_mean_std import RunningMeanStd
from envs.model import Policy
from utils import plot_hist, drawPath
from envs.arguments import get_args
import numpy as np

def phrlEvaluate(actor_critic, running_mean_std, epoch, test=False, **kwargs):
    gamma, stack_frame_num, path = kwargs['gamma'], kwargs['stack_frame'], kwargs['data']
    env = PassiveHapticsEnv(gamma,  stack_frame_num, path,  eval=True)
    touch_cnt   = 0
    reward      = 0
    collide     = 0
    pde         = 0     # Physical Distance    Error
    poe         = 0     # Physical Orientation Error
    distance    = 0.0   # Traveled Virtual Distance
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
    pde_list = []
    
    env.reset()
    for t in trange(0, num):
        env.reset()
        if actor_critic != None:
            ret, pe, pe_l, oe, t_cnt, gt, gr, gc, x, y, vx, vy, std1, std2, std3, c, dis = env.step_specific_path(actor_critic, running_mean_std, t, gain=True)
            std_list1.extend(std1)
            std_list2.extend(std2)
            std_list3.extend(std3)
            gt_list.extend(gt)
            gr_list.extend(gr)
            gc_list.extend(gc)

        else: # None
            ret, pe, pe_l, oe, t_cnt, _, _, _ ,x, y, vx, vy, _, _, _, c, dis = env.step_specific_path(None, None, t, gain=False)    
  
        reward += ret
        collide += c     
        pde += pe
        poe += oe # orientation error
        touch_cnt += t_cnt
        distance += dis
        x_l.append(x)
        y_l.append(y)
        vx_l.append(vx)
        vy_l.append(vy)
        pde_list.extend(pe_l)
        
        
        
    
    reward = (reward / num).item()
    pde    = (pde    / touch_cnt) if touch_cnt > 0 else pde
    poe    = (poe    / num)
    dis_ret= (distance / collide)
    std1   = np.mean(std_list1).item()
    std2   = np.mean(std_list2).item()
    std3   = np.mean(std_list3).item()
    gt     = np.mean(gt_list).item()
    gr     = np.mean(gr_list).item()
    gc     = np.mean(gc_list).item()
    touch_cnt = touch_cnt / num
    pde_list = sorted(pde_list)
    # print(pde_list)
    pde_med = np.median(pde_list)
    rets ={
        "reward": reward, "pde" : pde,   "poe" : poe,    "collide": collide, "pde_med": pde_med, "dis_reset" : dis_ret,
        "std1"  : std1,   "std2": std2,  "std3": std3,
        "gt"    : gt,     "gr"  : gr ,   "gc"  : gc,
        "x"     : x_l,    "y"   : y_l,   "vx"  : vx_l,    "vy"     : vy_l   ,
        "touch_cnt": touch_cnt,
        "gt_l"  :gt_list, "gr_l": gr_list, "gc_l": gc_list
    }
    
    return rets


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RL')
    args = get_args()
    # args = parser.parse_args()
    # param_name = os.path.join(args.load_param)
    print('loading the ————', args.load_epoch)
    envs = PassiveHapticsEnv(args.gamma, args.stack_frame, eval=True, path=args.data)
    # Loading the actor-critc model
    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        args.net_width)
    optimizer = optim.Adam(actor_critic.parameters(), lr=args.lr) 
    # Experimental settings
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")
    actor_critic.to(device)
    # actor_critic = torch.load(param_name)
    if args.load_epoch != 0:
        ckpt = torch.load(os.path.join('./trained_models', args.env_name + "/%d.pth" % (args.load_epoch)))
    actor_critic.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    # actor_critic = torch.load('./trained_models/' + args.env_name + '/%d.pth' % args.load_epoch)
    print("Loading the " + args.env_name + '/_%d.pt' % args.load_epoch + ' to train')
    # num = 500
    # draw = True
    draw = False
    print('running ', args.test_frames, " paths:")
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
    params = args.__dict__
    rets  = phrlEvaluate(actor_critic, None, 0, True, **params)
    ret   = phrlEvaluate(None, None, 0, test=True, **params)
    
    log_dir = os.path.join("result_dir", args.env_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    plot_hist(log_dir, rets["gt_l"], rets["gr_l"], rets["gc_l"])
    drawPath(ret["vx"], ret["vy"], ret["x"], ret["y"], rets["x"], rets["y"], envs, args, -100)
    print(args.env_name)
    print("With Alignment:\tPDE_MEAN:%.4f\tPDE_MED:%.4f\treset:%d\tDistance_per_reset:%.2f" % (rets["pde"], rets["pde_med"], rets["collide"], rets["dis_reset"]))
    print("No   Alignment:\tPDE_MEAN:%.4f\tPDE_MED:%.4f\treset:%d\tDistance_per_reset:%.2f" % (ret["pde"],  ret["pde_med"] , ret["collide"],  ret["dis_reset"] ))
 
    # reward, r_none, distance_physical, dis_nosrl, angle_srl, angle_none, flag, r_1, r_2, r_3, std_list1, std_list2, std_list3, gt_list, gr_list, gc_list, collide, collide_, collide_frank = PassiveHapticRdwEvaluateFrank(actor_critic, args.seed,
    #                                      1, 0.99, None, None, 10, None, 0, args.env_name, False, num, draw, evalType=2)
    # len = len(r_1)
    # left = int(len/4)
    # right = int(len*3/4)
    # print("TYPE3:")
    # print("SRL:\tdistance:{:.2f}\t{:2f}|{:.2f}|{:.2f}\tanglr:{:.2f}\tIQR:{:.2f}".format(distance_physical/num, r_1[0], r_1[int(num/2)], r_1[num-1], angle_srl.item()/num,r_1[right]-r_1[left] ), "\tcollide", collide)
    # print("None:\tdistance:{:.2f}\t{:2f}|{:.2f}|{:.2f}\tanglr:{:.2f}\tIQR:{:.2f}".format(dis_nosrl/num, r_2[0], r_2[int(num/2)], r_2[num-1], angle_none.item()/num, r_2[right]-r_2[left]), "\tcollide", collide_)
    # print("FRANK:\tdistance:{:.2f}\t{:2f}|{:.2f}|{:.2f}\t\t\tIQR:{:.2f}".format(0.0, r_3[0], r_3[int(num/2)], r_3[num-1], r_3[right]-r_3[left]), "\tcollide", collide_frank)

