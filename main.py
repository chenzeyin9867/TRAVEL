import os
from pickle import NONE
import time
from numpy.core.numeric import roll
from tensorboardX import SummaryWriter
import numpy as np
import torch
from torch._C import layout
import torch.optim as optim
from envs import algo, myutils
from envs.arguments import get_args
from envs.envs_general import  PassiveHapticsEnv
from envs.model import Policy
from envs.storage import RolloutStorage
from evaluation import phrlEvaluate
from envs.distributions import FixedNormal
from tqdm import trange
from running_mean_std import RunningMeanStd
from utils import drawPath, plot_hist


OBS_NORM = False
def main():
    args = get_args()
    params = args.__dict__
    writer1 = SummaryWriter('runs/' + args.env_name)
    if not os.path.exists('runs/' + args.env_name):
        os.makedirs('runs/' + args.env_name)
    
    # Experimental settings
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")
    
    # Debug infos
    print(device)
    print(args)
    
    # Instance the env, training batch
    envs = PassiveHapticsEnv(args.gamma, args.stack_frame, eval=False, path=args.data, obs_norm = OBS_NORM)
    
    if OBS_NORM:
        running_mean_std = RunningMeanStd(envs.observation_space.shape)
    
    # Loading the actor-critc model
    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        args.net_width)
    optimizer = optim.Adam(actor_critic.parameters(), lr=args.lr) 
    actor_critic.to(device)
    
    if args.load_epoch != 0:
        ckpt = torch.load(os.path.join('./trained_models', args.env_name + "/%d.pth" % (args.load_epoch)))
        actor_critic.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        # actor_critic = torch.load('./trained_models/' + args.env_name + '/%d.pth' % args.load_epoch)
        print("Loading the " + args.env_name + '/_%d.pt' % args.load_epoch + ' to train')
        
    agent = algo.PPO(   
        actor_critic,
        args.clip_param,
        args.ppo_epoch,
        args.num_mini_batch,
        args.value_loss_coef,
        args.entropy_coef,
        optimizer,
        lr=args.lr,
        eps=args.eps,
        max_grad_norm=args.max_grad_norm)
    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)
    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    t_start = time.time()
    num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes

    for j in trange(args.load_epoch, num_updates):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            myutils.update_linear_schedule(
                agent.optimizer, j, num_updates, args.lr)

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                obs = rollouts.obs[step]
                if OBS_NORM:                # using obs norm
                    obs = running_mean_std.process(obs)
                value, action_mean, action_std = actor_critic.act(obs)
                dist = FixedNormal(action_mean, action_std)
                action = dist.rsample()
                
                action_log_prob = dist.log_probs(action)
                clipped_action = torch.clamp(action, -1.0, 1.0)
                
                
            # Obser reward and next obs
            obs, reward, done, infos = envs.step(clipped_action)
            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in infos])
            rollouts.insert(obs, action, action_log_prob, value, torch.Tensor([reward]), masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(rollouts.obs[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, entropy_loss, total_loss, explained_variance = agent.update(rollouts, args)
        if OBS_NORM:
            running_mean_std.update(rollouts.obs)
        else:
            running_mean_std = NONE
        rollouts.after_update()
        
        
        # Linear decay the std
        # actor_critic.dist.logstd = actor_critic.dist.initstd * (num_updates - j + 100) / (num_updates)

        # save for every interval episode or for the last epoch
        if (j % args.save_interval == 0
                or j == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.env_name)
            try:
                os.makedirs(save_path)
            except OSError:
                pass
            path =  os.path.join(save_path,  "%d.pth" % j)
            torch.save(
                {
                    'global_step':                  j,
                    'model_state_dict':             actor_critic.state_dict(),
                    'optimizer_state_dict':         agent.optimizer.state_dict(),
                    'running_mean'        :         running_mean_std.mean if OBS_NORM else 0,
                    'running_var':                  running_mean_std.var  if OBS_NORM else 0
                }, path
            )
            print("Save checkpoint ", path)

        num = 100
        if j == args.load_epoch:
                rets_ = phrlEvaluate(None, running_mean_std, j, test=False, **params) # Only run once
        if j % args.log_interval == 0:
                rets  = phrlEvaluate(actor_critic, running_mean_std, j, test=False, **params)
                drawPath(rets_["vx"], rets_["vy"], rets_["x"], rets_["y"], rets["x"], rets["y"], envs, args, j)
                
                hist_path = os.path.join("plot_result", args.env_name + "/hist/" + str(j))
                if not os.path.exists(hist_path):
                    os.makedirs(hist_path)
                plot_hist(hist_path, rets['gt_l'], rets['gr_l'], rets['gc_l'])
                
                print(args.env_name)
                lr = args.lr - (args.lr * (j / float(num_updates)))
                print( "Epoch_%d/%d\t" % (j, num_updates), "lr:%.8f" % (lr), 
                    "\tr_phrl:%.2f\tr_none:%.2f\tpde_phrl:%.2f\tpde_none:%.2f\tpde_phrl_med:%.2f\tpde_none_med:%.2f\ttouch_phrl:%.2f\ttouch_none:%.2f"
                        %(rets["reward"], rets_["reward"], rets["pde"], rets_["pde"], rets["pde_med"], rets_["pde_med"], rets["touch_cnt"], rets_["touch_cnt"]))
                print("std:%.3f %.3f %.3f\t\tgt:%.3f\tgr:%.3f\tgc:%.3f\t"
                    % (rets["std1"], rets["std2"], rets["std3"], rets["gt"], rets["gr"], rets["gc"]), end="")
                print("reset_phrl:", rets["collide"], " reset_none:", rets_["collide"], "\t|t:%.2f " % (time.time() - t_start)) 
                t_start = time.time()
            
            
        # Handle the tensorboard 
        writer1.add_scalar('Loss/value_loss', value_loss, global_step=j)
        writer1.add_scalar('Loss/actor_loss', action_loss, global_step=j)
        writer1.add_scalar('Loss/entropy_loss', entropy_loss, global_step=j)
        writer1.add_scalar('Loss/total_loss', total_loss, global_step=j)
        # Metrics
        writer1.add_scalar('Metric/pde', rets["pde"], global_step=j)
        writer1.add_scalar("Metric/pde_med", rets["pde_med"], global_step=j)
        writer1.add_scalar('Metric/touch_cnt', rets["touch_cnt"], global_step=j)
        writer1.add_scalar('Metric/phrl_reward', rets["reward"], global_step=j)
        writer1.add_scalar("Metric/explained_var", explained_variance, global_step=j)
        # Gains
        writer1.add_scalar('gains/gt', rets["gt"], global_step=j)
        writer1.add_scalar('gains/gr', rets["gr"], global_step=j)
        writer1.add_scalar('gains/gc', rets["gc"], global_step=j)
        # Vars
        writer1.add_scalar('Vars/std1',  rets["std1"],    global_step=j)
        writer1.add_scalar('Vars/std2',  rets["std2"],    global_step=j)
        writer1.add_scalar('Vars/std3',  rets["std3"],    global_step=j)
        writer1.add_scalar('Vars/reset', rets["collide"], global_step=j)
        
        # writer1.add_image()
        # layout = {
        #     'Loss':{
        #         'v_loss': ['Multiline', ["value_loss"]],
        #         # 'a_loss': action_loss,
        #         # 'e_loss': entropy_loss,
        #         # 't_loss': total_loss
        #     },
        #     # 'Metrics':{
        #     #     'pde': pde,
        #     #     'poe': poe,
        #     #     'collide': collide,
        #     #     'expained_var': explained_variance
        #     # },
        #     # 'var':{
        #     #     'gt': gt,
        #     #     'gc': gc,
        #     #     'gr': gr,
        #     #     'std_gt': std1,
        #     #     "std_gr": std2,
        #     #     "std_gc": std3
        #     # }
            
        # }
        # writer1.add_custom_scalars(layout)
if __name__ == "__main__":
    main()

