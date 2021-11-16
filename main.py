import os
import time
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


# OBS_NORM = False
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
    envs = PassiveHapticsEnv(args.gamma, args.stack_frame, eval=False, path=args.data)
    
    # Loading the actor-critc model
    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        args.net_width)
    optimizer = optim.Adam(actor_critic.parameters(), lr=args.lr) 
    actor_critic.to(device)
    
    if args.load_epoch != 0:
        ckpt = torch.load(os.path.join('./trained_modes', args.env_name + "/%d.pth" % (args.load_epoch)))
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

        for step in trange(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action_mean, action_std = actor_critic.act(rollouts.obs[step])
                dist = FixedNormal(action_mean, action_std)
                action = dist.sample()
                action = torch.clamp(action, -1.0, 1.0)
                action_log_prob = dist.log_probs(action)

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)
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

        rollouts.after_update()

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
                }, path
            )
            print("Save checkpoint ", path)

        num = 100
        if j == args.load_epoch:
                r_none, pde_, poe_, collide_, _, _, _, _, _, _  = phrlEvaluate(None, j, **params)
        if j % args.log_interval == 0:
            r_eval, pde, poe, collide, std1, std2, std3, gt, gr, gc = phrlEvaluate(actor_critic, j, **params)

            print(args.env_name)
            print(
                "Epoch_%d/%d" % (j, num_updates), 
                "|r_phrl:{:.2f} |r_none:{:.2f} |pde_phrl:{:.2f} |pde_none:{:.2f} "
                "|poe_phrl:{:.2f} |poe_none:{:.2f}"
                .format(entropy_loss, r_eval.item(), r_none.item(), pde, pde_, poe, poe_))
            print("std:{:.3f} {:.3f} {:.3f}" 
                  "|gt:{:.2f}|gr:{:.2f} |gc:{:.2f}\t|".
                  format(np.mean(std1), np.mean(std2), np.mean(std3), 
                  np.mean(gt).item(), np.mean(gr).item(), np.mean(gc).item()),
                  "reset_phrl:", collide, " reset_none:", collide_,  
                  "\t|t:{:.2f} ".format(time.time() - t_start)) 
            t_start = time.time()
            
            
        # Handle the tensorboard 
        writer1.add_scalar('Loss/value_loss', value_loss, global_step=j)
        writer1.add_scalar('Loss/actor_loss', action_loss, global_step=j)
        writer1.add_scalar('entropy_loss', entropy_loss, global_step=j)
        writer1.add_scalar('total_loss', total_loss, global_step=j)
        writer1.add_scalar('pde', pde, global_step=j)
        writer1.add_scalar('poe', poe, global_step=j)
        writer1.add_scalar('phrl_reward', r_eval, global_step=j)
        
        writer1.add_scalar('gt', np.mean(gt).item(), global_step=j)
        writer1.add_scalar('gr', np.mean(gr).item(), global_step=j)
        writer1.add_scalar('gc', np.mean(gc).item(), global_step=j)
        writer1.add_scalar('reset', collide, global_step=j)
        writer1.add_scalar("explained_var", explained_variance, global_step=j)
        layout = {
            'Loss':{
                'v_loss': ['Multiline', ["Loss/value_loss"]],
                # 'a_loss': action_loss,
                # 'e_loss': entropy_loss,
                # 't_loss': total_loss
            },
            # 'Metrics':{
            #     'pde': pde,
            #     'poe': poe,
            #     'collide': collide,
            #     'expained_var': explained_variance
            # },
            # 'var':{
            #     'gt': gt,
            #     'gc': gc,
            #     'gr': gr,
            #     'std_gt': std1,
            #     "std_gr": std2,
            #     "std_gc": std3
            # }
            
        }
        writer1.add_custom_scalars(layout)
if __name__ == "__main__":
    main()

