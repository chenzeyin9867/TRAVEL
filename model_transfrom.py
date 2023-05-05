import torch
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
from gym.spaces.box import Box
model_weight_path = "/media/common/czy/newPHRL/trained_models/47_h8w8_p66_2to2new_gamma0.99_lr0.0003_r0.5-r_unagin_+o+3d_ratio_div10_b8192_clip0.1_p1.0_final0_nearst_Constraint_ppo4_ce0.005_layer4_hs256/6000.pth"  #自己的pth文件路径
out_onnx = './trained_models/onnx/h8w8p66_2to2.onnx'           #保存生成的onnx文件路径
actor_critic_dict = torch.load(model_weight_path)['model_state_dict']
# model.load_state_dict(torch.load(model_weight_path,map_location=torch.device('cpu'))) #加载自己的pth文件
actor_critic = Policy(
        Box(-1., 1., (60,)).shape,
        Box(-1.0, 1.0, (3,)),
        4)

actor_critic.load_state_dict(actor_critic_dict)

x1 = torch.randn(1, 60)
#define input and output nodes, can be customized
input_names = ["input"]
output_names = ["output"]
#convert pytorch to onnx
torch_out = torch.onnx.export(actor_critic, x1, out_onnx, export_params=True,
                              input_names=input_names, output_names=output_names)
print(torch_out)