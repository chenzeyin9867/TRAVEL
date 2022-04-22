import numpy as np

dir1 = "15_15_ablation_study_predict/"
dir2 = "15_15_ablation_study_wopredict/"
dir3 = "15_15_ablation_study_womapping/"

col_a = np.loadtxt("/media/common/czy/newPHRL/statistic_result/" + dir1 + "12_12_phrl_pde.txt").reshape(-1,1)

col_name_a = np.full((col_a.shape[0], 1), 1)
total_a = np.hstack((col_name_a, col_a))

col_b = np.loadtxt("/media/common/czy/newPHRL/statistic_result/" + dir2 + "12_12_phrl_pde.txt").reshape(-1,1)
col_name_b = np.full((col_b.shape[0], 1), 2)
total_b = np.hstack((col_name_b, col_b))

col_c = np.loadtxt("/media/common/czy/newPHRL/statistic_result/" + dir3 + "12_12_phrl_pde.txt").reshape(-1,1)
col_name_c = np.full((col_c.shape[0], 1), 3)
total_c = np.hstack((col_name_c, col_c))

total = np.vstack((total_a, total_b, total_c))

np.savetxt("15_15_ablationstudy_total_pde.txt", total, fmt="%.02f")


col_a = np.loadtxt("/media/common/czy/newPHRL/statistic_result/" + dir1 + "12_12_phrl_reset.txt").reshape(-1,1)

col_name_a = np.full((col_a.shape[0], 1), 1)
total_a = np.hstack((col_name_a, col_a))

col_b = np.loadtxt("/media/common/czy/newPHRL/statistic_result/" + dir2 + "12_12_phrl_reset.txt").reshape(-1,1)
col_name_b = np.full((col_b.shape[0], 1), 2)
total_b = np.hstack((col_name_b, col_b))

col_c = np.loadtxt("/media/common/czy/newPHRL/statistic_result/" + dir3 + "12_12_phrl_reset.txt").reshape(-1,1)
col_name_c = np.full((col_c.shape[0], 1), 3)
total_c = np.hstack((col_name_c, col_c))

total = np.vstack((total_a, total_b, total_c))
# total = np.vstack((total_a, total_b))

np.savetxt("15_15_ablationstudy_total_reset.txt", total, fmt="%.02f")