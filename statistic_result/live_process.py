import numpy as np

dir = "live-user/"

col_a = np.loadtxt("/media/common/czy/newPHRL/statistic_result/" + dir + "DLError.txt").reshape(-1,1)

col_name_a = np.full((col_a.shape[0], 1), 1)
total_a = np.hstack((col_name_a, col_a))

col_b = np.loadtxt("/media/common/czy/newPHRL/statistic_result/" + dir + "NoneError.txt").reshape(-1,1)
col_name_b = np.full((col_b.shape[0], 1), 2)
total_b = np.hstack((col_name_b, col_b))

col_c = np.loadtxt("/media/common/czy/newPHRL/statistic_result/" + dir + "REAError.txt")
col_c = np.delete(col_c, col_c==-1).reshape(-1, 1)
col_name_c = np.full((col_c.shape[0], 1), 3)
total_c = np.hstack((col_name_c, col_c))


total = np.vstack((total_a, total_b, total_c))

np.savetxt("live_pde.txt", total, fmt="%.02f")


col_a = np.loadtxt("/media/common/czy/newPHRL/statistic_result/" + dir + "DLReset.txt").reshape(-1,1)

col_name_a = np.full((col_a.shape[0], 1), 1)
total_a = np.hstack((col_name_a, col_a))

col_b = np.loadtxt("/media/common/czy/newPHRL/statistic_result/" + dir + "NoneReset.txt").reshape(-1,1)
col_name_b = np.full((col_b.shape[0], 1), 2)
total_b = np.hstack((col_name_b, col_b))

col_c = np.loadtxt("/media/common/czy/newPHRL/statistic_result/" + dir + "REAReset.txt")
col_c = np.delete(col_c, col_c==-1).reshape(-1, 1)
col_name_c = np.full((col_c.shape[0], 1), 3)
total_c = np.hstack((col_name_c, col_c))


total = np.vstack((total_a, total_b, total_c))

np.savetxt("live_reset.txt", total, fmt="%.02f")