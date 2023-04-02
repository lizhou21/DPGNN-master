import numpy as np
import glob
import sys
import os



save_dir= sys.argv[1]
res_acc = []
res_f1 = []
for i in range(100):
    ofile = 'seed_'+ str(i)
    with open(os.path.join(save_dir, ofile), 'r') as f:
        line = f.readlines()[-1]
        line = line.strip().split()

        res_acc.append(float(line[-3]))
        res_f1.append(float(line[-1]))

res_acc = np.array(res_acc)
res_f1 = np.array(res_f1)
res_acc = np.sort(res_acc)
res_f1 = np.sort(res_f1)
res_acc = res_acc[5:-5]
res_f1 = res_f1[5:-5]
print(res_acc)
print(res_f1)
print('mean_acc:', np.mean(res_acc))
print('max_acc:', np.max(res_acc))
print('min_acc:', np.min(res_acc))
print('std_acc:', np.std(res_acc))

print('mean_f1:', np.mean(res_f1))
print('max_f1:', np.max(res_f1))
print('min_f1:', np.min(res_f1))
print('std_f1:', np.std(res_f1))