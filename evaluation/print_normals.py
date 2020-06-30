import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import glob

filepath = 'test/inference/*.*'
for filename in glob.glob(filepath):
    if '.npy' in filename and "parameters" in filename:
      print(filename)
      data = np.load(filename)
      print(str(data),"\n")
print('done')





# [[ 1.1883793   0.35464397 -0.17278007]
#  [-0.03823359  0.17064653 -0.72920305]
#  [-1.4104024   1.4765792   0.09783125]
#  [-0.01730385  0.26602593 -0.7670637 ]
#  [-1.440785    1.0959535  -0.10083918]]




# [[ 1.1883793   0.35464397 -0.17278007]
#  [-0.03823359  0.17064653 -0.72920305]
#  [-1.4104024   1.4765792   0.09783125]
#  [-0.01730385  0.26602593 -0.7670637 ]
#  [-1.440785    1.0959535  -0.10083918]] 