import numpy as np
import os
path1=r'D:\znmz\yolov5\traindata\labels'
i=0
for fr in os.listdir(path1):
    data=np.loadtxt(os.path.join(path1, fr), ndmin=2)
    data.astype(np.float32)
    print(data)