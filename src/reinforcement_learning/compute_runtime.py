import numpy as np

total_time = 0.0
for i in range(10):
    f = open('../experiments/results%d/total_time.txt'%(i))
    time = f.readlines()
    time = float(time[0])
    print(time)
    total_time += time
    f.close()
print(total_time)
print(total_time/10)
