import numpy as np
import os

path = "./train_sim_traces"
out_path = "./mrc_div_train"
BITS_IN_BYTE = 8.0
MBITS_IN_BITS = 1000000.0
TIME_INTERVAL = 5.0
files = os.listdir(path)
div_num = [0, 1, 2, 3, 4, 5, 6, 50]

count = 0
count01 = 0

for file in files:
    count += 1
    print(count)
    print(file)

    data = []
    time_all = []
    idx = 0
    cur_path = path+"/"+file
    if not os.path.isdir(file):
        with open(cur_path, 'rb') as f:
            for line in f:
                time_all.append(float(line.split()[0]))
                data.append(float(line.split()[1]))
            f.close()

        data = np.array(data)

        throughput_mean = np.mean(data)

        for i in range(len(div_num)):
            if div_num[i] >= throughput_mean:
                idx = i

                if idx ==1:
                    count01 += 1

                break

print count01

        #
        # new_path = out_path+"/"+str(div_num[idx-1]) + '-' + str(div_num[idx]) + "-" + file
        # with open(new_path, 'wb') as fnew:
        #     for i in range(len(data)):
        #         fnew.write(bytes(str(time_all[i]) + " " + str(data[i]) + '\n'))
        #     fnew.close()

