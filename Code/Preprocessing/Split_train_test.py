import os
import numpy as np
import pdb
import json

IDs = np.loadtxt(fname = "./All_HCP_ID.txt", dtype=str)
Total_num = len(IDs)
Train_rate = 0.8

np.random.seed(19950504)
np.random.shuffle(IDs)
Shuffled_IDs = IDs
Train_IDs = Shuffled_IDs[:int(Total_num*Train_rate)].tolist()
Test_IDs = Shuffled_IDs[int(Total_num*Train_rate):].tolist()

# pdb.set_trace()
# json_string = json.dumps(Train_IDs)
with open('train_files.json', 'w') as outfile:
    json.dump(Train_IDs, outfile)

# json_string = json.dumps(Test_IDs)
with open('test_files.json', 'w') as outfile:
    json.dump(Test_IDs, outfile)

