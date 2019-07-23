import pandas as pd
import numpy as np
import random
import os

#Example usage:
# import gps_data_grab as gdb
# sample = gdb.data_sample_tensor(18,60)
#
#

######## load the data set #########
# Change this data directory!!!!

os.getcwd()
filedirectory = "C:\\Users\\HeLi0001\\Desktop\\best_journeys\\"


def grab_random_gps_blocks(batchsize, blocksize=60):
    blocks = []
    blocks_collect = 0
    offset = blocksize // 2
    while blocks_collect < batchsize:
        file = random.choice(os.listdir(filedirectory))
        ds = pd.read_csv(filedirectory + file)
        for k in range(ds.shape[0] // blocksize):
            block = ds[k * blocksize: (k + 1) * blocksize].values
            start = np.copy(block[0, :])
            start[2] = 0
            start[3] = 0
            start[4] = 0
            start[6] = 0
            start[7] = 0
            start[8] = 0
            newblock = block - start
            if not np.isnan( newblock ).any():
                blocks.append( newblock )
                blocks_collect += 1
        for k in range((ds.shape[0] - offset) // blocksize):
            block = ds[k * blocksize + offset: (k + 1) * blocksize + offset].values
            start = np.copy(block[0, :])
            start[2] = 0
            start[3] = 0
            start[4] = 0
            start[6] = 0
            start[7] = 0
            start[8] = 0
            newblock = block - start
            if not np.isnan( newblock ).any():
                blocks.append( newblock )
                blocks_collect += 1
    blocks = random.sample(blocks, batchsize)
    return blocks


def data_sample_tensor(batchsize, blocksize):
    blocks = grab_random_gps_blocks(batchsize, blocksize)
    tnorm = blocksize - 1
    tblocks = np.zeros((batchsize, blocksize, 10))
    for foo, block in enumerate(blocks):
        tblocks[foo, :, 0] = block[:, 0] / tnorm
        tblocks[foo, :, 1] = block[:, 1] / tnorm / 27
        tblocks[foo, :, 2] = block[:, 2] / 27
        tblocks[foo, :, 3] = block[:, 3] / 27
        tblocks[foo, :, 4] = block[:, 4]
        tblocks[foo, :, 5] = block[:, 5] / tnorm / 27
        tblocks[foo, :, 6] = block[:, 6] / 27
        tblocks[foo, :, 7] = block[:, 7] / 27
        tblocks[foo, :, 8] = block[:, 8]
        tblocks[foo, :, 9] = block[:, 9] / 10
    return tblocks
