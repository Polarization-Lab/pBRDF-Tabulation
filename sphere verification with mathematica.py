# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 16:32:44 2023

@author: Stefan
"""
import numpy as np
import matplotlib.pyplot as plt
import os

def openBinaryFile_m(filePath):
    raw = np.fromfile(filePath, dtype=np.int16)
    raw = np.reshape(raw, (600,600,3))
    raw = np.flipud(raw)
    return raw

def getBinaryFiles(binaryFilePath):
    binaryFilePathList = []
    # walk through folder and get all binary files
    for subdirs, dirs, files in os.walk(binaryFilePath):
        for fileName in files:
            # ignore the .DS_Store files
            if fileName.__contains__(".DS_Store"):
                continue
            filePath = os.path.join(binaryFilePath,fileName)
            binaryFilePathList.append(filePath)
    return binaryFilePathList

def getBinaryFile(binaryFilePathList, aoc, aoi):
    for binaryFilePath in binaryFilePathList:
        # parse the AOC and AOI from the file name
        if binaryFilePath.__contains__(str(aoi)):
            return binaryFilePath
    return ""



#%% Import geometries created in Python


coord = np.zeros((3,600,600,15),dtype=np.int16)
acquisitions = np.arange(20,91,5)
for i in range(15):
    coord[:,:,:,i] = np.reshape(np.fromfile('./SphereGeometriesRed_py_verification/sphere_pBRDF_index_{}.bin'.format(acquisitions[i]), dtype=np.int16),(3,600,600)) 

#%% Plotting sphere coordinates 

indx = 10
fig, axs = plt.subplots(ncols = 3)

axs[0].imshow(coord[0,:,:,indx])
axs[0].set_title('theta h')
axs[1].imshow(coord[1,:,:,indx])
axs[1].set_title('theta d')
axs[2].imshow(coord[2,:,:,indx])
axs[2].set_title('phi d')
fig.suptitle('Python Generated Sphere geometries')
plt.show()


#%% Import geometries created in Mathematica

binaryFilePath = "./SphereGeometriesRed_m_verification"

binaryList = getBinaryFiles(binaryFilePath)

binaryFilePath = getBinaryFile(binaryList,acquisitions[indx], acquisitions[indx])

table = openBinaryFile_m(binaryFilePath)

# Take difference between python and mathematica geometries
diff = table - np.transpose(coord[:,:,:,indx], (1, 2, 0))

#%% Plot difference

fig, axs = plt.subplots(ncols = 3)

axs[0].imshow(diff[:,:,0])
axs[0].set_title('theta h')
axs[1].imshow(diff[:,:,1])
axs[1].set_title('theta d')
axs[2].imshow(diff[:,:,2])
axs[2].set_title('phi d')
fig.suptitle('Mathematica vs. Python Difference')
plt.show()
