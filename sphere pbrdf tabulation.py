# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 10:34:18 2022

@author: qjare
"""

import numpy as np
import os
import time
from tensor import write_tensor
from tensor import read_tensor
import rgb950_functions as rgb
from scipy.interpolate import NearestNDInterpolator
# from scipy.interpolate import LinearNDInterpolator



 
def RotM(tht):
    MM = np.zeros((4,4))
    MM[0,:] = [1,0.0,0.0,0.0];
    MM[1,:] = [0.0,np.cos(2*tht),-np.sin(2*tht),0.0];
    MM[2,:] = [0.0,np.sin(2*tht),np.cos(2*tht),0.0];
    MM[3,:] = [0.0,0.0,0.0,1.0];
    return MM



def openBinaryFile_m(filePath):
    raw = np.fromfile(filePath, dtype=np.int16)
    raw = np.reshape(raw, (600,600,3))
    raw = np.flipud(raw)
    return raw

def openBinaryFile_py(filePath):
    np.reshape(np.fromfile(filePath, dtype=np.int16),(3,600,600)).transpose((1,2,0))

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
        if binaryFilePath.__contains__(aoi):
            return binaryFilePath
    return ""

# get list of all the CMMI files and their corresponding wavelengths
def getCMMIFiles(cmmiFilePath):
    # create a dictionary to hold the cmmi files based on wavelength
    cmmiFileWavelengths = {}

    # walk through folder and get all binary files
    for subdirs, dirs, files in os.walk(cmmiFilePath):
        for fileName in files:
            # ignore the .DS_Store files
            if fileName.__contains__(".DS_Store"):
                continue
            
            if (fileName.__contains__(".cmmi")):
                # get the file path
                filePath = os.path.join(cmmiFilePath, subdirs)
                filePath = os.path.join(filePath,fileName)

                # parse the wavelength out of the file name
                wavelength = fileName.rsplit("_", 5)[3]

                # check to see if wavelengthData contains the wavelength
                if wavelength in cmmiFileWavelengths:
                    # wavelengthData contains the wavelength
                    # add the file to array

                    # get a reference to the wavelength file array
                    wavelengthFiles = cmmiFileWavelengths[wavelength]

                    # add the cmmi file path to the wavelength file array
                    wavelengthFiles.append(filePath)

                else:
                    # wavelengthData does not contains the wavelength
                    # create new array
                    wavelengthFiles = []
                    wavelengthFiles.append(filePath)

                    # add the new array to the wavelengths array
                    cmmiFileWavelengths[wavelength] = wavelengthFiles

    return cmmiFileWavelengths
    
tic=time.perf_counter()

# set file paths for cmmi and binary files
binaryFilePath = "./SphereGeometriesRed"
cmmiFilePath = "Z:\\Projects\\Oculus\\Data Sample Library\\RGB950 Polarimeter\\Measurements\\2023\\forschner_cbox\\Spheres\\green"

# sphere geometries generated using python (1) or mathematica (0) script
python_or_mathematica = 1

# get list of binary file paths
binaryList = getBinaryFiles(binaryFilePath)

# get the list of cmmi file paths
cmmiList = getCMMIFiles(cmmiFilePath)

# initialize the MM_total and index_tracker dictionaries
MM_total = read_tensor('./scenes/textures/pBRDF_red_test.pbsdf')
MM_total['M'][:,:,:,:,:,:] = 0
MM_total['wvls']=np.array([360,451,524,662,830]).astype(np.uint16)
index_tracker = np.zeros([361,91,91,3])



# loop through each wavelength
for wavelength in cmmiList:
    # get the cmmi files for each wavelengths
    wavelengthFiles = cmmiList[wavelength]
    
    print(wavelength)
    
    if wavelength == '451':
        wavelengthIndex = 0
    elif wavelength == '524':
        wavelengthIndex = 1
    else:
        wavelengthIndex = 2
        
    
        
    # set index tracker to be 0
    k=0

    # loop over the 30 cmmi files and 30 binary files
    for filePath in wavelengthFiles:
        
        
        # get the Mueller Matrix data from the file
        mm = rgb.readCMMI(filePath)

        # parse out the AOC and AOI out of the CMMI file name
        aoi = filePath.rsplit("_", 5)[5]
        aoc = filePath.rsplit("_", 5)[5]
        aoi = aoi.rsplit(".",2)[0]
        aoc = aoc.rsplit(".",2)[0]

        # lookup binary file based on the aoc and aoi
        binaryFilePath = getBinaryFile(binaryList,aoc, aoi)

        # open the binary file here
        if python_or_mathematica == 1:
            
            # sphere geometry generated in python
            table = openBinaryFile_py(binaryFilePath)
        else:
            
            # for using sphere geometries generated in Mathematica
            table = openBinaryFile_m(binaryFilePath)
        
        
        
        # if wavelengthIndex != 0:
        #     table = np.roll(table, -8, axis=1)
        
        print(k)
        # iterate k
        k = k + 1
        
        
        
        # loop over the 600x600 pixels
        for i in range(600):
            for j in range(600):
                #access the angle coord. per pixel
                # original
                theta_d_idx = table[i,j][1]
                theta_h_idx = table[i,j][0]
                phi_d_idx = table[i,j][2]

                #Get the Mueller matrix at pixel (i,j)
                if theta_h_idx != -1:
                    MM = mm[:,i,j] 
                    if (np.isnan(MM).any()==1):
                        print('issue')
                    
                    # normalize the mm data
                    # norm = np.linalg.norm(MM)
                    #normMM = MM /MM[0]
    
                    # reshape to 4x4
                    MM =RotM(np.pi/2)@(MM.reshape(4,4))@RotM(-np.pi/2)
                    
    
                    
                    
                    
                    # add data to dictionary and increment the index tracker
                    index_tracker[phi_d_idx,theta_d_idx,theta_h_idx,wavelengthIndex] +=1;
                    MM_total['M'][phi_d_idx,theta_d_idx,theta_h_idx,wavelengthIndex,:,:] += MM;
                    # print("MM_total")
                    # print(MM_total['M'][phi_d_idx,theta_d_idx,theta_h_idx,wavelengthIndex,:,:])
                    # print("MM")
                    # print(MM)
        



# calculate the mean using MM_total and index_tracker
#Computing the average MM per bin
# loop over all the bins
for i in range(361):
    for j in range(91):
        for k in range(91):
            # loop over all the wavelengths
            for m in range(3):
                if (index_tracker[i,j,k,m] == 0):
                    # MM_total['M'][i,j,k,m,:,:]=0
                    # putting the ideal depolarizer instead of 0
                    MM_total['M'][i,j,k,m,0,0] = 1
                    MM_total['M'][i,j,k,m,0,1] = 0
                    MM_total['M'][i,j,k,m,0,2] = 0
                    MM_total['M'][i,j,k,m,0,3] = 0
                    MM_total['M'][i,j,k,m,1,0] = 0
                    MM_total['M'][i,j,k,m,2,0] = 0
                    MM_total['M'][i,j,k,m,3,0] = 0
                    MM_total['M'][i,j,k,m,1,1] = 0
                    MM_total['M'][i,j,k,m,1,2] = 0
                    MM_total['M'][i,j,k,m,1,3] = 0
                    MM_total['M'][i,j,k,m,2,1] = 0
                    MM_total['M'][i,j,k,m,2,2] = 0
                    MM_total['M'][i,j,k,m,2,3] = 0
                    MM_total['M'][i,j,k,m,3,1] = 0
                    MM_total['M'][i,j,k,m,3,2] = 0
                    MM_total['M'][i,j,k,m,3,3] = 0
                else:
                    MM_total['M'][i,j,k,m,:,:] = MM_total['M'][i,j,k,m,:,:]/index_tracker[i,j,k,m];  

                if (np.isnan(MM_total['M'][i,j,k,m,:,:]).any()==1):
                    print('issue')      


for wl in range(3):
    indices=np.nonzero(MM_total['M'][:,:,:,wl,0,0])
    
    m00=np.zeros(np.shape(indices)[1])
    m01=np.zeros(np.shape(indices)[1])
    m02=np.zeros(np.shape(indices)[1])
    m03=np.zeros(np.shape(indices)[1])
    m10=np.zeros(np.shape(indices)[1])
    m11=np.zeros(np.shape(indices)[1])
    m12=np.zeros(np.shape(indices)[1])
    m13=np.zeros(np.shape(indices)[1])
    m20=np.zeros(np.shape(indices)[1])
    m21=np.zeros(np.shape(indices)[1])
    m22=np.zeros(np.shape(indices)[1])
    m23=np.zeros(np.shape(indices)[1])
    m30=np.zeros(np.shape(indices)[1])
    m31=np.zeros(np.shape(indices)[1])
    m32=np.zeros(np.shape(indices)[1])
    m33=np.zeros(np.shape(indices)[1])
    phi_d_list=np.zeros(np.shape(indices)[1])
    theta_d_list=np.zeros(np.shape(indices)[1])
    theta_h_list=np.zeros(np.shape(indices)[1])
    
    for qq in range(np.shape(indices)[1]):
        m00[qq]=MM_total['M'][indices[0][qq],indices[1][qq],indices[2][qq],wl,0,0]
        m01[qq]=MM_total['M'][indices[0][qq],indices[1][qq],indices[2][qq],wl,0,1]
        m02[qq]=MM_total['M'][indices[0][qq],indices[1][qq],indices[2][qq],wl,0,2]
        m03[qq]=MM_total['M'][indices[0][qq],indices[1][qq],indices[2][qq],wl,0,3]
        m10[qq]=MM_total['M'][indices[0][qq],indices[1][qq],indices[2][qq],wl,1,0]
        m11[qq]=MM_total['M'][indices[0][qq],indices[1][qq],indices[2][qq],wl,1,1]
        m12[qq]=MM_total['M'][indices[0][qq],indices[1][qq],indices[2][qq],wl,1,2]
        m13[qq]=MM_total['M'][indices[0][qq],indices[1][qq],indices[2][qq],wl,1,3]
        m20[qq]=MM_total['M'][indices[0][qq],indices[1][qq],indices[2][qq],wl,2,0]
        m21[qq]=MM_total['M'][indices[0][qq],indices[1][qq],indices[2][qq],wl,2,1]
        m22[qq]=MM_total['M'][indices[0][qq],indices[1][qq],indices[2][qq],wl,2,2]
        m23[qq]=MM_total['M'][indices[0][qq],indices[1][qq],indices[2][qq],wl,2,3]
        m30[qq]=MM_total['M'][indices[0][qq],indices[1][qq],indices[2][qq],wl,3,0]
        m31[qq]=MM_total['M'][indices[0][qq],indices[1][qq],indices[2][qq],wl,3,1]
        m32[qq]=MM_total['M'][indices[0][qq],indices[1][qq],indices[2][qq],wl,3,2]
        m33[qq]=MM_total['M'][indices[0][qq],indices[1][qq],indices[2][qq],wl,3,3]
        phi_d_list[qq]=MM_total['phi_d'][0,indices[0][qq]]
        theta_d_list[qq]=MM_total['theta_d'][0,indices[1][qq]]
        theta_h_list[qq]=MM_total['theta_h'][0,indices[2][qq]]
    
    phi_d=MM_total['phi_d'][0]
    theta_d=MM_total['theta_d'][0]
    theta_h=MM_total['theta_h'][0]
    print('Interpolator Run: {}'.format(wl+1))
    X,Y,Z=np.meshgrid(phi_d,theta_d,theta_h,indexing='ij')
    interp00=NearestNDInterpolator(list(zip(phi_d_list,theta_d_list,theta_h_list )),m00)
    interp01=NearestNDInterpolator(list(zip(phi_d_list,theta_d_list,theta_h_list )),m01)
    interp02=NearestNDInterpolator(list(zip(phi_d_list,theta_d_list,theta_h_list )),m02)
    interp03=NearestNDInterpolator(list(zip(phi_d_list,theta_d_list,theta_h_list )),m03)
    interp10=NearestNDInterpolator(list(zip(phi_d_list,theta_d_list,theta_h_list )),m10)
    interp11=NearestNDInterpolator(list(zip(phi_d_list,theta_d_list,theta_h_list )),m11)
    interp12=NearestNDInterpolator(list(zip(phi_d_list,theta_d_list,theta_h_list )),m12)
    interp13=NearestNDInterpolator(list(zip(phi_d_list,theta_d_list,theta_h_list )),m13)
    interp20=NearestNDInterpolator(list(zip(phi_d_list,theta_d_list,theta_h_list )),m20)
    interp21=NearestNDInterpolator(list(zip(phi_d_list,theta_d_list,theta_h_list )),m21)
    interp22=NearestNDInterpolator(list(zip(phi_d_list,theta_d_list,theta_h_list )),m22)
    interp23=NearestNDInterpolator(list(zip(phi_d_list,theta_d_list,theta_h_list )),m23)
    interp30=NearestNDInterpolator(list(zip(phi_d_list,theta_d_list,theta_h_list )),m30)
    interp31=NearestNDInterpolator(list(zip(phi_d_list,theta_d_list,theta_h_list )),m31)
    interp32=NearestNDInterpolator(list(zip(phi_d_list,theta_d_list,theta_h_list )),m32)
    interp33=NearestNDInterpolator(list(zip(phi_d_list,theta_d_list,theta_h_list )),m33)
    
    m00interp=interp00(X,Y,Z)
    m01interp=interp01(X,Y,Z)
    m02interp=interp02(X,Y,Z)
    m03interp=interp03(X,Y,Z)
    m10interp=interp10(X,Y,Z)
    m11interp=interp11(X,Y,Z)
    m12interp=interp12(X,Y,Z)
    m13interp=interp13(X,Y,Z)
    m20interp=interp20(X,Y,Z)
    m21interp=interp21(X,Y,Z)
    m22interp=interp22(X,Y,Z)
    m23interp=interp23(X,Y,Z)
    m30interp=interp30(X,Y,Z)
    m31interp=interp31(X,Y,Z)
    m32interp=interp32(X,Y,Z)
    m33interp=interp33(X,Y,Z)
    
    MM_total['M'][:,:,:,wl,0,0]=m00interp
    MM_total['M'][:,:,:,wl,0,1]=m01interp
    MM_total['M'][:,:,:,wl,0,2]=m02interp
    MM_total['M'][:,:,:,wl,0,3]=m03interp
    MM_total['M'][:,:,:,wl,1,0]=m10interp
    MM_total['M'][:,:,:,wl,1,1]=m11interp
    MM_total['M'][:,:,:,wl,1,2]=m12interp
    MM_total['M'][:,:,:,wl,1,3]=m13interp
    MM_total['M'][:,:,:,wl,2,0]=m20interp
    MM_total['M'][:,:,:,wl,2,1]=m21interp
    MM_total['M'][:,:,:,wl,2,2]=m22interp
    MM_total['M'][:,:,:,wl,2,3]=m23interp
    MM_total['M'][:,:,:,wl,3,0]=m30interp
    MM_total['M'][:,:,:,wl,3,1]=m31interp
    MM_total['M'][:,:,:,wl,3,2]=m32interp
    MM_total['M'][:,:,:,wl,3,3]=m33interp



MM_total['M'][:,:,:,4,:,:]=MM_total['M'][:,:,:,2,:,:]
MM_total['M'][:,:,:,3,:,:]=MM_total['M'][:,:,:,2,:,:]
MM_total['M'][:,:,:,2,:,:]=MM_total['M'][:,:,:,1,:,:]
MM_total['M'][:,:,:,1,:,:]=MM_total['M'][:,:,:,0,:,:]
# MM_total['M'][:,:,:,1,:]=MM_total['M'][:,:,:,wavelengthIndex,:]
# MM_total['M'][:,:,:,2,:]=MM_total['M'][:,:,:,wavelengthIndex,:]
# MM_total['M'][:,:,:,3,:]=MM_total['M'][:,:,:,wl,:]
# for i in range(5):
#     MM_total['M'][:,:,:,i,:]=MM_total['M'][:,:,:,0,:]

#load the pbsdf file
# file is sample pBSDF file
# A = tensor.read_tensor('C:/Users/qjare/Desktop/pBRDF_stuff/pBRDF_test.pbsdf')
# A['M'] = MM_total['M'];
# A['wvls'] = MM_total['wvls'];
write_tensor("./green_sphere_depolarizer_pBRDF.pbsdf",**MM_total); 

toc=time.perf_counter()
print(f"Completed pBRDF calculation in {toc - tic:0.4f} seconds. Neat!")
