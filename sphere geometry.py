# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 18:58:33 2023

@author: Stefan Forschner
"""
import numpy as np

import scipy as sc
import matplotlib.pyplot as plt
import os



# This script generates the Rusinkiewicz coordinate angles for measured sphere geometries

#%% Functions  

def readCMMI(inputfilename):
    raw = np.fromfile(inputfilename,np.float32).newbyteorder('>');
    M = np.zeros([16,600,600])
    for i in range(16):
        M[i,:,:] = np.flipud(raw[5+i::16][0:(600*600)].reshape([600,600]).T)
    return M

def Rusinkiewicz(wi,wo,n):
    
    if np.isnan(wi).any() or np.isnan(wo).any() or np.isnan(n).any():
        return np.nan, np.nan, np.nan,np.nan
    elif np.array_equal(n, wo):
        b = np.array([-1,0,0])
    else:
        b = np.cross(n, wo) / np.linalg.norm(np.cross(n,wo))
    
    t = np.cross(b, n) / np.linalg.norm(np.cross(b,n))
    
    U = np.array([t, b, n]).T
    h = (wo-wi) / np.linalg.norm(wo-wi)
    hprime = np.linalg.inv(U)@h
    theta_h = np.arctan2(np.sqrt(hprime[0]**2+hprime[1]**2), hprime[2])
    phi_h = np.arctan2(hprime[1], hprime[0])
    
    d = np.linalg.inv(U) @ sc.spatial.transform.Rotation.from_rotvec(-theta_h * b).as_matrix() @ sc.spatial.transform.Rotation.from_rotvec(-phi_h * n).as_matrix() @ -wi
    
    theta_d = np.arctan2(np.sqrt(d[0]**2+d[1]**2), d[2])
    phi_d = -np.arctan2(d[1], d[0])
    
    return theta_h, phi_h, theta_d, phi_d
        
        
        
        
def SphereAngles(aoc, x, y, radius, pixel_ratio, cam_dist=25, src_dist = 66):
    
    cam_loc = np.array([0,0,cam_dist+radius])
    pix_pos = np.array([x * pixel_ratio, y * pixel_ratio, 0])
    
    w_o = (cam_loc - pix_pos) / np.linalg.norm(cam_loc-pix_pos)
    delta = np.dot(-w_o, cam_loc)**2 - np.linalg.norm(cam_loc)**2 + radius**2
    
    
    if delta < 0:
        pos = np.nan
        return np.nan, np.nan, np.nan, np.nan
    else:
        l = - np.dot(-w_o, cam_loc) - np.sqrt(delta)
        pos = cam_loc - l * w_o
        
        # Assuming AOC is in degrees
        src_loc = (src_dist + radius) * np.array([np.sin(aoc*np.pi/180), 0, np.cos(aoc*np.pi/180)])
        
        w_i = (pos - src_loc)/np.linalg.norm(pos - src_loc)
        n = pos / np.linalg.norm(pos)
        return Rusinkiewicz(w_i, w_o, n)   

def pBRDFTableRounding(coords_list):
    [theta_h, phi_h, theta_d, phi_d] = coords_list
    if np.isnan(theta_h) or np.isnan(phi_h) or np.isnan(theta_d) or np.isnan(phi_d):
        return np.nan, np.nan, np.nan
    else:
        th = np.round(90*np.sqrt(theta_h*2/np.pi))
        td = np.round(theta_d*180/np.pi)
        pd = np.round(360 * np.interp(phi_d, (-np.pi, np.pi), (0,1)))
        
        return th, td, pd





#%% Radial masking

sphere_directory = 'P:\\Projects\\Oculus\\Data Sample Library\\RGB950 Polarimeter\\Measurements\\2023\\forschner_cbox\\Spheres\\red'


mm = readCMMI('{}\\sls_sphere_red_451_20.cmmi'.format(sphere_directory))

# radius in inches converted to cm
R = 2.54 

# indices for edges of sphere

xEdge1 = 113
xEdge2 = 510
yEdge1 = 71
yEdge2 = 461

# Plot to aid in alignment of sphere edges

plt.imshow(mm[0, yEdge1:yEdge2,xEdge1:xEdge2], cmap='gray', vmin = 0, vmax = 0.2)
plt.title('Mask Alignment Tool')
plt.show()

#%%

# mean diameter of sphere in pixels
pixD = np.mean([xEdge2-xEdge1, yEdge2-yEdge1])
rho = 2 * R / pixD

xOffset = np.around(np.mean([xEdge2, xEdge1]))
yOffset = np.around(np.mean([yEdge2, yEdge1]))

mask = np.zeros((600,600))

for i in range(600):
    for j in range(600):
        if np.sqrt((i - xOffset)**2 + (j - yOffset)**2) <=np.floor(pixD/2):
            mask[j,i] = 1
        else:
            mask[j,i] = np.nan


# Plot to verify application of radial mask

fig, axs = plt.subplots(ncols=2)
fig.suptitle('Before / After Mask')
axs[0].imshow(mm[0,:,:], cmap='gray', vmin = 0, vmax = 0.2)
axs[1].imshow(mm[0,:,:]*mask, cmap='gray', vmin = 0, vmax = 0.2)
plt.show()

   
#%% intensity masking

# importing sphere data
mms662 = np.zeros((16,600,600,15))
for i in range(15):
    mms662[:,:,:,i] = readCMMI('{}\\sls_sphere_red_662_{}.cmmi'.format(sphere_directory, i*5+20))

# assign intensity threshold
threshold = 5.0


shadow_mask = np.ones((600,600,15))

# Set value in mask as intederminate if below threshold
for k in range(15):
    shadow_mask[:,:,k] = mask
    for i in range(600):
        for j in range(600):
            if mms662[0,j,i,k]*mask[j,i] != np.nan and mms662[0,j,i,k]*mask[j,i] < threshold:
                shadow_mask[j,i,k] = np.nan
            
# Plot for verification

fig = plt.figure()
plt.imshow(mms662[0,:,:,1]*shadow_mask[:,:,1])

#%% Bin indexing

aoc = np.arange(20,91,5)
coord = np.zeros((3,600,600,15), dtype=np.int16)

for qq in range(15):
    for i in range(600):
        for j in range(600):
            if np.isnan(shadow_mask[j,i,qq]):
                coord[:,j,i,qq] = -1
            else:
                
                # generating Rusinkiewicz coordinates from sphere
                coords_temp =np.array(pBRDFTableRounding(SphereAngles(aoc[qq], i-int(xOffset), j-int(yOffset), R, rho)))
                coords_temp[np.isnan(coords_temp)] = -1
                coord[:,j,i,qq] = coords_temp
    
#%% Filesave

# file save directory & filename
save_file_loc = './SphereGeometriesRed/'
save_file_name = 'sphere_pBRDF_index' # filename will include _ followed by aoc

for g in range(15):
    coord[:,:,:,g].tofile('{}{}_{}.bin'.format(save_file_loc, save_file_name, aoc[g]))