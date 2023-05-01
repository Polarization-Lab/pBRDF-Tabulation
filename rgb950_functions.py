# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 11:46:23 2022

@authors: qjare, forschner
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from matplotlib.colors import ListedColormap
from rgb950_reconstruction import W_mat

import scipy.linalg as slin

plt.rcParams['animation.ffmpeg_path'] ='C:\\ffmpeg\\bin\\ffmpeg.exe'
FFwriter = anim.FFMpegWriter(fps=10)

#Blue to Red Color scale for S1 and S2
colmap = np.zeros((255,3));
# Red
colmap[126:183,0]= np.linspace(0,1,57);
colmap[183:255,0]= 1; 
# Green
colmap[0:96,1] = np.linspace(1,0,96);
colmap[158:255,1]= np.linspace(0,1,97); 
# Blue
colmap[0:71,2] = 1;
colmap[71:128,2]= np.linspace(1,0,57); 
colmap2 = colmap[128:,:]
colmap = ListedColormap(colmap)


# wavelength params = [theta_A, delta_A, theta_LP, theta_G, delta_G]
wv_947 = [102.894, 68.2212, -1.17552, -104.486, 122.168]
wv_451 = [13.41, 143.13, -0.53, -17.02, 130.01]
wv_524 = [13.41, 120.00, -0.53, -17.02, 124.55]
wv_662 = [13.41, 94.780, -0.53, -17.02, 127.12]

params = [wv_451, wv_524, wv_662, wv_947]

def readRMMD(inputfilename):
    raw = np.fromfile(inputfilename,np.int32).newbyteorder('>');
    [num,xdim,ydim]=raw[1:4]
    out=np.reshape(raw[5:5+num*xdim*ydim],[num,xdim,ydim])
    out = np.flip(out, axis=1)
    return out

def readCMMI(inputfilename):
    raw = np.fromfile(inputfilename,np.float32).newbyteorder('>');
    M = np.zeros([16,600,600])
    for i in range(16):
        M[i,:,:] = np.flipud(raw[5+i::16][0:(600*600)].reshape([600,600]).T)
    return M

def makeRMMDbin(inputfilepath, outputfilepath, wv = wv_947, psg_rot=False):
    '''

    Parameters
    ----------
    inputfilename : string
        Full path to input rmmd file.
    outputfilename : string
        Path for binary file output.
    wv : list, optional
        Five index list of wavelength parameters. See the list at top of rgb950_functions for examples. The default is wv_947.
    psg_rot : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    None.

    '''
    rmmd = readRMMD(inputfilepath)
    n_len = len(rmmd[:,0,0])-1
    rmmd = np.float32(np.reshape(rmmd[:n_len, :, :], [n_len, 360_000]))
    W_plus = np.float32(np.linalg.pinv(W_mat(wv, n_len, psg_rot=psg_rot)))
    mm = np.matmul(W_plus, rmmd).reshape([16,600,600])
    mm.tofile(outputfilepath)

def makeMMbin(inputfilename,outputfilename):
    raw = np.fromfile(inputfilename,np.float32).newbyteorder('>');
    M = np.zeros([16,600,600],np.float32)
    for i in range(16):
        M[i,:,:] = np.flipud(raw[5+i::16][0:(600*600)].reshape([600,600]).T)
    M2=np.reshape(M,(16*600*600))
    M2.tofile(outputfilename)
    
def readMMbin(inputfilename):
    raw = np.fromfile(inputfilename,np.float32);
    M = np.reshape(raw,[16,600,600])
    return M

def MMImagePlot(MM,minval,maxval, title=''):
    f, axarr = plt.subplots(nrows = 4,ncols = 4,figsize=(6, 5))
    f.suptitle(title, fontsize=20)
    
    MM = MM.reshape([4,4,600,600])
    #normalization
    MM = MM/MM[0,0,:,:]
    for i in range(4):
        for j in range(4):
            im=axarr[i,j].imshow(MM[i,j,:,:],cmap=colmap,vmin = minval,vmax=maxval)
            axarr[i,j].set_xticks([])
            axarr[i,j].set_yticks([])
            # im=axarr[i,j].imshow(MM[i,j,:,:], cmap=colmap)
    f.subplots_adjust(right=0.8)
    cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
    f.colorbar(im, cax=cbar_ax)
    # plt.tight_layout()

def get_polarizance(MM):
    '''returns polarizance magnitude and orientation of input mueller matrix'''
    P = MM[:4,:,:]
    
    lin_mag = np.sqrt(P[1]**2 + P[2]**2)/P[0]
    lin_orient = 0.5 * np.arctan2(P[2], P[1])
    
    return lin_mag, lin_orient

def get_diattenuation(MM):
    P = MM[[0,4,8],:,:]
    
    lin_mag = np.sqrt(P[1]**2 + P[2]**2)/P[0]
    lin_orient = 0.5 * np.arctan2(P[2], P[1])
    
    return lin_mag, lin_orient

def updateAnim(frame, rmmd, im, ax):
    im.set_array(rmmd[frame])
    ax.set_title('{}'.format(frame))
    return im

def animRMMD(rmmd, outputfilename = 'recent_animation'):
    '''
    Animates Sequential images contained within an RMMD file. 
    
    [Input]
        rmmd : (string) path from program to rmmd file to open.
    
    [Output]
        Matplotlib Animated plot of sequential eye images.
    '''
    fig = plt.figure(figsize=(4,4))
    fig.suptitle(outputfilename)
    ax = plt.subplot()
    ax.set_xticks([])
    ax.set_yticks([])
    im = ax.imshow(rmmd[0], animated=True)
    fig.colorbar(im)
    ani = anim.FuncAnimation(fig, updateAnim, 40, fargs=(rmmd, im, ax))
    ani.save('./rmmd_videos/{}.mp4'.format(outputfilename), writer = FFwriter)
    return ani

    


def RetardanceVector(MM):
    m00 = MM[0,0]
    M = MM/m00
    D = M[0,1:]
    Dmag = np.linalg.norm(D)
    mD = np.sqrt(1-Dmag**2)*np.identity(3) + (1-np.sqrt(1-Dmag**2))*np.outer(D/Dmag,D/Dmag)
    MD=np.vstack((np.concatenate(([1],D)),np.concatenate((D[:,np.newaxis],mD),axis=1)))
    Mprime = M@np.linalg.inv(MD)
    [mR, mDelta] = slin.polar(Mprime[1:,1:])
    MR = np.vstack((np.concatenate(([1],np.zeros(3))),np.concatenate((np.zeros(3)[:,np.newaxis],mR),axis=1)))
    Tr = np.trace(MR)/2 -1
    if Tr < -1:
        Tr = -1
    elif Tr > 1:
        Tr = 1
    R = np.arccos(Tr)
    Rvec = R/(2*np.sin(R))*np.array([np.sum(np.array([[0,0,0],[0,0,1],[0,-1,0]]) * mR),np.sum(np.array([[0,0,-1],[0,0,0],[1,0,0]]) * mR),np.sum(np.array([[0,1,0],[-1,0,0],[0,0,0]]) * mR)])
    return Rvec

def plot_aolp(MM, cmap='hsv', diatt = 0, axtitle='AoLP'):
    MM = MM.reshape(16, 600, 600)
    if diatt == 1:
        mag, lin = get_diattenuation(MM)
    else:
        mag, lin = get_polarizance(MM)
    fig = plt.figure()
    ax = plt.subplot()
    ax.set_title(axtitle)
    ax.set_xticks([])
    ax.set_yticks([])
    im = ax.imshow(lin, cmap=cmap, vmin = -np.pi/2, vmax = np.pi/2, interpolation='none')
    cb = fig.colorbar(im, )
    cb.ax.set_yticks([-np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2], [r'$-90\degree$', r'$-45\degree$', '0', r'$45\degree$', r'$90\degree$'], fontsize=12)
   
def plot_mag(MM, cmap='viridis', diatt=0, axtitle='Magnitude'):
    MM = MM.reshape(16, 600, 600)
    if diatt == 1:
        mag, lin = get_diattenuation(MM)
    else:
        mag, lin = get_polarizance(MM)
    fig = plt.figure()
    ax = plt.subplot()
    ax.set_title(axtitle)
    ax.set_xticks([])
    ax.set_yticks([])
    im = ax.imshow(mag, cmap=cmap, vmin=0, vmax=1, interpolation='none')
    cb = fig.colorbar(im, )
    cb.ax.set_yticks([0, 0.25, 0.50, 0.75, 1.0], ['0', '0.25', '0.50', '0.75', '1.00'], fontsize=12)


def plot_retardance_linear(ret_vec):
    
    ret_vec = ret_vec.reshape([600, 600, 3])
    
    fig = plt.figure()
    ax = plt.subplot()
    ax.set_title('Linear Retardance')
    ax.set_xticks([])
    ax.set_yticks([])
    im = ax.imshow(ret_vec[:,:,0], cmap='hsv', vmin = -np.pi, vmax = np.pi, interpolation='none')
    cb = fig.colorbar(im,)
    cb.ax.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], [r'$-\pi$', r'$-\frac{\pi}{2}$', '0', r'$\frac{\pi}{2}$', r'$\pi$'], fontsize=12)
    

def plot_retardance_mag(ret_vec):
    
    ret_mag = np.zeros([360_000])
    for jj in range(len(ret_mag)):
        ret_mag[jj] = np.linalg.norm(ret_vec[jj,:])
    ret_mag = ret_mag.reshape([600, 600])
    
    fig = plt.figure()
    ax = plt.subplot()
    ax.set_title('Retardance Magnitude')
    ax.set_xticks([])
    ax.set_yticks([])
    im = ax.imshow(ret_mag, cmap='turbo', vmin = 0, vmax = np.pi, interpolation='none')
    cb = fig.colorbar(im,)
    cb.ax.set_yticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi], ['0', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3 \pi}{4}$', r'$\pi$'], fontsize=12)
    
