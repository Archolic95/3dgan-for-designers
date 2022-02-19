
'''
utils.py

Some utility functions

'''

import scipy.ndimage as nd
import scipy.io as io
import matplotlib
import params
import binvox_rw
# if params.device.type != 'cpu':
#     matplotlib.use('Agg')
from datetime import datetime
import cv2
import matplotlib.pyplot as plt
import skimage.measure as sk
from mpl_toolkits import mplot3d
import matplotlib.gridspec as gridspec
import numpy as np
from torch.utils import data
from torch.autograd import Variable
import torch
import os
import pickle

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure

def getVoxelFromMat(path, cube_len=64):
    if cube_len == 32:
        voxels = io.loadmat(path)['instance'] # 30x30x30
        voxels = np.pad(voxels, (1, 1), 'constant', constant_values=(0, 0))

    else:
        # voxels = np.load(path) 
        # voxels = io.loadmat(path)['instance'] # 64x64x64
        # voxels = np.pad(voxels, (2, 2), 'constant', constant_values=(0, 0))
        # print (voxels.shape)
        voxels = io.loadmat(path)['instance'] # 30x30x30
        voxels = np.pad(voxels, (1, 1), 'constant', constant_values=(0, 0))
        voxels = nd.zoom(voxels, (2, 2, 2), mode='constant', order=0)
        # print ('here')
    print (voxels.shape)
    return voxels


def getVFByMarchingCubes(voxels, threshold=0.5):
    v, f = sk.marching_cubes_classic(voxels, level=threshold)
    return v, f


def plotVoxelVisdom(voxels, visdom, title):
    v, f = getVFByMarchingCubes(voxels)
    visdom.mesh(X=v, Y=f, opts=dict(opacity=0.5, title=title))


def SavePloat_Voxels(voxels, path, iteration):
    voxels = voxels[:8].__ge__(0.5)
    fig = plt.figure(figsize=(32, 16))
    gs = gridspec.GridSpec(2, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(voxels):
        x, y, z = sample.nonzero()
        ax = plt.subplot(gs[i], projection='3d')
        ax.set_proj_type('ortho')
        ax.scatter(x, y, z, zdir='z', c='red')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        # ax.set_aspect('equal')
    # print (path + '/{}.png'.format(str(iteration).zfill(3)))
    plt.savefig(path + '/{}.png'.format(str(iteration).zfill(3)), bbox_inches='tight')
    plt.close()


class ShapeNetDataset(data.Dataset):

    def __init__(self, root, args, train_or_val="train"):
        
        
        self.root = root
        self.listdir = os.listdir(self.root)
        # print (self.listdir)  
        # print (len(self.listdir)) # 10668

        data_size = len(self.listdir)
#        self.listdir = self.listdir[0:int(data_size*0.7)]
        self.listdir = self.listdir[0:int(data_size)]
        
        print ('data_size =', len(self.listdir)) # train: 10668-1000=9668
        self.args = args

    def __getitem__(self, index):
        with open(self.root + self.listdir[index], "rb") as f:
            # volume = np.asarray(getVoxelFromMat(f, params.cube_len), dtype=np.float32)
            # print (volume.shape)
            volume = np.asarray(binvox_rw.read_as_3d_array(f).data, dtype=np.float32)
        return torch.FloatTensor(volume)

    def __len__(self):
        return len(self.listdir)


def generateZ(args, batch):

    if params.z_dis == "norm":
        Z = torch.Tensor(batch, params.z_dim).normal_(0, 0.33).to(params.device)
    elif params.z_dis == "uni":
        Z = torch.randn(batch, params.z_dim).to(params.device).to(params.device)
    else:
        print("z_dist is not normal or uniform")

    return Z

def PlotSingleVoxel(voxels, figsize=(32, 32), axisoff=False, edgecolor = 'teal', facecolor='deepskyblue', alpha = 0.5, linewidth = 0.2):
    voxels = voxels.__ge__(0.5)
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, 1)
    gs.update(wspace=0.05, hspace=0.05)

    x, y, z = voxels.nonzero()
    ax = plt.subplot(gs[0], projection='3d')
    ax.set_proj_type('ortho')
    ax.voxels(voxels, 
    facecolor=facecolor,
    edgecolor=edgecolor,
    alpha = alpha,
    linewidth =linewidth)
    #ax.set_xticklabels([])
    #ax.set_yticklabels([])
    # ax.set_aspect('equal')
    if axisoff:
        ax.set_axis_off()

    # print (path + '/{}.png'.format(str(iteration).zfill(3)))
    # plt.savefig(path + '/{}.png'.format(str(iteration).zfill(3)), bbox_inches='tight')
    # plt.close()
    plt.show()

def PlotMarchingCubeMesh(voxels, path=None,cubelen=64,  edgecolor = 'teal', facecolor='deepskyblue', alpha = 0.5, linewidth = 0.2):
    verts, faces, normals, values = measure.marching_cubes_lewiner(voxels, 0)
    # faces = faces
    # Display resulting triangular mesh using Matplotlib. This can also be done
    # with mayavi (see skimage.measure.marching_cubes_lewiner docstring).
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_proj_type('ortho')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces])
    mesh.set_edgecolor(edgecolor)
    mesh.set_facecolor(facecolor)
    mesh.set_alpha(alpha)
    mesh.set_linewidth(linewidth)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, cubelen)  # a = 6 (times two for 2nd ellipsoid)
    ax.set_ylim(0, cubelen)  # b = 10
    ax.set_zlim(0, cubelen)  # c = 16

    
    NOW = datetime.now() # current date and time
    TIMESTAMP = NOW.strftime("%m_%d_%Y_%H_%M_%S_%f")

    # Saving Image mode
    if path is not None:
        if not os.path.exists(path):
            os.makedirs(path)
        ax.set_axis_off()
        plt.savefig(path + '/{}.png'.format(str(TIMESTAMP)), bbox_inches='tight')

        faces = faces + 1
        
        thefile = open(path +'/'+TIMESTAMP+'.obj', 'w')
        for item in verts:
            thefile.write("v {0} {1} {2}\n".format(item[0],item[1],item[2]))

        for item in normals:
            thefile.write("vn {0} {1} {2}\n".format(item[0],item[1],item[2]))

        for item in faces:
            thefile.write("f {0}//{0} {1}//{1} {2}//{2}\n".format(item[0],item[1],item[2]))  

        thefile.close()
        return TIMESTAMP
    else:
        plt.tight_layout()
        plt.show()
    plt.close()


def SerializePlyFiles(voxels, file_path, i, j):
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    voxel_grid = voxels > 0.5
    xyz_points = np.nonzero(voxel_grid)[:-1]
    #print(xyz_points)
    full_path = file_path+'vec_'+str(i).zfill(5)+'_'+str(j).zfill(5)+'.ply'
    # print(xyz_points.shape)
    # Write header of .ply file
    fid = open(full_path,'wt')
    fid.write('ply\n')
    fid.write('format ascii 1.0\n')
    fid.write(f'element vertex {xyz_points.shape[0]}\n')
    fid.write('property float x\n')
    fid.write('property float y\n')
    fid.write('property float z\n')
    fid.write('end_header\n')

    # Write 3D points to .ply file
    for i in range(xyz_points.shape[0]):
        fid.write(f'{xyz_points[i,0]} {xyz_points[i,1]} {xyz_points[i,2]}\n')
    fid.close()