import cv2
import numpy as np
from skimage import io
import os,glob
import random
import tifffile

from scipy.ndimage.interpolation import rotate
from scipy.special import factorial
import shutil

# def split(word):
#     return [char for char in word]

# def neighbors(c):
#     coords = tuple(c)
#     dim = len(coords)
#     nb = []
#     for i in range(3**dim):
#         ternary=np.base_repr(i,base=3)
#         chars = split(str(ternary).zfill(dim))
#         new = np.asarray(list(map(int, chars)))-1 + coords
#         nb.append(tuple(new))
#     return (nb)

# def add_point(array,coords):
#     nb = neighbors(coords)
#     for i in range(len(nb)):
#         array[nb[i]]=1

def gaussian_mixture(x,u1,s1,a1,u2,s2,a2):
    a = a1*np.exp(-(x-u1)**2/s1) + a2*np.exp(-(x-u2)**2/s2)
    return a/np.sum(a)

def make_dir(dir,replace=True):
    if replace == True:
        if not os.path.isdir(dir):
            os.mkdir(dir)
    if replace == False:
        new_dir = dir
        i = 1
        while os.path.isdir(new_dir):
            i+=1
            new_dir = dir[:-1] + '_'+str(i)+'/'
        os.mkdir(new_dir)
        return new_dir
        
def binarize(array,threshold):
    """
    Makes an array binary: every pixel with intensity >= threshold becomes 1, the rest is 0.
    """
    return (array/np.max(array)) >= threshold

def spherical_to_cartesian(r,input_angles):
    """
    (Hyper) Spherical to cartesian coordinates in N dimensions.
    
    Args:
        r -- radius (dim=1)
        input_angles -- angles (dim=N-1)
    Returns:
        cartesian coordinates x1,...,xN
    """
    angles = np.append(input_angles,0)
    dimension = len(angles)
    table = [angles for i in range(dimension)]
    table = np.asarray(table)
    table = np.sin(np.tril(table,k=-1)) + np.diag(np.cos(angles)) + np.triu(np.ones(dimension),k=1)
    return r*np.prod(table,axis=1)

def shape_block(dimensions):
    """
    Returns a rectangle of ones.
    
    Args:
        dimensions -- N-dimensional tuple
    """
    return np.ones(dimensions)

def shape_ellipsoid(dimensions):
    """
    Returns an ellipsoid.
    """
    radii = np.asarray(dimensions)/2
    mask = np.zeros(dimensions)
    for point in np.ndindex(mask.shape):
        condition = np.sum(((point-radii)/radii)**2)
        if condition<=1:
            mask[point]=1
    return mask

def square_pad(image):
    larger_dim = np.max(image.shape)
    L = int(1.8*larger_dim) #########
    difference = ((np.ones(len(image.shape))*L - image.shape)/2)
    padding = np.stack((difference,difference),axis=1).astype('int')
    return np.pad(image,padding)

def shape_rotate(shape,angles):
    """
    Rotates a given shape by the given angles. Only works for 2 and 3 dimensions.
    """
    shape_max = np.max(shape)
    new_shape = np.copy(shape)/shape_max
    reshape_status = False
    if len(shape.shape)==2:
        new_shape = rotate(new_shape, angles, mode='constant', axes=(0,1), reshape=reshape_status,order=1)
    elif len(shape.shape)==3:
        new_shape = rotate(new_shape, angles[0], mode='constant', axes=(0,1), reshape=reshape_status,order=1)
        new_shape = rotate(new_shape, angles[1], mode='constant', axes=(1,2), reshape=reshape_status,order=1)
        new_shape = rotate(new_shape, angles[2], mode='constant', axes=(0,2), reshape=reshape_status,order=1)
    else:
        print("Rotation in {} dimensions not implemented. Returning original shape".format(len(shape.shape)))
        return shape
    return new_shape*shape_max

def gaussian_noise(dimensions,mean,std):
    """
    Generates Gaussian noise.
    """
    return np.random.normal(mean,std,size=dimensions)

def OF_to_HSV(f01):
    """
    Converts a 2D optical flow vector field to a HSV map (3 channels). Should only be used in 2D.
    """
    dimensions = f01[0].shape
    if len(dimensions)!=2:
        print("Optical flow visualization only works for 2 dimensions. Returning original flow.")
        return f01
    height, width = dimensions
    hsv = np.zeros((height,width,3), dtype=np.uint8)
    hsv[..., 1] = 255
    mag, ang = cv2.cartToPolar(f01[0,:,:], f01[1,:,:])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR) 
    return bgr

def flow_to_HSV(load_dir):
    """
    Converts all optical flow tiff files in load_dir to a HSV tiff. Only for 2D.
    """
    save_dir = load_dir + 'HSV/'
    make_dir(save_dir)
    HSV = []
    for filename in sorted(glob.glob(os.path.join(load_dir, '*.tif'))):
        frame = tifffile.imread(filename)
        new = OF_to_HSV(frame)
        save_filename = filename.replace(load_dir, '')
        tifffile.imwrite(save_dir+save_filename,new)
        
def make_dataset(dir):
    """
    Reads all files inside dir and makes a 2D list of them. Used for SuperSloMo. For it to work, all files inside dir must be images.
    """
    framesPath = []
    # Find and loop over all the clips in root `dir`.
    for index, folder in enumerate(os.listdir(dir)):
        clipsFolderPath = os.path.join(dir, folder)
        # Skip items which are not folders.
        if not (os.path.isdir(clipsFolderPath)):
            continue
        framesPath.append([])
        # Find and loop over all the frames inside the clip.
        for image in sorted(os.listdir(clipsFolderPath)):
            # Add path to list.
            framesPath[index].append(os.path.join(clipsFolderPath, image))
    return framesPath       

def folder_SloMo(parent_dir, load_dir, save_dir,frames_per_clip,train_ratio=0.9):
    """
    Parent_dir: folder where load and save folders are located
    """
    
    load_dir = parent_dir + load_dir
    save_dir = parent_dir + save_dir
    train_dir = save_dir + 'train/'
    val_dir = save_dir + 'validation/'
    make_dir(save_dir)
    make_dir(train_dir)
    make_dir(val_dir)
    
    clip_idx = 0
    for _, folder in enumerate(os.listdir(load_dir)):
        current_dir = os.path.join(load_dir, folder)        
        framesPath = make_dataset(current_dir)
        for i in range(len(framesPath)):
            frames_in_this_folder = len(framesPath[i])
            number_of_clips = frames_in_this_folder // frames_per_clip
            for j in range(number_of_clips):
                if np.random.uniform(0,1)<train_ratio:
                    clip_dir = train_dir+str(clip_idx)+'/'
                else:
                    clip_dir = val_dir+str(clip_idx)+'/'
                clip_idx += 1
                make_dir(clip_dir)
                for k in range(frames_per_clip):
                    source = framesPath[i][j*frames_per_clip+k]
                    shutil.copy2(source,clip_dir)
    
        
# def folder_SloMo(parent_dir, OG_dir, save_dir,frames_per_clip,train_ratio=0.9,initial_clip_idx=0):
#     """
#     Only for SuperSloMo.
#     """
    
#     OG_dir = parent_dir + OG_dir
#     save_dir = parent_dir + save_dir
#     train_dir = save_dir + 'train/'
#     val_dir = save_dir + 'validation/'
#     make_dir(save_dir)
#     make_dir(train_dir)
#     make_dir(val_dir)
    
#     framesPath = make_dataset(OG_dir)
#     clip_idx = initial_clip_idx
#     for i in range(len(framesPath)):
#         frames_in_this_folder = len(framesPath[i])
#         number_of_clips = frames_in_this_folder // frames_per_clip
#         for j in range(number_of_clips):
#             if np.random.uniform(0,1)<train_ratio:
#                 clip_dir = train_dir+str(clip_idx)+'/'
#             else:
#                 clip_dir = val_dir+str(clip_idx)+'/'
#             clip_idx += 1
#             make_dir(clip_dir)
#             for k in range(frames_per_clip):
#                 source = framesPath[i][j*frames_per_clip+k]
#                 shutil.copy2(source,clip_dir)
                
def add_motion_to_coords(coords_initial,displacement):
    """
    Adds the displacements to the initial coordinates of a trajectory.
    """
    return np.concatenate([[coords_initial], coords_initial+np.cumsum(displacement,axis=0)])

def add_motion_to_coords_list(coords_initial_list,displacement_list):
    """
    The same as above but for a whole list.
    """
    return  [add_motion_to_coords(coords_initial_list[i],displacement_list[i]) for i in range(len(coords_initial_list))] 

def spacetime_concatenate(position,time_array):
    return np.concatenate((position,np.expand_dims(time_array,1)),axis=1)

def trajectory_cut_minimum(trajectory_list,motion_parameters_list,min_cut):
    new_trajectory_list = []
    new_motion_parameters_list = []
    for i in range(len(trajectory_list)):
        traj = trajectory_list[i]
        if len(traj) >= min_cut:
            new_trajectory_list.append(traj)
            new_motion_parameters_list.append(motion_parameters_list[i])
    return new_trajectory_list, new_motion_parameters_list

def shape_propagation_static(shape_start,frame_trajectory):
    """
    Shapes don't change as they propagate.
    """
    shape = []
    shape.append(shape_start)
    for t in range(1,len(frame_trajectory)):
        shape.append(shape_start)
    return shape

def shape_propagation_rotate(shape_start,frame_trajectory,angles):
    """
    Shapes rotate as they propagate.
    """
    shape = []
    shape.append(shape_start)
    shape_current = shape_start
    for t in range(1,len(frame_trajectory)):
        shape_current = shape_rotate(shape_current,angles)
        shape.append(shape_current)
    return shape

def read_flo_file(path):
    with open(path, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        assert (202021.25 == magic), 'Magic number incorrect. Invalid .flo file'
        h = np.fromfile(f, np.int32, count=1)[0]
        w = np.fromfile(f, np.int32, count=1)[0]
        data = np.fromfile(f, np.float32, count=2 * w * h)
    # Reshape data into 3D array (columns, rows, bands)
    data2D = np.resize(data, (w, h, 2))
    return data2D

def write_flo_file(filename, flow):
    """
    write optical flow in Middlebury .flo format
    :param flow: optical flow map
    :param filename: optical flow file path to be saved
    :return: None
    """
    f = open(filename, 'wb')
    magic = np.array([202021.25], dtype=np.float32)
    (height, width) = flow.shape[0:2]
    w = np.array([width], dtype=np.int32)
    h = np.array([height], dtype=np.int32)
    magic.tofile(f)
    w.tofile(f)
    h.tofile(f)
    flow.tofile(f)
    f.close()     
    
def save_flow(filename, flow, extension):
    if extension == '.tif':
        tifffile.imsave(filename,flow)
    elif extension == '.flo':
        write_flo_file(filename, flow)





