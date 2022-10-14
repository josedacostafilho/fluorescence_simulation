import cv2
from PIL import Image
import numpy as np
from skimage import io
import os,glob
import random
from fbm import fgn
import tifffile
import pickle
from easydict import EasyDict
from time import time

from scipy.signal import oaconvolve
from scipy.ndimage.interpolation import rotate
from scipy.ndimage import gaussian_filter
                
from utils import *

class Generate():
    """
    Class that does pretty much everything.
    """
    
    def __init__(self,cfg):        
        self.cfg = cfg
        
        if not hasattr(self.cfg,"output_dir"):
            print("Output directory not specified. Default = data/test/ ")
            self.cfg.output_dir = 'data/test/'
        if not hasattr(self.cfg,"skew"):
            print("Skewing of data not specified. Default = False")
            self.cfg.skew = False
        if self.cfg.skew == True and not hasattr(self.cfg,"skew_step"):
            print("Skew step not given. Default = 2")
            self.cfg.skew_step = 2 
        if not hasattr(self.cfg,"N_frames"):
            print("Number of frames not specified. Default = 100")
            self.cfg.N_frames = 100     
        if not hasattr(self.cfg,"padding_initialization"):
            print("Initialization padding not specified. Default = 0")
            self.cfg.padding_initialization = [0 for i in range(len(self.cfg.resolution_SK))]
                  
        self.cfg.output_dir = make_dir(self.cfg.output_dir,replace=False)       
        self.cfg.resolution_SK = np.asarray(self.cfg.resolution_SK)
        self.resolution = np.copy(self.cfg.resolution_SK)
        if self.cfg.skew==True:
            self.resolution[-1] = self.cfg.resolution_SK[-1]+(self.resolution[0]-1)*self.cfg.skew_step
        
        self.dim = len(self.resolution)
        
    def coordinates_random(self):
        """
        Generates N_objects coordinates, obeying the initialization bounds.
        """
        lower_bound = np.asarray(self.cfg.padding_initialization)
        upper_bound = self.resolution - lower_bound
        coordinates = [np.random.randint(lower_bound[i],upper_bound[i],size=self.cfg.N_objects_start) for i in range(self.dim)]
        return np.transpose(coordinates)
    
    def time_array_random(self):
        start_time, end_time = np.clip(np.sort(np.random.uniform(-0.1,1.1,size=2)),0,1) 
        if not hasattr(self.cfg,"initial_occupancy") and np.random.uniform(0,1) < self.cfg.initial_occupancy:
            start_time = 0
        time_points = np.maximum(int(self.cfg.time_resolution*(end_time-start_time)),3)
        time_array = np.sort(np.random.uniform(start_time,end_time,size=time_points))
        time_array[0], time_array[-1] = start_time, end_time
        return time_array
    
    def time_array_complete(self):
        return np.linspace(0,1,self.cfg.time_resolution,endpoint=True)
    
    def time_array_list_function(self):
        """
        Generates time arrays for all trajectories. The number of trajectories might change afterwards due to splitting/disappearing/merging.
        """
        if not hasattr(self.cfg,"time_resolution"):
            print("No time resolution specified. Using default = 50 ")
            self.cfg.time_resolution = 50        
        if not hasattr(self.cfg,"time_array"):
            print("No time array function specified. Using default = random ")
            function = self.time_array_random
        elif self.cfg.time_array == "random":
            function = self.time_array_random
        elif self.cfg.time_array == "complete":
            function = self.time_array_complete
        else:
            raise NotImplementedError(self.cfg.time_array)
    
        return [function() for i in range(self.cfg.N_objects_start)]
    
    def displacement_linear(self,time_array,velocity_cartesian):
        return np.transpose([np.diff(time_array)*velocity_cartesian[i] for i in range(velocity_cartesian.shape[0])])
    
    def displacement_brownian(self,time_array,diffusion_constant,alpha):
        displacement = np.zeros((len(time_array)-1,self.dim))
        for i in range(self.dim):
            displacement[:,i] = np.sqrt(2*diffusion_constant)*fgn(n=int(len(time_array)-1), hurst=0.5*alpha, length=int(len(time_array)-1), method="daviesharte")*np.diff(time_array)
        return displacement    
    
    def motion_routine_uniform(self,time_array):
        alpha = np.random.uniform(self.cfg.alpha_bounds[0],self.cfg.alpha_bounds[1])
        diffusion_constant = np.random.uniform(self.cfg.diffusion_constant_bounds[0],self.cfg.diffusion_constant_bounds[1])
        
        directed_speed = np.random.uniform(self.cfg.directed_speed_bounds[0],self.cfg.directed_speed_bounds[1])
        directed_angles = np.random.uniform(self.cfg.directed_angles_bounds[0],self.cfg.directed_angles_bounds[1])
        directed_angles = np.asarray(directed_angles)
        velocity_cartesian = spherical_to_cartesian(directed_speed,directed_angles)
        
        motion_parameters = {
            'diffusion_constant':diffusion_constant,
            'alpha':alpha,
            'directed_speed':directed_speed,
            'directed_angles':directed_angles
        }
        motion = self.displacement_brownian(time_array,diffusion_constant,alpha) + self.displacement_linear(time_array,velocity_cartesian)
        return motion, motion_parameters
            
    def motion_routine_gaussian(self,time_points): 
        pass

    def trajectory_cut(self,trajectory,case):
        """
        Cuts trajectories according to some criterion.
        """
        
        if case=="boundaries":
            lower_bound = np.asarray(self.cfg.padding_propagation)
            upper_bound = self.resolution - lower_bound
            def criterion(point):
                return np.any(point <= lower_bound) or np.any(point >= upper_bound)
        if case=="skew":
            if self.dim!=3:
                return trajectory
            step = self.cfg.skew_step
            def criterion(point):           
                z, y, x = point
                return x < step*z or x > self.cfg.resolution_SK[2]+step*z

        new_trajectories = []
        last_cut = 0
        for t in range(trajectory.shape[0]):
            point = trajectory[t,:-1]
            if criterion(point):
                new_trajectories.append(trajectory[last_cut:t])
                last_cut = t+1
        if len(trajectory)-last_cut > 0:
            new_trajectories.append(trajectory[last_cut:])
        return new_trajectories
        
    def trajectory_cut_list(self,trajectory_list,motion_parameters_list,case):
        new_trajectories = []
        new_parameters = []
        for i in range(len(trajectory_list)):
            new_traj = self.trajectory_cut(trajectory_list[i],case)
            for j in range(len(new_traj)):
                new_trajectories.append(new_traj[j])
                new_parameters.append(motion_parameters_list[i])
        return new_trajectories, new_parameters     
    
    def trajectory_generate(self):
        """
        Generates trajectories.
        """        
        if not hasattr(self.cfg,"padding_propagation"):
            print("Propagation padding not specified. Default = 0")
            self.cfg.padding_propagation =   [0 for i in range(self.dim)]       
                  
        if not hasattr(self.cfg,"motion_routine"):
            print("No motion routine function specified. Using default = uniform.")
            function = self.motion_routine_uniform
        elif self.cfg.motion_routine == "uniform":
            function = self.motion_routine_uniform
        elif self.cfg.motion_routine == "gaussian":
            function = self.motion_routine_gaussian
        else:
            raise NotImplementedError(self.cfg.motion_routine)

        if not hasattr(self.cfg,"N_objects_start"):
            print("Initial number of objects not specified. Default = 10")
            self.cfg.N_objects_start = 10
        coords_initial_list = self.coordinates_random()
        time_array_list = self.time_array_list_function()   
        displacement_list = []
        motion_parameters_list = []
        for i in range(self.cfg.N_objects_start):
            displacement, motion_parameters = function(time_array_list[i])
            displacement_list.append(displacement)
            motion_parameters_list.append(motion_parameters)   

        position_list = add_motion_to_coords_list(coords_initial_list,displacement_list) 
            
        trajectory_list = [spacetime_concatenate(position_list[i],time_array_list[i]) for i in range(len(position_list))]         
        trajectory_list, motion_parameters_list = self.trajectory_cut_list(trajectory_list,motion_parameters_list,"boundaries")
        if self.cfg.skew==True:
            trajectory_list, motion_parameters_list = self.trajectory_cut_list(trajectory_list,motion_parameters_list,"skew")   
        trajectory_list, motion_parameters_list = trajectory_cut_minimum(trajectory_list, motion_parameters_list,1)
        
        with open(self.cfg.output_dir+'trajectory_list', 'wb') as f:
            pickle.dump(trajectory_list,f)
        with open(self.cfg.output_dir+'motion_parameters_list', 'wb') as f:
            pickle.dump(motion_parameters_list,f)
        return trajectory_list, motion_parameters_list 
    
    def trajectory_load(self,change_dir=False,load_dir=''):
        if change_dir == False:
            load_dir = self.output_dir
        with open(load_dir+'trajectory_list', 'rb') as f:
            trajectory_list = pickle.load(f)
        with open(load_dir+'motion_parameters_list', 'rb') as f:
            motion_parameters_list = pickle.load(f)
        return trajectory_list, motion_parameters_list
    
    def intensity_constant_factor(self,shape_mask):
        """
        Random constant factors for intensities according to a certain distribution.
        """
        if self.cfg.intensity_distribution == "uniform":
            if not hasattr(self.cfg,"intensity_bounds"):
                print("No intensity bounds given for uniform distribution. Default = [0,1]")
                self.cfg.intensity_bounds = [0,1]
            intensity = np.random.uniform(self.cfg.intensity_bounds[0],self.cfg.intensity_bounds[1])
        elif self.cfg.intensity_distribution == "gaussian":
            if not hasattr(self.cfg,"intensity_gaussian"):
                print("No intensity mean and variance given for Gaussian distribution. Default = [0.5,0.1]")
                self.cfg.intensity_gaussian = [0.5,0.2]
            intensity = np.random.normal(self.cfg.intensity_gaussian[0],self.cfg.intensity_gaussian[1])
            intensity = np.clip(intensity,0,1)
        elif self.cfg.intensity_distribution == "gmm":
            if not hasattr(self.cfg,"intensity_gmm"):
                print("No parameters given for GMM distribution. Default = [100,0.3,0.05,1.5,0.9,0.005,1]")
                self.cfg.intensity_gmm = [100,0.3,0.05,1.5,0.9,0.005,1]
            p = self.cfg.intensity_gmm
            sample_space = np.linspace(0,1,p[0])
            distribution = gaussian_mixture(sample_space,p[1],p[2],p[3],p[4],p[5],p[6])
            intensity = np.random.choice(sample_space,p=distribution)
        else:
            raise NotImplementedError(self.cfg.intensity_distribution)
        return intensity*shape_mask
                  
    def shape_ellipsoid_dimensions(self):
        if self.cfg.ellipsoid_distribution == "uniform":
            dimensions = np.random.randint(self.cfg.ellipsoid_dimensions_bounds[0], self.cfg.ellipsoid_dimensions_bounds[1], size=self.dim)
        elif self.cfg.ellipsoid_distribution == "gaussian":    
            dimensions = np.random.normal(self.cfg.ellipsoid_dimensions_gaussian[0], self.cfg.ellipsoid_dimensions_gaussian[1], size=self.dim)
            dimensions = np.clip(dimensions,self.cfg.ellipsoid_dimensions_bounds[0], self.cfg.ellipsoid_dimensions_bounds[1])
        elif self.cfg.ellipsoid_distribution == "gmm":
            if not hasattr(self.cfg,"ellipsoid_dimensions_gmm"):
                print("No parameters given for GMM ellipsoid distribution. Default = [1000,4.3,0.1,3,16,6,1]")
                self.cfg.ellipsoid_dimensions_gmm = [1000,4.3,0.1,8,16,6,1]
            p = self.cfg.ellipsoid_dimensions_gmm
            sample_space = np.linspace(self.cfg.ellipsoid_dimensions_bounds[0], self.cfg.ellipsoid_dimensions_bounds[1],p[0])
            distribution = gaussian_mixture(sample_space,p[1],p[2],p[3],p[4],p[5],p[6])
            dimensions = np.random.choice(sample_space,size=self.dim,p=distribution) 
        else:
            raise NotImplementedError(self.cfg.ellipsoid_distribution)
        return dimensions.astype('int')
    
    def shape_initialize(self):
        if self.cfg.shapes == "points":
            dimensions = np.ones(self.dim,dtype='int')*self.cfg.point_source_diameter ######
            shape_mask = shape_ellipsoid(dimensions) 
            return square_pad(shape_mask)
        elif self.cfg.shapes == "ellipsoids":
            dimensions = self.shape_ellipsoid_dimensions()
            shape_mask = shape_ellipsoid(dimensions)
            shape_mask = square_pad(shape_mask)
            if self.dim==2:
                angles = np.random.uniform(0,360)
            if self.dim==3:
                angles = np.random.uniform(0,360,size=3)   
            return shape_rotate(shape_mask,angles)
        else:
            raise NotImplementedError(self.cfg.shapes)
            
    def initialization_routine(self,frame_trajectory_list):      
        """
        Generates initial shapes and intensities for all objects.
        """                  
        return [self.intensity_constant_factor(self.shape_initialize()) for i in range(len(frame_trajectory_list))]
    
    def intensity_propagate(self,shape_list,frame_trajectory_list):
        if not hasattr(self.cfg,"propagation_intensity"):
            print("No intensity propagation routine given. Default = constant.")
            def function(shape,t):
                return shape
        if self.cfg.propagation_intensity == "constant":
            def function(shape,t):
                return shape
        elif self.cfg.propagation_intensity == "bleaching":
            if not hasattr(self.cfg,"half_time"):
                print("Bleaching half time not given. Default = 420")
                self.cfg.half_time = 420
            def function(shape,t):
                return shape*np.exp(-t/self.cfg.half_time)
        else:
            raise NotImplementedError(self.cfg.propagation_intensity)
           
        return [[function(shape_list[i][t],t) for t in range(len(shape_list[i]))] for i in range(len(shape_list))]
    
    def shape_propagate(self,shape_start_list,frame_trajectory_list):
        if self.cfg.propagation_shape == "static":
            function = shape_propagation_static
        elif self.cfg.propagation_shape == "rotate":
            def function(shape_start,frame_trajectory):
                if not hasattr(self.cfg,"shape_rotation_max"):
                    print("No shape rotation max angle given. Default = 10 degrees.")
                    self.cfg.shape_rotation_max = 10
                if self.dim==2:
                    angles = np.random.uniform(0,self.cfg.shape_rotation_max)
                elif self.dim==3:
                    angles = np.random.uniform(0,self.cfg.shape_rotation_max,size=3)
                else:
                    print("Shape rotation is only valid for 2D and 3D. Returning static propagation.")
                    return shape_propagation_static
                return shape_propagation_rotate(shape_start,frame_trajectory,angles)
        else:
            raise NotImplementedError(self.cfg.propagation_shape)
    
        return [function(shape_start_list[i],frame_trajectory_list[i]) for i in range(len(frame_trajectory_list))]
    
    def propagation_routine(self,shape_list,frame_trajectory_list):        
        shape_list = self.shape_propagate(shape_list,frame_trajectory_list)
        return self.intensity_propagate(shape_list,frame_trajectory_list)       
    
    def frame_deskew(self,frame,noise=False):
        """
        Deskews frames. Note that we first originate deskewed frames (by erasing the corners of the 3D volume),
        then afterwards we skew. So it's in the opposite order of what happens in real life.
        """
        if noise==True:
            offset = self.cfg.gaussian_mean
        else:
            offset = 0
        if self.dim!=3:
            return frame
        else:
            new_frame = np.copy(frame)
            step = self.cfg.skew_step
            for z in range(self.resolution[0]):
                new_frame[z,:,:step*z] = offset
                new_frame[z,:,step*z+int(self.cfg.resolution_SK[2]):]=offset
            return new_frame
        
    def flow_deskew(self,flow):
        """
        Deskews optical flow.
        """
        return np.asarray([self.frame_deskew(flow[i]) for i in range(self.dim)])
        
    def frame_skew(self,frame):
        """
        Skews frames from the deskewed data.
        """
        if self.dim!=3:
            return frame
        else:
            new_frame = np.zeros(self.cfg.resolution_SK,dtype=frame.dtype)
            step = self.cfg.skew_step
            for z in range(self.cfg.resolution_SK[0]):
                new_frame[z] = frame[z,:,step*z:step*z+int(self.cfg.resolution_SK[2])]
            return new_frame
        
    def flow_skew(self,flow):
        return np.asarray([self.frame_skew(flow[i]) for i in range(self.dim)])
    
    def video_trajectory_interpolate(self,trajectory):
        """
        Trajectories are defined over an abstract time array. Now they are interpolated to the actual number of frames.
        This allows us to generate the same data but with finer/coarser timescales.
        """
        time_array = trajectory[:,-1]
        frame_time_array = np.unique(np.rint(time_array*self.cfg.N_frames).astype('int'))
        frame_time_array = np.arange(np.min(frame_time_array),np.max(frame_time_array))
        length = len(frame_time_array)
        frame_trajectory = np.zeros((length,trajectory.shape[1]))
        for i,n in enumerate(frame_time_array):
            t = n/self.cfg.N_frames
            if t <= np.min(time_array):
                frame_trajectory[i,:-1] = trajectory[0,:-1]
                frame_trajectory[i,-1] = frame_time_array[0]
            elif t >= np.max(time_array):
                frame_trajectory[i,:-1] = trajectory[-1,:-1]
                frame_trajectory[i,-1] = frame_time_array[-1]
            else:
                T_closest = np.argmin(np.abs(time_array-t))
                t_closest = time_array[T_closest]
                if t_closest <= t:
                    T_lower,T_upper = T_closest,T_closest+1
                if t_closest > t:
                    T_lower,T_upper = T_closest-1,T_closest
                T_lower = np.maximum(np.minimum( T_lower , trajectory.shape[0]-1 ) , 0)
                T_upper = np.maximum(np.minimum( T_upper , trajectory.shape[0]-1 ) , 0)
                t_lower,t_upper = time_array[T_lower],time_array[T_upper]
                frame_trajectory[i,:-1] = ((t_upper-t)*trajectory[T_lower,:-1] + (t-t_lower)*trajectory[T_upper,:-1])/(t_upper-t_lower)
                frame_trajectory[i,-1] = frame_time_array[i]           
        return np.rint(frame_trajectory).astype('int')
    
    def video_trajectory_interpolate_list(self,trajectory_list,motion_parameters_list):
        if not hasattr(self.cfg,"trajectory_min_cut"):
            print("No minimum length for trajectories given. Default = 1.")
            self.cfg.trajectory_min_cut = 1        
        frame_trajectory_list = [self.video_trajectory_interpolate(trajectory_list[i]) for i in range(len(trajectory_list))]
        frame_trajectory_list, motion_parameters_list = trajectory_cut_minimum(frame_trajectory_list, motion_parameters_list,self.cfg.trajectory_min_cut)
        with open(self.cfg.output_dir+'frame_trajectory_list', 'wb') as f:
            pickle.dump(frame_trajectory_list,f)
        with open(self.cfg.output_dir+'frame_motion_parameters_list', 'wb') as f:
            pickle.dump(motion_parameters_list,f)
        return frame_trajectory_list, motion_parameters_list
    
    def picture_trajectory_list(self):
        N_frames, N_objects = self.cfg.N_frames, self.cfg.N_objects_start
        lower_bound = np.asarray(self.cfg.padding_initialization)
        upper_bound = self.resolution - lower_bound
        frame_trajectory_list = []
        for n in range(N_objects):
            for t in range(N_frames):
                traj = []
                for i in range(self.dim):
                    x = np.random.randint(lower_bound[i],upper_bound[i],size=1)
                    traj.append(x.squeeze().tolist())
                traj.append(t)
                traj = np.expand_dims(np.asarray(traj),0)
                frame_trajectory_list.append(traj)
        return frame_trajectory_list
    
    def video_make_frames(self,frame_trajectory_list,shape_list):   
        """
        Makes the frames. Very long function, maybe I should break it down into smaller ones.
        Generates:
        Ground truth: each object mask, with its intensity
        Segmentation: each object mask with unit intensity. Kinda redundant I now.
        OF: optical flow from current to next frame.
        rOF: optical flow from current to previous frame.
        TO DO: optical flow only takes into account translation. Should be improved to consider rotation as well.
        """
        if not hasattr(self.cfg,"optical_flow"):
            self.cfg.optical_flow = True 
        if not hasattr(self.cfg,"binarize_threshold"):
            self.cfg.binarize_threshold = 0.1 
        if not hasattr(self.cfg,"flow_extension"):
            self.cfg.flow_extension = '.flo'   
        with open(self.cfg.output_dir+'shape_list', 'wb') as f:
            pickle.dump(shape_list,f)
        N_objects = len(frame_trajectory_list)
        for n in range(self.cfg.N_frames): 
            start = time()
            n_str = str(n).zfill(4)
            frame_GT = np.zeros(self.resolution)
            frame_seg = np.zeros(self.resolution)
            GT_dir = self.cfg.output_dir + 'video_GT/'
            seg_dir = self.cfg.output_dir + 'video_seg/'
            if self.cfg.optical_flow == True:
                frame_OF = np.asarray([np.zeros(self.resolution) for i in range(self.dim)])
                frame_rOF = np.asarray([np.zeros(self.resolution) for i in range(self.dim)])
                OF_dir = self.cfg.output_dir + 'video_OF/'
                rOF_dir = self.cfg.output_dir + 'video_rOF/'            
            for i in range(N_objects):
                if n in frame_trajectory_list[i][:,-1]:
                    t = np.where(frame_trajectory_list[i][:,-1]==n)[0][0]
                    traj = frame_trajectory_list[i][:,:-1]
                    new_object = np.zeros(self.resolution)
                    if t == traj.shape[0]-1:
                        speed = traj[t]-traj[t]
                    else:
                        speed = traj[t+1]-traj[t]
                    if t == 0:
                        speed_r = traj[t]-traj[t]
                    else:
                        speed_r = traj[t-1] - traj[t]
                    coords = tuple(traj[t].astype('int'))
                    shape = shape_list[i][t]
                    new_object[coords] = 1
                    new_object = np.abs(oaconvolve(new_object,shape,mode='same'))
                    new_object_seg = binarize(new_object,self.cfg.binarize_threshold)
                    new_object_GT  = np.copy(new_object_seg)*np.max(shape) #####################################
                    frame_GT  += new_object_GT
                    frame_seg += new_object_seg
                    if self.cfg.optical_flow == True:
                        new_object_OF  = np.asarray([new_object_seg*np.flip(speed[i]) for i in range(self.dim)])
                        new_object_rOF = np.asarray([new_object_seg*np.flip(speed_r[i]) for i in range(self.dim)])
                        frame_OF  += new_object_OF
                        frame_rOF += new_object_rOF
            frame_GT  = np.nan_to_num(frame_GT).astype('float32')
            frame_seg = np.nan_to_num(frame_seg).astype('float32')            
            make_dir(GT_dir)
            make_dir(seg_dir)
            if self.cfg.optical_flow == True:
                frame_OF  = np.nan_to_num(frame_OF).astype('float32').transpose(1,2,0)
                frame_rOF = np.nan_to_num(frame_rOF).astype('float32').transpose(1,2,0) 
                make_dir(OF_dir)
                make_dir(rOF_dir)
            if self.cfg.skew==True:
                frame_GT  = self.frame_deskew(frame_GT)
                frame_seg = self.frame_deskew(frame_seg)
                frame_GT_SK  = self.frame_skew(frame_GT)
                frame_seg_SK = self.frame_skew(frame_seg)  
                tifffile.imwrite(GT_dir +'frame_'+n_str+'.tif',frame_GT_SK)
                tifffile.imwrite(seg_dir+'frame_'+n_str+'.tif',frame_seg_SK)                
                GT_dir  += 'DS/'
                seg_dir += 'DS/'
                make_dir(GT_dir)
                make_dir(seg_dir)
                if self.cfg.optical_flow == True:
                    frame_OF  = self.flow_deskew(frame_OF)
                    frame_rOF = self.flow_deskew(frame_rOF)
                    frame_OF_SK  = self.flow_skew(frame_OF)
                    frame_rOF_SK = self.flow_skew(frame_rOF) 
                    save_flow(OF_dir +'frame_'+n_str+self.cfg.flow_extension,frame_OF_SK, self.cfg.flow_extension)
                    save_flow(rOF_dir+'frame_'+n_str+self.cfg.flow_extension,frame_rOF_SK, self.cfg.flow_extension)
                    OF_dir  += 'DS/'
                    rOF_dir += 'DS/'
                    make_dir(OF_dir)
                    make_dir(rOF_dir)                   
            tifffile.imwrite(GT_dir +'frame_'+n_str+'.tif',frame_GT)
            tifffile.imwrite(seg_dir+'frame_'+n_str+'.tif',frame_seg)
            if self.cfg.optical_flow == True:
                save_flow(OF_dir +'frame_'+n_str+self.cfg.flow_extension,frame_OF, self.cfg.flow_extension)
                save_flow(rOF_dir+'frame_'+n_str+self.cfg.flow_extension,frame_rOF, self.cfg.flow_extension)
            if n==0:
                total_time = (time()-start)*(self.cfg.N_frames-1)
                print("--- Estimated remaining time = {} seconds ({} hours)".format(total_time,total_time/3600))
            
    def frame_PSF_convolve(self,frame):
        """
        Convolves frames with PSF.
        """
        frame_max = np.max(frame)
        if frame_max==0:
            return frame
        else:
            new_frame = gaussian_filter(frame/frame_max,self.cfg.sigma)
            new_frame = new_frame*frame_max/np.max(new_frame)
            return new_frame
        
    def frame_noise_poisson(self, frame,N_photons_signal):
        """
        Adds multiplicative Poisson noise to a frame.
        """
        if np.max(frame)==0:
            return frame
        else:
            normalized = np.abs(np.copy(frame)*N_photons_signal)
            return np.random.poisson(normalized,normalized.shape)
    
    def frame_noise_gaussian(self,frame):
        """
        Adds additive Gaussian noise to a frame.
        """
        return frame + gaussian_noise(self.resolution,self.cfg.gaussian_mean,self.cfg.gaussian_std)
    
    def video_PSF_convolve(self,GT_dir='video_GT/'):
        """
        Convolve all frames in GT directory with PSF.
        """
        if not hasattr(self.cfg,"sigma"):
            print("PSF sigma not given. Default = 2")
            self.cfg.sigma = 2
        load_dir = self.cfg.output_dir+GT_dir
        if self.cfg.skew==True:
            load_dir+='DS/'
        for filename in sorted(glob.glob(os.path.join(load_dir, '*.tif'))):
            save_filename = filename.replace(load_dir, '')
            frame = tifffile.imread(filename)
            frame = self.frame_PSF_convolve(frame).astype('float32')
            
            save_dir = self.cfg.output_dir + 'video_PSF/'
            make_dir(save_dir)
            if self.cfg.skew==True:
                frame = self.frame_deskew(frame)
                frame_SK = self.frame_skew(frame)
                tifffile.imwrite(save_dir+save_filename,frame_SK)
                save_dir +='DS/'
                make_dir(save_dir)
            tifffile.imwrite(save_dir+save_filename,frame)

    def video_noise(self,SNR,PSF_dir='video_PSF/',global_max=1):
        """
        Adds noise to all frames in PSF directory.
        The gaussian noise is fixed by args.
        The Poisson noise is given such that frames have, on average, the desired SNR
        SNR = (peak-average_noise)/std_noise
        I ignored the std of Poisson noise for convenience (aka laziness)
        global_max: when objects overlap their intensities sum. This makes peaks have values larger than
        N_photons_signal*1. This is a brute force solution to ignore or not these overlaps.
        """
        if not hasattr(self.cfg,"gaussian_mean"):
            print("Gaussian noise mean not given. Default = 100")
            self.cfg.gaussian_mean = 100
        if not hasattr(self.cfg,"gaussian_std"):
            print("Gaussian noise std not given. Default = 3.4")
            self.cfg.gaussian_std = 3.4         
                  
        N_photons_signal = (SNR*self.cfg.gaussian_std)/global_max
        load_dir = self.cfg.output_dir+PSF_dir
        if self.cfg.skew==True:
            load_dir+='DS/'
        for filename in sorted(glob.glob(os.path.join(load_dir, '*.tif'))):
            save_filename = filename.replace(load_dir, '')
            frame = tifffile.imread(filename)
            frame = self.frame_noise_poisson(frame,N_photons_signal)
            frame = self.frame_noise_gaussian(frame).astype('float32')
            save_dir = self.cfg.output_dir + 'SNR_'+str(SNR)+'/'
            make_dir(save_dir)         
            if self.cfg.skew==True:
                frame = self.frame_deskew(frame,noise=True)
                frame_SK = self.frame_skew(frame)
                tifffile.imwrite(save_dir+save_filename,frame_SK)
                save_dir +='DS/'
                make_dir(save_dir)
            tifffile.imwrite(save_dir+save_filename,frame)   
    
    def video_noise_list(self,PSF_dir='video_PSF/',global_max=1):
        if not hasattr(self.cfg,"SNR_list"):
            print("SNR list not specified.")
        else:          
            for SNR in self.cfg.SNR_list:
                self.video_noise(SNR,PSF_dir,global_max)        
                
    def video_RGB(self,video_dir):
        """
        Converts greyscale tiffs to RGB pngs (all 3 channels the same).
        Used for Optical flow models. Only for 2D.
        """
        load_dir = self.cfg.output_dir + video_dir
        if self.dim!=2:
            print('Conversion to RGB is for two dimensional data only!')
        else:
            rgb_dir = self.cfg.output_dir + 'RGB/'
            save_dir = rgb_dir + video_dir
            make_dir(rgb_dir)            
            make_dir(save_dir)
            global_max = []
            for filename in sorted(glob.glob(os.path.join(load_dir, '*.tif'))):
                frame = tifffile.imread(filename)
                global_max.append(np.max(frame))
            global_max = np.max(np.asarray(global_max))
            for filename in sorted(glob.glob(os.path.join(load_dir, '*.tif'))):
                save_filename = filename.replace(load_dir, '')
                save_filename = save_filename.replace('.tif', '')
                frame = tifffile.imread(filename).astype('float')
                frame = (frame*255/global_max)
                frame = np.stack((frame,frame,frame),axis=2).astype('uint8')
                image = Image.fromarray(frame)
                image.save(save_dir+save_filename+'.png')
                  
    def video_RGB_list(self):
        if not hasattr(self.cfg,"SNR_list"):
            print("SNR list not specified.")
        else:          
            for SNR in self.cfg.SNR_list:
                self.video_RGB('SNR_'+str(SNR))
        self.video_RGB('video_GT')                
        self.video_RGB('video_PSF')
                  
    def video_skew(self,load_dir,save_dir):
        """
        Skews a given video.
        """
        make_dir(save_dir)
        for filename in sorted(glob.glob(os.path.join(load_dir, '*.tif'))):
            frame = tifffile.imread(filename)
            frame = self.frame_skew(frame)
            save_filename = filename.replace(load_dir, '')
            tifffile.imwrite(save_dir+save_filename,frame)
            
    def video_SNR(self,video_dir):
        """
        Calculates the SNR of a given video. I don't even use it anymore.
        """
        load_dir = self.cfg.output_dir+video_dir
        SNR = []
        for filename in sorted(glob.glob(os.path.join(load_dir, '*.tif'))):
            frame = tifffile.imread(filename)
            smooth = gaussian_filter(frame,1)
            signal = np.max(smooth)
            ratio = (signal-self.cfg.gaussian_mean)/self.cfg.gaussian_std
            SNR.append(ratio)
        return np.asarray(SNR)
    
    def video_statistics(self,video_dir):
        """
        Returns mean, std and global max of a given video.
        """
        load_dir = self.cfg.output_dir+video_dir
        mean, std, global_max = [], [], []
        for filename in sorted(glob.glob(os.path.join(load_dir, '*.tif'))):
            frame = tifffile.imread(filename)
            mean.append(np.average(frame))
            std.append(np.std(frame))
            global_max.append(np.max(frame))
        mean = np.asarray(mean)
        std = np.asarray(std)
        global_max = np.asarray(global_max)
        return mean, std, global_max
    

    
    
