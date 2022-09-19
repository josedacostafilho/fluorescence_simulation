import json
import argparse
from easydict import EasyDict
from time import time

from data_generation import *


if __name__ == '__main__':

	start = time()
	
	parser = argparse.ArgumentParser()
	parser.add_argument('-c', '--config', default='configs/cfg2D.json')
	parser.add_argument('-m', '--mode', default="video")
	args = parser.parse_args()

	with open(args.config) as f:
        	cfg = EasyDict(json.load(f))

	model = Generate(cfg)
	
	print("*Start*")
	print("*Generating trajectories*")
	trajectory_list, motion_parameters_list = model.trajectory_generate()
	
	if args.mode == "video":
		frame_trajectory_list, frame_motion_parameters_list = model.video_trajectory_interpolate_list(trajectory_list,motion_parameters_list)
	elif args.mode == "pictures":
		frame_trajectory_list = model.picture_trajectory_list()
	else:
		raise NotImplementedError(args.mode)
	
	print("*Generating shapes*")
	shape_list = model.initialization_routine(frame_trajectory_list) 

	shape_list = model.propagation_routine(shape_list,frame_trajectory_list)
	
	print("*Making frames*")

	model.video_make_frames(frame_trajectory_list,shape_list)
	
	print("*PSF convolution*")
	
	model.video_PSF_convolve()
	
	print("*Adding noise*")
	
	model.video_noise_list()
	
	total_time = time()-start
	
	print("*Completed after {} seconds*".format(total_time))
	
	
	
	
