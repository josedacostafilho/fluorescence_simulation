import json
import argparse
from easydict import EasyDict
from time import time

from data_generation import *


if __name__ == '__main__':

    start_total = time()

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='configs/cfg2D.json')
    parser.add_argument('-m', '--mode', default="video")
    parser.add_argument('--rgb', action='store_false')
    args = parser.parse_args()

    with open(args.config) as f:
            cfg = EasyDict(json.load(f))

    model = Generate(cfg)

    print("*Start*")

    print("*Generating trajectories*")
    start = time()
    trajectory_list, motion_parameters_list = model.trajectory_generate()

    if args.mode == "video":
        frame_trajectory_list, frame_motion_parameters_list = model.video_trajectory_interpolate_list(trajectory_list,motion_parameters_list)
    elif args.mode == "pictures":
        frame_trajectory_list = model.picture_trajectory_list()
    else:
        raise NotImplementedError(args.mode)
    print('--- Time = ',time()-start)

    print("*Generating shapes*")
    start = time()
    shape_list = model.initialization_routine(frame_trajectory_list) 
    shape_list = model.propagation_routine(shape_list,frame_trajectory_list)
    print('--- Time = ',time()-start)

    print("*Making frames*")
    start = time()
    model.video_make_frames(frame_trajectory_list,shape_list)
    print('--- Time = ',time()-start)

    print("*PSF convolution*")	
    start = time()
    model.video_PSF_convolve()	
    print('--- Time = ',time()-start)

    print("*Adding noise*")
    start = time()
    model.video_noise_list()
    print('--- Time = ',time()-start)

    if args.rgb == True:
        print("*Making RGB images*")
        start = time()
        model.video_RGB_list()
        print('--- Time = ',time()-start)

    total_time = time()-start_total

    print("*Completed after {} seconds*".format(total_time))




