import os
import glob
import logging
import numpy as np
import shutil
import sys
from datetime import datetime
DEF_NB=100

if __name__ == "__main__":
    # Handy command line interface (-h for help)
    import argparse

    parser = argparse.ArgumentParser(
        description="Deplace/copy N couples (X,y) of data")
    parser.add_argument("--in_train_ima", required=True,
                        help="Path to input training images")
    parser.add_argument("--in_train_dem", required=True,
                        help="Path to input training dem")
    parser.add_argument("--in_valid_ima", required=True,
                        help="Path to input valid images")
    parser.add_argument("--in_valid_dem", required=True,
                        help="Path to input valid dem")
    parser.add_argument("--out_train", required=True,
                        help="Path to output train images")
    parser.add_argument("--out_valid", required=True,
                        help="Path to output valid images")
    parser.add_argument("--nb", type=int, default=DEF_NB,
                        help="Number of images to move (default : %d)"%DEF_NB)
    parser.add_argument("--move", action="store_true", default=False,
                        help='deplace (and not copy) files')
    args = parser.parse_args()


    #logger = create_logger(log_path=args.log, log_name="base")
    in_source1=os.path.abspath(args.in_train_ima)
    in_source2=os.path.abspath(args.in_train_dem)
    out_source1=os.path.abspath(args.in_valid_ima)
    out_source2=os.path.abspath(args.in_valid_dem)
    in_target = os.path.abspath(args.out_train)
    out_target = os.path.abspath(args.out_valid)
    nb=int(args.nb)
    deplace=args.move

    # reading list of images in list_im
    if os.path.exists(out_source1)==False:
        os.makedirs(out_source1)
    if os.path.exists(out_source2)==False:
        os.makedirs(out_source2)
    if os.path.exists(out_target)==False:
        os.makedirs(out_target)

    list_im=glob.glob('%s/*png'%in_source1)

    nb_data=min(nb,len(list_im))
    #print(nb_data)
    randperm=np.random.permutation(len(list_im))
    if deplace is False:
        for i in range(nb_data):
            _,name=os.path.split(list_im[randperm[i]])
            command='cp %s/%s %s/%s'%(in_source1,name,out_source1,name)
            print(command)
            os.system(command)
            command='cp %s/%s %s/%s'%(in_source2,name,out_source2,name)
            print(command)
            os.system(command)
            command='cp %s/%s %s/%s'%(in_target,name,out_target,name)
            print(command)
            os.system(command)
    else:
        for i in range(nb_data):
            _,name=os.path.split(list_im[randperm[i]])
            command='mv %s/%s %s/%s'%(in_source1,name,out_source1,name)
            print(command)
            os.system(command)
            command='mv %s/%s %s/%s'%(in_source2,name,out_source2,name)
            print(command)
            os.system(command)
            command='mv %s/%s %s/%s'%(in_target,name,out_target,name)
            print(command)
            os.system(command)
