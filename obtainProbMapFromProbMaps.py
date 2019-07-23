    
'''
Target: Obtain one overall probability map from the single probability maps
Created on Feb. 13, 2018
Author: Dong Nie 
'''



import SimpleITK as sitk
import argparse
import datetime
from multiprocessing import Pool
import os
import h5py
import numpy as np
from skimage import feature
from morpologicalTransformation import denoiseImg_closing,denoiseImg_isolation    
    
# Training settings
parser = argparse.ArgumentParser(description="PyTorch InfantSeg")
parser.add_argument("--how2normalize", type=int, default=4, help="how to normalize the data")
parser.add_argument("--threshold", type=float, default=0.9, help="how to normalize the data")
parser.add_argument("--basePath", default="/home/dongnie/myPytorchCodes/pytorch-SRResNet23D/", type=str, help="base path for this project's data")
parser.add_argument("--dataFolder", default="denseCRFprobMaps/", type=str, help="the name of the data source folder")
parser.add_argument("--save2Folder", default="pelvicSegRegContourH5/", type=str, help="the name of the save2folder")
parser.add_argument("--saveH5FN", default="train_segregcontour_", type=str, help="the prefix of the save file(h5)")
parser.add_argument("--h5FilesListFN", default="trainPelvic_segregcontour_list.txt", type=str, help="the prefix of the save file(h5)")
parser.add_argument("--adaptiveStepRatio", type=int, default=16, help="the adaptively step ratio to help decide proper step")


def main():
    global opt
    opt = parser.parse_args()

    print opt
    ids = range(0,30)

    ids = [1] 
    for ind in ids:
        datafilename0 = 'denseCrf3dProbMapClass_probMaps_net1106_sub{:02d}_0.nii.gz'.format(ind) #provide a sample name of your filename of data here
        datafn0 = os.path.join(opt.basePath+opt.dataFolder, datafilename0)
        datafilename1 = 'denseCrf3dProbMapClass_probMaps_net1106_sub{:02d}_1.nii.gz'.format(ind) #provide a sample name of your filename of data here
        datafn1 = os.path.join(opt.basePath+opt.dataFolder, datafilename1)
        datafilename2 = 'denseCrf3dProbMapClass_probMaps_net1106_sub{:02d}_2.nii.gz'.format(ind) #provide a sample name of your filename of data here
        datafn2 = os.path.join(opt.basePath+opt.dataFolder, datafilename2)
        datafilename3 = 'denseCrf3dProbMapClass_probMaps_net1106_sub{:02d}_3.nii.gz'.format(ind) #provide a sample name of your filename of data here
        datafn3 = os.path.join(opt.basePath+opt.dataFolder, datafilename3)
#         labelfilename='Case%02d_segmentation.nii.gz'%ind  # provide a sample name of your filename of ground truth here
#         labelfn=os.path.join(opt.basePath+opt.dataFolder, labelfilename)
        imgOrg = sitk.ReadImage(datafn0)
        mrimg0 = sitk.GetArrayFromImage(imgOrg)
        imgOrg = sitk.ReadImage(datafn1)
        mrimg1 = sitk.GetArrayFromImage(imgOrg)
        imgOrg = sitk.ReadImage(datafn2)
        mrimg2 = sitk.GetArrayFromImage(imgOrg)
        imgOrg = sitk.ReadImage(datafn3)
        mrimg3 = sitk.GetArrayFromImage(imgOrg)
        
        mrimg = mrimg1 + mrimg2 + mrimg3
        

        volout = sitk.GetImageFromArray(mrimg)
        sitk.WriteImage(volout, opt.basePath+opt.dataFolder+'probMap_model0110_sub{:02d}'.format(ind)+'.nii.gz')  
        
if __name__ == '__main__':
#     testGradients()     
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main()
