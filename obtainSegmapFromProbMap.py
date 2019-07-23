    
'''
Target: Obtain the segmentation maps from the probability maps
You can set the decision boundary as you want
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
parser.add_argument("--dataFolder", default="resTestCha/", type=str, help="the name of the data source folder")
parser.add_argument("--save2Folder", default="pelvicSegRegContourH5/", type=str, help="the name of the save2folder")
parser.add_argument("--saveH5FN", default="train_segregcontour_", type=str, help="the prefix of the save file(h5)")
parser.add_argument("--h5FilesListFN", default="trainPelvic_segregcontour_list.txt", type=str, help="the prefix of the save file(h5)")
parser.add_argument("--adaptiveStepRatio", type=int, default=16, help="the adaptively step ratio to help decide proper step")


def main():
    global opt
    opt = parser.parse_args()

    print opt
    ids = range(0,30)

 
    for ind in ids:
        datafilename = 'preTestCha_model0110_iter14w_prob_Prostate_sub%02d.nii.gz'%ind #provide a sample name of your filename of data here
        datafn = os.path.join(opt.basePath+opt.dataFolder, datafilename)
#         labelfilename='Case%02d_segmentation.nii.gz'%ind  # provide a sample name of your filename of ground truth here
#         labelfn=os.path.join(opt.basePath+opt.dataFolder, labelfilename)
        imgOrg = sitk.ReadImage(datafn)
        mrimg = sitk.GetArrayFromImage(imgOrg)
        
#         labelOrg=sitk.ReadImage(labelfn)
#         labelimg=sitk.GetArrayFromImage(labelOrg)
        
        inds = np.where(mrimg>opt.threshold)
        
        tmat_prob = np.zeros(mrimg.shape)
        tmat_prob[inds] = mrimg[inds]
        
        tmat = np.zeros(mrimg.shape)
        tmat[inds] = 1
        tmat = denoiseImg_closing(tmat, kernel=np.ones((20,20,20))) 
        tmat = denoiseImg_isolation(tmat, struct=np.ones((3,3,3)))   
#         tmat = dice(tmat,ctnp,1)       
#         diceBladder = dice(tmat,ctnp,1)
#         print 'sub%d'%ind,'dice1 = ',diceBladder
                               
        volout = sitk.GetImageFromArray(tmat)
        sitk.WriteImage(volout, opt.basePath+opt.dataFolder+'threshold_seg_model0110_sub{:02d}'.format(ind)+'.nii.gz')  

        volout = sitk.GetImageFromArray(tmat_prob)
        sitk.WriteImage(volout, opt.basePath+opt.dataFolder+'threshold_prob_model0110_sub{:02d}'.format(ind)+'.nii.gz')  
        
if __name__ == '__main__':
#     testGradients()     
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main()