    
'''
Target: Crop patches for kinds of medical images, such as hdr, nii, mha, mhd, raw and so on, and store them as hdf5 files
for single-scale patches
We also extract patches for each organ separately which aims at jointly training
We crop segmentation maps, regression maps, and the contours.
Created on Oct. 20, 2016

We use adaptively steps to extract patches

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
    
    
# Training settings
parser = argparse.ArgumentParser(description="PyTorch InfantSeg")
parser.add_argument("--how2normalize", type=int, default=4, help="how to normalize the data")
parser.add_argument("--basePath", default="/shenlab/lab_stor5/dongnie/challengeData/", type=str, help="base path for this project's data")
parser.add_argument("--dataFolder", default="testdata/", type=str, help="the name of the data source folder")
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
        datafilename='Case%02d.nii.gz'%ind #provide a sample name of your filename of data here
        datafn=os.path.join(opt.basePath+opt.dataFolder, datafilename)
        labelfilename='Case%02d_segmentation.nii.gz'%ind  # provide a sample name of your filename of ground truth here
        labelfn=os.path.join(opt.basePath+opt.dataFolder, labelfilename)
        imgOrg=sitk.ReadImage(datafn)
        mrimg=sitk.GetArrayFromImage(imgOrg)
        
        #labelOrg=sitk.ReadImage(labelfn)
        #labelimg=sitk.GetArrayFromImage(labelOrg)
        labelimg=0
        
        
        if opt.how2normalize == 1:
            mu=np.mean(mrimg)
            maxV, minV=np.percentile(mrimg, [99 ,1])
            print 'maxV,',maxV,' minV, ',minV
            mrimg=(mrimg-mu)/(maxV-minV)
            print 'unique value: ',np.unique(labelimg)
        
        #for training data in pelvicSeg
        elif opt.how2normalize == 2:
            mu=np.mean(mrimg)
            maxV, minV = np.percentile(mrimg, [99 ,1])
            print 'maxV,',maxV,' minV, ',minV
            mrimg = (mrimg-mu)/(maxV-minV)
            print 'unique value: ',np.unique(labelimg)
        
        #for training data in pelvicSegRegH5
        elif opt.how2normalize== 3:
            mu=np.mean(mrimg)
            std = np.std(mrimg)
            mrimg = (mrimg - mu)/std
            print 'maxV,',np.ndarray.max(mrimg),' minV, ',np.ndarray.min(mrimg)
            
                #for training data in pelvicSegRegH5
        elif opt.how2normalize== 4:
            maxV, minV = np.percentile(mrimg, [99.2 ,1])
            print 'maxV is: ',np.ndarray.max(mrimg)
            mrimg[np.where(mrimg>maxV)] = maxV
            print 'maxV is: ',np.ndarray.max(mrimg)
            mu=np.mean(mrimg)
            std = np.std(mrimg)
            mrimg = (mrimg - mu)/std
            print 'maxV,',np.ndarray.max(mrimg),' minV, ',np.ndarray.min(mrimg)
            
        volout = sitk.GetImageFromArray(mrimg)
        norm_filename = 'Case%02d_normalized.nii.gz'%ind
        sitk.WriteImage(volout,opt.basePath+opt.dataFolder+norm_filename)
        
if __name__ == '__main__':
#     testGradients()     
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main()
