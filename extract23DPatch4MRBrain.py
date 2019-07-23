    
'''
Target: Crop patches for kinds of medical images, such as hdr, nii, mha, mhd, raw and so on, and store them as hdf5 files
for single-scale patches
Created in June, 2016
Author: Dong Nie 
'''



import SimpleITK as sitk

from multiprocessing import Pool
import os, argparse
import h5py
import numpy as np

parser = argparse.ArgumentParser(description="PyTorch InfantSeg")
parser.add_argument("--how2normalize", type=int, default=6, help="how to normalize the data")

global opt
opt = parser.parse_args()


d1=16
d2=64
d3=64
dFA=[d1,d2,d3] # size of patches of input data
dSeg=[16,64,64] # size of pathes of label data
step1=8
step2=16
step3=16
step=[step1,step2,step3]
    

class ScanFile(object):   
    def __init__(self,directory,prefix=None,postfix=None):  
        self.directory=directory  
        self.prefix=prefix  
        self.postfix=postfix  
        
    ## go deep through the directory/file tree        
    def scan_files(self):    
        files_list=[]    
            
        for dirpath,dirnames,filenames in os.walk(self.directory):   
            ''''' 
            dirpath is a string, the path to the directory.   
            dirnames is a list of the names of the subdirectories in dirpath (excluding '.' and '..'). 
            filenames is a list of the names of the non-directory files in dirpath. 
            '''  
            for special_file in filenames:    
                if self.postfix:  
                    if  special_file.endswith(self.postfix):    
                        files_list.append(os.path.join(dirpath,special_file))    
                elif self.prefix:    
                    if special_file.startswith(self.prefix):  
                        files_list.append(os.path.join(dirpath,special_file))    
                else:    
                    files_list.append(os.path.join(dirpath,special_file))    
                                  
        return files_list    
    
    ## go deep through the directory/file tree  
    def scan_subdir(self):  
        subdir_list=[]  
        for dirpath,dirnames,files in os.walk(self.directory):  
            subdir_list.append(dirpath)  
        return subdir_list      
    
    ## just visit the current directory instead of go deeply over the whole file tree
    def scan_immediate_subdir(self):
        subdir_list=[]  
        
        for name in os.listdir(self.directory):
            if os.path.isdir(os.path.join(self.directory, name)):
                subdir_list.append(name)
                
        return subdir_list
    

    
'''
Actually, we donot need it any more,  this is useful to generate hdf5 database
'''
def extractPatch4OneSubject(matFA, matMR, matSeg, matMask, fileID ,d, step, rate):
  
    eps=5e-2
    rate1=1.0/2
    rate2=1.0/4
    [row,col,leng]=matFA.shape
    cubicCnt=0
    estNum=40000
    trainFA=np.zeros([estNum,1, dFA[0],dFA[1],dFA[2]],dtype=np.float16)
    trainMR=np.zeros([estNum,1,dFA[0],dFA[1],dFA[2]],dtype=np.float16)
    trainSeg=np.zeros([estNum,1,dSeg[0],dSeg[1],dSeg[2]],dtype=np.float16)
    trainMask=np.zeros([estNum,1,dSeg[0],dSeg[1],dSeg[2]],dtype=np.int8)


    print 'trainFA shape, ',trainFA.shape
    #to padding for input
    margin1=(dFA[0]-dSeg[0])/2
    margin2=(dFA[1]-dSeg[1])/2
    margin3=(dFA[2]-dSeg[2])/2
    cubicCnt=0
    marginD=[margin1,margin2,margin3]
    print 'matFA shape is ',matFA.shape
    matFAOut=np.zeros([row+2*marginD[0],col+2*marginD[1],leng+2*marginD[2]],dtype=np.float16)
    print 'matFAOut shape is ',matFAOut.shape
    matFAOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]]=matFA
    
    matMROut=np.zeros([row+2*marginD[0],col+2*marginD[1],leng+2*marginD[2]],dtype=np.float16)
    print 'matMROut shape is ',matMROut.shape
    matMROut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]]=matMR
    
    matSegOut=np.zeros([row+2*marginD[0],col+2*marginD[1],leng+2*marginD[2]],dtype=np.float16)
    matSegOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]]=matSeg
 
 
    matMaskOut=np.zeros([row+2*marginD[0],col+2*marginD[1],leng+2*marginD[2]],dtype=np.float16)
    matMaskOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]]=matMask
       
    #for mageFA, enlarge it by padding
    if margin1!=0:
        matFAOut[0:marginD[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]]=matFA[marginD[0]-1::-1,:,:] #reverse 0:marginD[0]
        matFAOut[row+marginD[0]:matFAOut.shape[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]]=matFA[matFA.shape[0]-1:row-marginD[0]-1:-1,:,:] #we'd better flip it along the 1st dimension
    if margin2!=0:
        matFAOut[marginD[0]:row+marginD[0],0:marginD[1],marginD[2]:leng+marginD[2]]=matFA[:,marginD[1]-1::-1,:] #we'd flip it along the 2nd dimension
        matFAOut[marginD[0]:row+marginD[0],col+marginD[1]:matFAOut.shape[1],marginD[2]:leng+marginD[2]]=matFA[:,matFA.shape[1]-1:col-marginD[1]-1:-1,:] #we'd flip it along the 2nd dimension
    if margin3!=0:
        matFAOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],0:marginD[2]]=matFA[:,:,marginD[2]-1::-1] #we'd better flip it along the 3rd dimension
        matFAOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],marginD[2]+leng:matFAOut.shape[2]]=matFA[:,:,matFA.shape[2]-1:leng-marginD[2]-1:-1]
        
      #for matMR, enlarge it by padding
    if margin1!=0:
        matMROut[0:marginD[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]]=matMR[marginD[0]-1::-1,:,:] #reverse 0:marginD[0]
        matMROut[row+marginD[0]:matMROut.shape[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]]=matMR[matMR.shape[0]-1:row-marginD[0]-1:-1,:,:] #we'd better flip it along the 1st dimension
    if margin2!=0:
        matMROut[marginD[0]:row+marginD[0],0:marginD[1],marginD[2]:leng+marginD[2]]=matMR[:,marginD[1]-1::-1,:] #we'd flip it along the 2nd dimension
        matMROut[marginD[0]:row+marginD[0],col+marginD[1]:matMROut.shape[1],marginD[2]:leng+marginD[2]]=matMR[:,matMR.shape[1]-1:col-marginD[1]-1:-1,:] #we'd flip it along the 2nd dimension
    if margin3!=0:
        matMROut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],0:marginD[2]]=matMR[:,:,marginD[2]-1::-1] #we'd better flip it along the 3rd dimension
        matMROut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],marginD[2]+leng:matMROut.shape[2]]=matMR[:,:,matMR.shape[2]-1:leng-marginD[2]-1:-1]    
    
    #for matseg, enlarge it by padding
    if margin1!=0:
        matSegOut[0:marginD[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]]=matSeg[marginD[0]-1::-1,:,:] #reverse 0:marginD[0]
        matSegOut[row+marginD[0]:matSegOut.shape[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]]=matSeg[matSeg.shape[0]-1:row-marginD[0]-1:-1,:,:] #we'd better flip it along the 1st dimension
    if margin2!=0:
        matSegOut[marginD[0]:row+marginD[0],0:marginD[1],marginD[2]:leng+marginD[2]]=matSeg[:,marginD[1]-1::-1,:] #we'd flip it along the 2nd dimension
        matSegOut[marginD[0]:row+marginD[0],col+marginD[1]:matSegOut.shape[1],marginD[2]:leng+marginD[2]]=matSeg[:,matSeg.shape[1]-1:col-marginD[1]-1:-1,:] #we'd flip it along the 2nd dimension
    if margin3!=0:
        matSegOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],0:marginD[2]]=matSeg[:,:,marginD[2]-1::-1] #we'd better flip it along the 3rd dimension
        matSegOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],marginD[2]+leng:matSegOut.shape[2]]=matSeg[:,:,matSeg.shape[2]-1:leng-marginD[2]-1:-1]
  
      #for matseg, enlarge it by padding
    if margin1!=0:
        matMaskOut[0:marginD[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]]=matMask[marginD[0]-1::-1,:,:] #reverse 0:marginD[0]
        matMaskOut[row+marginD[0]:matMaskOut.shape[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]]=matMask[matMask.shape[0]-1:row-marginD[0]-1:-1,:,:] #we'd better flip it along the 1st dimension
    if margin2!=0:
        matMaskOut[marginD[0]:row+marginD[0],0:marginD[1],marginD[2]:leng+marginD[2]]=matMask[:,marginD[1]-1::-1,:] #we'd flip it along the 2nd dimension
        matMaskOut[marginD[0]:row+marginD[0],col+marginD[1]:matMaskOut.shape[1],marginD[2]:leng+marginD[2]]=matMask[:,matMask.shape[1]-1:col-marginD[1]-1:-1,:] #we'd flip it along the 2nd dimension
    if margin3!=0:
        matMaskOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],0:marginD[2]]=matMask[:,:,marginD[2]-1::-1] #we'd better flip it along the 3rd dimension
        matMaskOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],marginD[2]+leng:matMaskOut.shape[2]]=matMask[:,:,matMask.shape[2]-1:leng-marginD[2]-1:-1]
              
    dsfactor = rate
    
    for i in range(0,row-dSeg[0],step[0]):
        for j in range(0,col-dSeg[1],step[1]):
            for k in range(0,leng-dSeg[2],step[2]):
                volMask = matMaskOut[i:i+dSeg[0],j:j+dSeg[1],k:k+dSeg[2]]
                if np.sum(volMask)<eps:
                    continue
                cubicCnt = cubicCnt+1
                #index at scale 1
                volMask = matMask[i:i+dSeg[0],j:j+dSeg[1],k:k+dSeg[2]]

                volSeg = matSeg[i:i+dFA[0],j:j+dFA[1],k:k+dFA[2]]
                volFA = matFAOut[i:i+dFA[0],j:j+dFA[1],k:k+dFA[2]]
                
                volMR = matMROut[i:i+dFA[0],j:j+dFA[1],k:k+dFA[2]]

                trainFA[cubicCnt,0,:,:,:] = volFA #32*32*32
                trainMR[cubicCnt,0,:,:,:] = volMR #32*32*32
                trainSeg[cubicCnt,0,:,:,:] = volSeg#32x32x32
                
                trainMask[cubicCnt,0,:,:,:] = volMask#32x32x32


    trainFA = trainFA[0:cubicCnt,:,:,:,:]
    trainMR = trainMR[0:cubicCnt,:,:,:,:]  
    trainSeg = trainSeg[0:cubicCnt,:,:,:,:]
    trainMask = trainMask[0:cubicCnt,:,:,:,:]


    with h5py.File('./trainPETCT_snorm_64_%s.h5'%fileID,'w') as f:
        f['dataT1'] = trainFA
        f['dataFlair'] = trainMR        
        f['dataIR'] = trainSeg
        f['dataSeg'] = trainSeg
     
    with open('./trainMR18_snorm_64_list.txt','a') as f:
        f.write('./trainMR18_snorm_64_%s.h5\n'%fileID)
    return cubicCnt
        
def main():
    print opt
    path = '/shenlab/lab_stor5/dongnie/MRBRAIN18/training/'
    scan = ScanFile(path)  
    dirs = scan.scan_immediate_subdir()  

    
    for currDir in dirs:
         
        print 'now we visit: ', currDir
        
        fn_T1 = os.path.join(path,currDir,'pre','reg_T1.nii.gz')
        fn_IR = os.path.join(path,currDir,'pre','reg_IR.nii.gz')
        fn_flair = os.path.join(path,currDir,'pre','FLAIR.nii.gz')
        fn_seg = os.path.join(path,currDir, 'segm.nii.gz')
        
        imgOrg = sitk.ReadImage(fn_T1)
        t1np = sitk.GetArrayFromImage(imgOrg)

        imgOrg1 = sitk.ReadImage(fn_IR)
        irnp = sitk.GetArrayFromImage(imgOrg1)
        
        imgOrg2 = sitk.ReadImage(fn_flair)
        flairnp = sitk.GetArrayFromImage(imgOrg2)

        labelOrg = sitk.ReadImage(fn_seg)
        segnp = sitk.GetArrayFromImage(labelOrg)

        mrnp = t1np
        ctnp = irnp
        if opt.how2normalize == 1:
            maxV, minV = np.percentile(mrnp, [99, 1])
            print 'maxV,', maxV, ' minV, ', minV
            mrnp = (mrnp - mu) / (maxV - minV)
            print 'unique value: ', np.unique(ctnp)

        # for training data in pelvicSeg
        if opt.how2normalize == 2:
            maxV, minV = np.percentile(mrnp, [99, 1])
            print 'maxV,', maxV, ' minV, ', minV
            mrnp = (mrnp - mu) / (maxV - minV)
            print 'unique value: ', np.unique(ctnp)

        # for training data in pelvicSegRegH5
        if opt.how2normalize == 3:
            std = np.std(mrnp)
            mrnp = (mrnp - mu) / std
            print 'maxV,', np.ndarray.max(mrnp), ' minV, ', np.ndarray.min(mrnp)

        if opt.how2normalize == 4:
            maxLPET = 149.366742
            maxPercentLPET = 7.76
            minLPET = 0.00055037
            meanLPET = 0.27593288
            stdLPET = 0.75747500

            # for rsCT
            maxCT = 27279
            maxPercentCT = 1320
            minCT = -1023
            meanCT = -601.1929
            stdCT = 475.034

            # for s-pet
            maxSPET = 156.675962
            maxPercentSPET = 7.79
            minSPET = 0.00055037
            meanSPET = 0.284224789
            stdSPET = 0.7642257

            # matLPET = (mrnp - meanLPET) / (stdLPET)
            matLPET = (mrnp - minLPET) / (maxPercentLPET - minLPET)
            matCT = (ctnp - meanCT) / stdCT
            matSPET = (flairnp - minSPET) / (maxPercentSPET - minSPET)

        if opt.how2normalize == 5:
            # for rsCT
            maxCT = 27279
            maxPercentCT = 1320
            minCT = -1023
            meanCT = -601.1929
            stdCT = 475.034

            print 'ct, max: ', np.amax(ctnp), ' ct, min: ', np.amin(ctnp)

            # matLPET = (mrnp - meanLPET) / (stdLPET)
            matLPET = mrnp
            matCT = (ctnp - meanCT) / stdCT
            matSPET = flairnp

        if opt.how2normalize == 6:
            maxPercentT1, minPercentT1 = np.percentile(t1np, [99.5, 0])
            maxPercentFlair, minPercentFlair = np.percentile(flairnp, [99.5, 0])
            maxPercentIR, minPercentIR = np.percentile(irnp, [99.5, 0])

            print 'maxPercentT1: ',maxPercentT1, ' minPercentT1: ',minPercentT1, ' maxPercentFlair: ',maxPercentFlair, 'minPercentFlair: ', minPercentFlair,' maxPercentIR: ',maxPercentIR, ' minPercentIR: ',minPercentIR

            matT1 = (t1np - minPercentT1)/(maxPercentT1 - minPercentT1)
            matFlair = (flairnp - minPercentFlair) / (maxPercentFlair - minPercentFlair)
            matIR = (irnp - minPercentIR) / (maxPercentIR - minPercentIR)

            print 'maxT1: ',np.amax(matT1), ' maxFlair: ', np.amax(matFlair), ' maxIR: ', np.amax(matIR)
            print 'minT1: ', np.amin(matT1), ' minFlair: ', np.amin(matFlair), ' minIR: ', np.amin(matIR)

    
        matSeg = segnp
        
        fileID = currDir 
        rate = 1
        cubicCnt = extractPatch4OneSubject(matT1, matFlair, matIR, matSeg, fileID,dSeg,step,rate)
        #cubicCnt = extractPatch4OneSubject(mrnp, matCT, hpetnp, maskimg, fileID,dSeg,step,rate)
        print '# of patches is ', cubicCnt

        # reverse along the 1st dimension
        rmrimg = matT1[matT1.shape[0] - 1::-1, :, :]
        rmatFlair = matFlair[matFlair.shape[0] - 1::-1, :, :]
        rmatIR = matIR[matIR.shape[0] - 1::-1, :, :]
        rmatSeg = matSeg[matSeg.shape[0] - 1::-1, :, :]
        fileID = fileID+'r'
        cubicCnt = extractPatch4OneSubject(rmrimg, rmatFlair, rmatIR, rmatSeg, fileID,dSeg,step,rate)
        print '# of patches is ', cubicCnt
    
if __name__ == '__main__':     
    main()
