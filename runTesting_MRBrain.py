# from __future__ import print_function
import argparse, os
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torch.optim as optim
import torch
import torch.utils.data as data_utils
from utils import *
from ganComponents import *
from nnBuildUnits import CrossEntropy2d
from nnBuildUnits import computeSampleAttentionWeight
from nnBuildUnits import adjust_learning_rate
import time
from morpologicalTransformation import denoiseImg_closing, denoiseImg_isolation

# Training settings
parser = argparse.ArgumentParser(description="PyTorch InfantSeg")
parser.add_argument("--gpuID",default='5',type=str, help="the GPU ID")
parser.add_argument("--NDim", type=int, default=2, help="the dimension of the shape, 1D, 2D or 3D?")
parser.add_argument("--in_modalities", type=int, default=3, help="the number of input modalities?")
parser.add_argument("--in_channels", type=int, default=9, help="the input channels ?")
parser.add_argument("--out_channels", type=int, default=11, help="the output channels (num of classes)?")
# parser.add_argument("--in_slices", type=int, default=3, help="the num of consecutive slices for input unit?")
# parser.add_argument("--out_slices", type=int, default=1, help="the num of consecutive slices for output unit?")
parser.add_argument("--input_sz", type=arg_as_list, default=[3, 192, 192], help="the input patch size of list")
parser.add_argument("--output_sz", type=arg_as_list, default=[1, 192, 192], help="the output patch size of list")
parser.add_argument("--test_step_sz", type=arg_as_list, default=[1, 16, 16], help="the step size at testing one subject")
# parser.add_argument("--input_sz", type=arg_as_list, default=[16, 64, 64], help="the input patch size of list")
# parser.add_argument("--output_sz", type=arg_as_list, default=[16, 64, 64], help="the output patch size of list")
# parser.add_argument("--test_step_sz", type=arg_as_list, default=[2, 16, 16], help="the step size at testing one subject")
parser.add_argument("--isSegReg", action="store_true", help="is Seg and Reg?", default=False)
parser.add_argument("--isDiceLoss", action="store_true", help="is Dice Loss used?", default=True)
parser.add_argument("--isSoftmaxLoss", action="store_true", help="is Softmax Loss used?", default=False)
parser.add_argument("--isContourLoss", action="store_true", help="is Contour Loss used?", default=False)
parser.add_argument("--isDeeplySupervised", action="store_true", help="is deeply supervised mechanism used?",
                    default=True)
parser.add_argument("--isHighResolution", action="store_true", help="is high resolution used?", default=True)
parser.add_argument("--isResidualEnhancement", action="store_true", help="is residual learning operation enhanced?",
                    default=False)
parser.add_argument("--isLongConcatConnection", action="store_true", help="is the long skip connection concatenated?",
                    default=True)
parser.add_argument("--isAttentionConcat", action="store_true", help="do we use attention based concatenation?",
                    default=True)
parser.add_argument("--isViewExpansion", action="store_true", help="is view expanded?", default=True)
parser.add_argument("--isAdLoss", action="store_true", help="is adversarial loss used?", default=False)
parser.add_argument("--isSpatialDropOut", action="store_true", help="is spatial dropout used?", default=False)
parser.add_argument("--isFocalLoss", action="store_true", help="is focal loss used?", default=False)
parser.add_argument("--isSampleImportanceFromAd", action="store_true",
                    help="is sample importance from adversarial network used?", default=False)
parser.add_argument("--dropoutRate", type=float, default=0.25, help="Spatial Dropout Rate. Default=0.25")
parser.add_argument("--lambdaAD", type=float, default=0, help="loss coefficient for AD loss. Default=0")
parser.add_argument("--adImportance", type=float, default=0, help="Sample importance from AD network. Default=0")

parser.add_argument("--batchSize", type=int, default=10, help="training batch size")
parser.add_argument("--numofIters", type=int, default=200000, help="number of iterations to train for")
parser.add_argument("--showTrainLossEvery", type=int, default=100, help="number of iterations to show train loss")
parser.add_argument("--saveModelEvery", type=int, default=5000, help="number of iterations to save the model")
parser.add_argument("--showTestPerformanceEvery", type=int, default=1000,
                    help="number of iterations to show test performance")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate. Default=1e-4")
parser.add_argument("--decLREvery", type=int, default=100000, help="Sets the learning rate to the initial LR decayed by momentum every n iterations, Default: n=40000")
parser.add_argument("--cuda", action="store_true", help="Use cuda?", default=True)
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start_epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--threads", type=int, default=1, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float, help="weight decay, Default: 1e-4")
parser.add_argument("--pretrained", default="", type=str, help="path to pretrained model (default: none)")

parser.add_argument("--isTestonAttentionRegion", action="store_true", help="Test on the attention region?",
                    default=False)
# parser.add_argument("--modelPath",
#                     default="/shenlab/lab_stor5/dongnie/MRBRAIN18/modelFiles/SegMRBrain192_3D_GeneDice_WCE_lr5e3_lrdce_viewExp_DS_Dp_HR_0813_100000.pt",
#                     type=str, help="prefix of the to-be-saved model name")
# parser.add_argument("--modelPath",
#                     default="/shenlab/lab_stor5/dongnie/MRBRAIN18/modelFiles/SegMRBrain_3D_GeneDice_WCEv2_lr2e3_lrdce_viewExp_DS_Dp_HR_0813_100000.pt",
#                     type=str, help="prefix of the to-be-saved model name")
# parser.add_argument("--modelPath",
#                     default="/shenlab/lab_stor5/dongnie/MRBRAIN18/modelFiles/SegMRBrain_3D_GeneDice_WCEv3_lr5e3_lrdce_viewExp_DS_Dp_HR_SEUNet_0813_100000.pt",
#                     type=str, help="prefix of the to-be-saved model name")
# parser.add_argument("--modelPath",
#                     default="/shenlab/lab_stor5/dongnie/MRBRAIN18/modelFiles/SegMRBrain192_3D_GeneDice_WCEv2_lr5e3_lrdce_viewExp_DS_Dp_HR_0923_100000.pt",
#                     type=str, help="prefix of the to-be-saved model name")
parser.add_argument("--modelPath",
                    default="/shenlab/lab_stor5/dongnie/MRBRAIN18/modelFiles/SegMRBrain192_3D_GeneDice_WCEv2_lr5e3_lrdce_viewExp_DS_Dp_HR_SEUNet_0923_100000.pt",
                    type=str, help="prefix of the to-be-saved model name")
# parser.add_argument("--modelPath", default="/home/dongnie/Desktop/myPyTorch/pytorch-SRResNet23D/SegCha_wce_wdice_viewExp_resEnhance_lrdcr_1216_100000.pt", type=str, help="prefix of the to-be-saved model name")
parser.add_argument("--prefixPredictedFN", default="MR18_model10w_3D_GeneDice_WCEv2_lr2e3_lrdce_viewExp_DS_Dp_HR_SEUNET_0923_", type=str,
                    help="prefix of the to-be-saved predicted filename")
parser.add_argument("--how2normalize", type=int, default=6, help="how to normalize the data")
parser.add_argument("--resType", type=int, default=0,
                    help="resType: 0: segmentation map (integer); 1: regression map (continuous); 2: segmentation map + probability map")

global opt
opt = parser.parse_args()

def main():
    print opt

    path_test = '/home/dongnie/warehouse/pelvicSeg/prostateChallenge/data/'
    path_test = '/shenlab/lab_stor5/dongnie/challengeData/data/'
    path_test = '/shenlab/lab_stor5/dongnie/MRBRAIN18/training/'

    if opt.isSegReg:
        netG = ResSegRegNet(opt.in_channels, opt.out_channels, nd=opt.NDim)
    elif opt.isContourLoss:
        netG = ResSegContourNet(opt.in_channels, opt.out_channels, nd=opt.NDim,
                                isRandomConnection=opt.isResidualEnhancement, isSmallDilation=opt.isViewExpansion,
                                isSpatialDropOut=opt.isSpatialDropOut, dropoutRate=opt.dropoutRate)
    elif opt.isLongConcatConnection and opt.isDeeplySupervised and opt.isHighResolution:
        netG = HRResUNet_DS(opt.in_channels, opt.out_channels, isRandomConnection=opt.isResidualEnhancement,
                            isSmallDilation=opt.isViewExpansion, isSpatialDropOut=opt.isSpatialDropOut,
                            dropoutRate=opt.dropoutRate, TModule='HR', FModule=opt.isAttentionConcat, nd=opt.NDim)
    elif opt.isDeeplySupervised and opt.isHighResolution:
        netG = HRResSegNet_DS(opt.in_channels, opt.out_channels, nd=opt.NDim,
                              isRandomConnection=opt.isResidualEnhancement, isSmallDilation=opt.isViewExpansion,
                              isSpatialDropOut=opt.isSpatialDropOut, dropoutRate=opt.dropoutRate)
    elif opt.isLongConcatConnection and opt.isHighResolution:
        netG = UNet(opt.in_channels, opt.out_channels, TModule='HR', FModule=opt.isAttentionConcat, nd=opt.NDim)
    elif opt.isDeeplySupervised:
        netG = ResSegNet_DS(opt.in_channels, opt.out_channels, nd=opt.NDim,
                            isRandomConnection=opt.isResidualEnhancement, isSmallDilation=opt.isViewExpansion,
                            isSpatialDropOut=opt.isSpatialDropOut, dropoutRate=opt.dropoutRate)
    elif opt.isHighResolution:
        netG = HRResSegNet(opt.in_channels, opt.out_channels, nd=opt.NDim, isRandomConnection=opt.isResidualEnhancement,
                           isSmallDilation=opt.isViewExpansion, isSpatialDropOut=opt.isSpatialDropOut,
                           dropoutRate=opt.dropoutRate)
    elif opt.isLongConcatConnection:
        netG = UNet(opt.in_channels, opt.out_channels, TModule=None, FModule=opt.isAttentionConcat, nd=opt.NDim)
    else:
        netG = ResSegNet(opt.in_channels, opt.out_channels, nd=opt.NDim, isRandomConnection=opt.isResidualEnhancement,
                         isSmallDilation=opt.isViewExpansion, isSpatialDropOut=opt.isSpatialDropOut,
                         dropoutRate=opt.dropoutRate)

    # netG.apply(weights_init)
    netG = netG.cuda()

    checkpoint = torch.load(opt.modelPath)
    netG.load_state_dict(checkpoint["model"])
    #     netG.load_state_dict(torch.load(opt.modelPath))

    ids = [1, 2, 3, 4, 6, 7, 8, 10, 11, 12, 13]
    ids = [45, 46, 47, 48, 49]
    scan = ScanFile(path_test)
    dirs = scan.scan_immediate_subdir()

    for currDir in dirs:
        print 'now we visit: ', currDir

        if opt.in_modalities == 3:
            t1_test_itk = sitk.ReadImage(os.path.join(path_test, currDir, 'pre', 'reg_T1.nii.gz'))
            ir_test_itk = sitk.ReadImage(os.path.join(path_test, currDir, 'pre', 'reg_IR.nii.gz'))
            flair_test_itk = sitk.ReadImage(os.path.join(path_test, currDir, 'pre', 'FLAIR.nii.gz'))

            t1np = sitk.GetArrayFromImage(t1_test_itk)
            flairnp = sitk.GetArrayFromImage(flair_test_itk)
            irnp = sitk.GetArrayFromImage(ir_test_itk)
        else:
            t1_test_itk = sitk.ReadImage(os.path.join(path_test, opt.test_input_file_name))
            t1np = sitk.GetArrayFromImage(t1_test_itk)

        ct_test_itk = sitk.ReadImage(os.path.join(path_test, currDir, 'segm.nii.gz'))
        spacing = ct_test_itk.GetSpacing()
        origin = ct_test_itk.GetOrigin()
        direction = ct_test_itk.GetDirection()
        ctnp = sitk.GetArrayFromImage(ct_test_itk)

        mrnp = t1np
        mu = np.mean(mrnp)

        # for training data in pelvicSeg
        if opt.how2normalize == 1:
            maxV, minV = np.percentile(mrnp, [99, 1])
            print 'maxV,', maxV, ' minV, ', minV
            mrnp = (mrnp - mu) / (maxV - minV)
            print 'unique value: ', np.unique(ctnp)

        # for training data in pelvicSeg
        elif opt.how2normalize == 2:
            maxV, minV = np.percentile(mrnp, [99, 1])
            print 'maxV,', maxV, ' minV, ', minV
            mrnp = (mrnp - mu) / (maxV - minV)
            print 'unique value: ', np.unique(ctnp)

        # for training data in pelvicSegRegH5
        elif opt.how2normalize == 3:
            std = np.std(mrnp)
            mrnp = (mrnp - mu) / std
            print 'maxV,', np.ndarray.max(mrnp), ' minV, ', np.ndarray.min(mrnp)

        elif opt.how2normalize == 4:
            maxV, minV = np.percentile(mrnp, [99.2, 1])
            print 'maxV is: ', np.ndarray.max(mrnp)
            mrnp[np.where(mrnp > maxV)] = maxV
            print 'maxV is: ', np.ndarray.max(mrnp)
            mu = np.mean(mrnp)
            std = np.std(mrnp)
            mrnp = (mrnp - mu) / std
            print 'maxV,', np.ndarray.max(mrnp), ' minV, ', np.ndarray.min(mrnp)

        if opt.how2normalize == 6:
            maxPercentT1, minPercentT1 = np.percentile(t1np, [99.5, 0])
            maxPercentFlair, minPercentFlair = np.percentile(flairnp, [99.5, 0])
            maxPercentIR, minPercentIR = np.percentile(irnp, [99.5, 0])

            print 'maxPercentT1: ', maxPercentT1, ' minPercentT1: ', minPercentT1, ' maxPercentFlair: ', maxPercentFlair, 'minPercentFlair: ', minPercentFlair, ' maxPercentIR: ', maxPercentIR, ' minPercentIR: ', minPercentIR

            matT1 = (t1np - minPercentT1) / (maxPercentT1 - minPercentT1)
            matFlair = (flairnp - minPercentFlair) / (maxPercentFlair - minPercentFlair)
            matIR = (irnp - minPercentIR) / (maxPercentIR - minPercentIR)

            print 'maxT1: ', np.amax(matT1), ' maxFlair: ', np.amax(matFlair), ' maxIR: ', np.amax(matIR)
            print 'minT1: ', np.amin(matT1), ' minFlair: ', np.amin(matFlair), ' minIR: ', np.amin(matIR)

        #full image version with average over the overlapping regions
        mat1 = np.expand_dims(matT1, axis=0)
        mat2 = np.expand_dims(matFlair, axis=0)
        mat3 = np.expand_dims(matIR, axis=0)
        print 'mat1: ',mat1.shape,' mat2: ',mat2.shape,' mat3:',mat3.shape

        matFA = np.concatenate((mat1, mat2, mat3), axis=0)
        print 'matFA.shape: ', matFA.shape
        matGT = ctnp
        print 'matGT.shape: ', matGT.shape
        # the attention regions
        row, col, leng = mrnp.shape
        if opt.isTestonAttentionRegion:
            # the attention regions
            y1 = int(leng * 0.25)
            y2 = int(leng * 0.75)
            x1 = int(col * 0.25)
            x2 = int(col * 0.75)
            matFA = matFA[:, y1:y2, x1:x2]  # note, matFA and matFAOut same size
            matGT = ctnp[:, y1:y2, x1:x2]
        else:
            matFA = matFA
            matGT = ctnp

        print 'matFA.shape: ', matFA.shape
        # matOut, _ = testOneSubject(matFA, matGT, opt.out_channels, opt.input_sz, opt.output_sz, opt.test_step_sz, netG,
        #                            opt.prefixModelName + '%d.pt' % iter, nd=opt.NDim)

        if opt.resType == 2:
            matOut, matProb, _ = testOneSubject(matFA, matGT, opt.out_channels, opt.input_sz, opt.output_sz,
                                                opt.test_step_sz, netG, opt.modelPath, resType=opt.resType, nd=opt.NDim)
        else:
            matOut, _ = testOneSubject(matFA, matGT, opt.out_channels, opt.input_sz, opt.output_sz, opt.test_step_sz,
                                       netG, opt.modelPath, resType=opt.resType, nd=opt.NDim)

        # matOut,_ = testOneSubject(matFA,matGT,opt.out_channels,opt.input_sz, opt.output_sz, opt.test_step_sz,netG,opt.modelPath, nd = opt.NDim)
        ct_estimated = np.zeros([mrnp.shape[0], mrnp.shape[1], mrnp.shape[2]])
        ct_prob = np.zeros([opt.out_channels, mrnp.shape[0], mrnp.shape[1], mrnp.shape[2]])

        if opt.isTestonAttentionRegion:
            ct_estimated[:, y1:y2, x1:x2] = matOut
            # ct_prob[:, :, y1:y2, x1:x2] = matProb
        else:
            ct_estimated = matOut
            # ct_prob = matProb

        ct_estimated = np.rint(ct_estimated)

        print 'pred: ', ct_estimated.dtype, ' shape: ', ct_estimated.shape
        print 'gt: ', ctnp.dtype, ' shape: ', ct_estimated.shape

        # ct_estimated = denoiseImg_closing(ct_estimated, kernel=np.ones((20, 20, 20)))
        # ct_estimated = denoiseImg_isolation(ct_estimated, struct=np.ones((3, 3, 3)))
        dice1 = dice(ct_estimated, ctnp, 1)
        dice2 = dice(ct_estimated, ctnp, 2)
        dice3 = dice(ct_estimated, ctnp, 3)
        dice4 = dice(ct_estimated, ctnp, 4)
        dice5 = dice(ct_estimated, ctnp, 5)
        dice6 = dice(ct_estimated, ctnp, 6)
        dice7 = dice(ct_estimated, ctnp, 7)
        dice8 = dice(ct_estimated, ctnp, 8)
        dice9 = dice(ct_estimated, ctnp, 9)
        dice10 = dice(ct_estimated, ctnp, 10)



        # print 'dice1 = ',diceBladder,' dice2= ',diceProstate,' dice3= ',diceRectumm
        print 'dice1=', dice1, ' dice2=',dice2,' dice3=',dice3,' dice4=',dice4,' dice5=',dice5,' dice6=',dice6,' dice7=',dice7,' dice8=',dice8,' dice9=',dice9,' dice10=',dice10

        volout = sitk.GetImageFromArray(ct_estimated)
        volout.SetSpacing(spacing)
        volout.SetOrigin(origin)
        volout.SetDirection(direction)
        sitk.WriteImage(volout, opt.prefixPredictedFN + '{}'.format(currDir) + '.nii.gz')
        #             netG.save_state_dict('Segmentor_model_%d.pt'%iter)
        #             netD.save_state_dic('Discriminator_model_%d.pt'%iter)
        dsc = dice1


if __name__ == '__main__':
    #     testGradients()
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpuID
    main()
