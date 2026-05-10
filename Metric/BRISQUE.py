# Example run: python -W ignore BRISQUE.py --test_dir /root/autodl-tmp/Result/RetinexNet/ExDark --read_subfolder True

import os
import PIL
from glob import glob
from PIL import Image
import numpy as np
from skimage import io, img_as_float
import imquality.brisque as brisque
import argparse
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument('--test_dir', type=str, default="F:\\test_lime",help='directory for testing inputs')
    parser.add_argument('--test_dir', type=str, default="D:\\File\\lowlight\\test\\test2",help='directory for testing inputs')
    # parser.add_argument('--test_dir', type=str,
    #                     default="E:\\lowlight_dataset\\NIQE\\new2_11_vgg_relu4_3_SELU_fea2_nopatch_mirror_n2_psnr_ssim_l1\\VV",#"E:\\lowlight_dataset\\NIQE\\VV\\testA",#
    #                     help='directory for testing inputs')
    # parser.add_argument('--test_dir', type=str,
    #                     default="E:\\code_beifen\\SEEGANv2\\ablation\\lr_net1_11_gammavgg_relu4_3_SELU_2DA_2fea_ls_1g1d_mirror_20251031\\VV\\images",#"E:\\lowlight_dataset\\NIQE\\VV\\testA",#
    #                     help='directory for testing inputs')
    # parser.add_argument('--test_dir', type=str,
    #                     default='E:\\code_beifen\\SEEGANv2\\ablation\\test_lime')
    # parser.add_argument('--test_dir', type=str, default='E:\\results\\other\\AGLLDiff\\DICM')
    # parser.add_argument('--test_dir', default='E:\\code_beifen\\SEEGANv2\\ablation\\lr_net1_11_gammavgg_relu4_3_SELU_2DA_2fea_ls_1g1d_mirror2\\LOL_blur\\images', type=str)  # HVI LEDNet FourierDiff




    #                     help='directory for testing inputs')
    parser.add_argument('--read_subfolder', type=bool, default=False)
    args = parser.parse_args()
    return args
    # parser.add_argument('-dirA', default='E:\\lowlight_dataset\\VE-LOL-L\\VE-LOL-L-Cap-Full\\VE-LOL-L-Cap-Normal_test', type=str)
    # parser.add_argument('-dirA', default='E:\\lowlight_dataset\\LOLdataset\\eval15\\high', type=str)
    # engan RetinexNet MBLLEN kind zero_dce zero_dce++ sci ruas LLFlow SGZ SNR_Net retinexformer URetinex zeroIG PIE PairLIE
    # parser.add_argument('-dirB', default='E:\\results\\engan\\ve_lol', type=str)
def cal_loe(inp):
    pass


def cal_brisque(inp, i, AssertionError_count):
    # imgOri = Image.open(inp)
    # img = img_as_float(imgOri)
    img = PIL.Image.open(inp)
    try:
        score = brisque.score(img)
    except AssertionError:
        score = 0
        AssertionError_count += 1
        print(i, " path[i] is", inp)
    return score, AssertionError_count


def main(args):
    img_dir = args.test_dir
    # for dataset_name in ['NPE','MEF','DICM','VV']:
    #     img_dir = 'D:\\File\\lowlight\\python\\hap_run\\ablation\\lr_net1_11_gammavgg_relu4_3_SELU_2DA_2fea_ls_1g1d_mirror\\'
    #     img_dir = img_dir + "\\" + dataset_name + "\\" + "images"


    if args.read_subfolder:
        path = sorted(glob(os.path.join(img_dir, '*/*')))
    else:
        path = sorted(glob(os.path.join(img_dir, '*')))

    list_brisque = []

    AssertionError_count = 0
    for i in range(len(path)):
    # for i in tqdm(range(1550, len(path))):

        # calculate scores
        # print(i, " path[i] is", path[i])
        score, AssertionError_count = cal_brisque(path[i], i, AssertionError_count)
        if score != 0:
        # append to list
        #     print( " path[i] is", path[i])
        #     print( " score is", score)
            list_brisque.append(score)

    # Average score for the dataset
    print("======={}=======>".format(img_dir))
    print("Have ", AssertionError_count, " times AssertionError.")
    print("Average BRISQUE:", "%.3f" % (np.mean(list_brisque)))



if __name__ == "__main__":
    args = parse_args()
    main(args)
