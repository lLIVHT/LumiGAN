import os
from glob import glob
import cv2
import math
import numpy as np
from scipy.ndimage.filters import convolve
from scipy.special import gamma
import argparse
from metric_util import reorder_image, to_y_channel
import pyiqa



def estimate_aggd_param(block):
    """Estimate AGGD (Asymmetric Generalized Gaussian Distribution) paramters.

    Args:
        block (ndarray): 2D Image block.

    Returns:
        tuple: alpha (float), beta_l (float) and beta_r (float) for the AGGD
            distribution (Estimating the parames in Equation 7 in the paper).
    """
    block = block.flatten()
    gam = np.arange(0.2, 10.001, 0.001)  # len = 9801
    gam_reciprocal = np.reciprocal(gam)
    r_gam = np.square(gamma(gam_reciprocal * 2)) / (
        gamma(gam_reciprocal) * gamma(gam_reciprocal * 3))

    left_std = np.sqrt(np.mean(block[block < 0]**2))
    right_std = np.sqrt(np.mean(block[block > 0]**2))
    gammahat = left_std / right_std
    rhat = (np.mean(np.abs(block)))**2 / np.mean(block**2)
    rhatnorm = (rhat * (gammahat**3 + 1) *
                (gammahat + 1)) / ((gammahat**2 + 1)**2)
    array_position = np.argmin((r_gam - rhatnorm)**2)

    alpha = gam[array_position]
    beta_l = left_std * np.sqrt(gamma(1 / alpha) / gamma(3 / alpha))
    beta_r = right_std * np.sqrt(gamma(1 / alpha) / gamma(3 / alpha))
    return (alpha, beta_l, beta_r)


def compute_feature(block):
    """Compute features.

    Args:
        block (ndarray): 2D Image block.

    Returns:
        list: Features with length of 18.
    """
    feat = []
    alpha, beta_l, beta_r = estimate_aggd_param(block)
    feat.extend([alpha, (beta_l + beta_r) / 2])

    # distortions disturb the fairly regular structure of natural images.
    # This deviation can be captured by analyzing the sample distribution of
    # the products of pairs of adjacent coefficients computed along
    # horizontal, vertical and diagonal orientations.
    shifts = [[0, 1], [1, 0], [1, 1], [1, -1]]
    for i in range(len(shifts)):
        shifted_block = np.roll(block, shifts[i], axis=(0, 1))
        alpha, beta_l, beta_r = estimate_aggd_param(block * shifted_block)
        # Eq. 8
        mean = (beta_r - beta_l) * (gamma(2 / alpha) / gamma(1 / alpha))
        feat.extend([alpha, mean, beta_l, beta_r])
    return feat


def niqe(img,
         mu_pris_param,
         cov_pris_param,
         gaussian_window,
         block_size_h=96,
         block_size_w=96):
    """Calculate NIQE (Natural Image Quality Evaluator) metric.

    Ref: Making a "Completely Blind" Image Quality Analyzer.
    This implementation could produce almost the same results as the official
    MATLAB codes: http://live.ece.utexas.edu/research/quality/niqe_release.zip

    Note that we do not include block overlap height and width, since they are
    always 0 in the official implementation.

    For good performance, it is advisable by the official implemtation to
    divide the distorted image in to the same size patched as used for the
    construction of multivariate Gaussian model.

    Args:
        img (ndarray): Input image whose quality needs to be computed. The
            image must be a gray or Y (of YCbCr) image with shape (h, w).
            Range [0, 255] with float type.
        mu_pris_param (ndarray): Mean of a pre-defined multivariate Gaussian
            model calculated on the pristine dataset.
        cov_pris_param (ndarray): Covariance of a pre-defined multivariate
            Gaussian model calculated on the pristine dataset.
        gaussian_window (ndarray): A 7x7 Gaussian window used for smoothing the
            image.
        block_size_h (int): Height of the blocks in to which image is divided.
            Default: 96 (the official recommended value).
        block_size_w (int): Width of the blocks in to which image is divided.
            Default: 96 (the official recommended value).
    """
    assert img.ndim == 2, (
        'Input image must be a gray or Y (of YCbCr) image with shape (h, w).')
    # crop image
    h, w = img.shape
    num_block_h = math.floor(h / block_size_h)
    num_block_w = math.floor(w / block_size_w)
    img = img[0:num_block_h * block_size_h, 0:num_block_w * block_size_w]

    distparam = []  # dist param is actually the multiscale features
    for scale in (1, 2):  # perform on two scales (1, 2)
        mu = convolve(img, gaussian_window, mode='nearest')
        sigma = np.sqrt(
            np.abs(
                convolve(np.square(img), gaussian_window, mode='nearest') -
                np.square(mu)))
        # normalize, as in Eq. 1 in the paper
        img_nomalized = (img - mu) / (sigma + 1)

        feat = []
        for idx_w in range(num_block_w):
            for idx_h in range(num_block_h):
                # process ecah block
                block = img_nomalized[idx_h * block_size_h //
                                      scale:(idx_h + 1) * block_size_h //
                                      scale, idx_w * block_size_w //
                                      scale:(idx_w + 1) * block_size_w //
                                      scale]
                feat.append(compute_feature(block))

        distparam.append(np.array(feat))
        # TODO: matlab bicubic downsample with anti-aliasing
        # for simplicity, now we use opencv instead, which will result in
        # a slight difference.
        if scale == 1:
            h, w = img.shape
            img = cv2.resize(
                img / 255., (w // 2, h // 2), interpolation=cv2.INTER_LINEAR)
            img = img * 255.

    distparam = np.concatenate(distparam, axis=1)

    # fit a MVG (multivariate Gaussian) model to distorted patch features
    mu_distparam = np.nanmean(distparam, axis=0)
    # use nancov. ref: https://ww2.mathworks.cn/help/stats/nancov.html
    distparam_no_nan = distparam[~np.isnan(distparam).any(axis=1)]
    cov_distparam = np.cov(distparam_no_nan, rowvar=False)

    # compute niqe quality, Eq. 10 in the paper
    invcov_param = np.linalg.pinv((cov_pris_param + cov_distparam) / 2)
    quality = np.matmul(
        np.matmul((mu_pris_param - mu_distparam), invcov_param),
        np.transpose((mu_pris_param - mu_distparam)))
    quality = np.sqrt(quality)

    return quality


def calculate_niqe(img, crop_border, input_order='HWC', convert_to='y'):
    """Calculate NIQE (Natural Image Quality Evaluator) metric.

    Ref: Making a "Completely Blind" Image Quality Analyzer.
    This implementation could produce almost the same results as the official
    MATLAB codes: http://live.ece.utexas.edu/research/quality/niqe_release.zip

    We use the official params estimated from the pristine dataset.
    We use the recommended block size (96, 96) without overlaps.

    Args:
        img (ndarray): Input image whose quality needs to be computed.
            The input image must be in range [0, 255] with float/int type.
            The input_order of image can be 'HW' or 'HWC' or 'CHW'. (BGR order)
            If the input order is 'HWC' or 'CHW', it will be converted to gray
            or Y (of YCbCr) image according to the ``convert_to`` argument.
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the metric calculation.
        input_order (str): Whether the input order is 'HW', 'HWC' or 'CHW'.
            Default: 'HWC'.
        convert_to (str): Whether coverted to 'y' (of MATLAB YCbCr) or 'gray'.
            Default: 'y'.

    Returns:
        float: NIQE result.
    """

    # we use the official params estimated from the pristine dataset.
    niqe_pris_params = np.load('niqe_pris_params.npz')
    mu_pris_param = niqe_pris_params['mu_pris_param']
    cov_pris_param = niqe_pris_params['cov_pris_param']
    gaussian_window = niqe_pris_params['gaussian_window']

    img = img.astype(np.float32)
    if input_order != 'HW':
        img = reorder_image(img, input_order=input_order)
        if convert_to == 'y':
            img = to_y_channel(img)
        elif convert_to == 'gray':
            img = cv2.cvtColor(img / 255., cv2.COLOR_BGR2GRAY) * 255.
        img = np.squeeze(img)

    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border]

    niqe_result = niqe(img, mu_pris_param, cov_pris_param, gaussian_window)

    return niqe_result


def main(args):
    # model1 = pyiqa.create_metric('unique', as_loss=False)
    # model2 = pyiqa.create_metric('pi', as_loss=False)
    model3 = pyiqa.create_metric('musiq', as_loss=False)
    img_dir = args.test_dir


    # method =
    # ['engan', 'kind', 'zero_dce', 'sci', 'ruas', 'LLFlow', 'SGZ', 'retinexformer', 'URetinex', 'zeroIG', 'PIE']
    # for method in ['retinexformer', 'URetinex',  'zeroIG', 'PIE']: ['NPE','MEF','DICM','LIME','VV']
    # for dataset_name in ['NPE','MEF','DICM','VV']:
    #     img_dir = 'D:\\File\\lowlight\\python\\hap_run\\ablation\\lr_net1_11_gammavgg_relu4_3_SELU_2DA_2fea_ls_1g1d_mirror\\'
    #     # img_dir = 'E:\\results\\SNR_Net\\'
    #     # img_dir = img_dir + "\\" + dataset_name + "\\"
    #     # img_dir = img_dir + method + "\\" + dataset_name
    #     img_dir = img_dir + "\\" + dataset_name + "\\" + "images"

        # NOTE: add sorted
    path = sorted(glob(os.path.join(img_dir, '*')))
    list_niqe = []
    list_unique = []
    list_pi = []
    list_musiq = []

    for i in range(len(path)):
        # read images
        img = cv2.imread(path[i])

        # calculate scores
        niqe_num = calculate_niqe(img, 0, input_order='HWC', convert_to='gray')
    #
        # append to list
        list_niqe.append(niqe_num)

        # calculate scores
    #
    #     score = model1(path[i])
    #
    #     # append to list
    #     list_unique.append(score)
    #
    #     # calculate scores
    #
    #     score = model2(path[i])
    #
    #     # append to list
    #     list_pi.append(score)
    #
    #
        score = model3(path[i])

        # append to list
        list_musiq.append(score.cpu())



    # Average score for the dataset
    # print("======={}=======>".format(img_dir))
    print("NIQE:", "%.3f" % (np.mean(list_niqe)))
    # print("Average UNIQUE:", "%.3f" % (np.mean(list_unique)))
    # print("Average PI:", "%.3f" % (np.mean(list_pi)))
    print("MUSIQ:", "%.3f" % (np.mean(list_musiq)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument('--test_dir', type=str,
    #                     default="E:\\lowlight_dataset\\NIQE\\new2_11_vgg_relu4_3_SELU_fea2_nopatch_mirror_n2_psnr_ssim_l1\\VV",
    #                     help='directory for clean images')_20251001
    # parser.add_argument('--test_dir', type=str,
    #                     default="E:\\code_beifen\\SEEGANv2\\ablation\\lr_net1_11_gammavgg_relu4_3_SELU_2DA_2fea_ls_1g1d_mirror_20251031\\VV\\images",
    #                     help='directory for clean images')
    # parser.add_argument('--test_dir', type=str,default='E:\\code_beifen\\SEEGANv2\\ablation\\test_lime')
    # parser.add_argument('--test_dir', type=str, default='E:\\lowlight_dataset\\NIQE\\NPE\\low')
    # parser.add_argument('--test_dir', type=str, default='E:\\results\\other\\ReDDiT\\LIME')
    parser.add_argument('--test_dir', type=str, default="D:\\File\\lowlight\\test\\test2",help='directory for testing inputs')
    # parser.add_argument('--test_dir', default='E:\\code_beifen\\SEEGANv2\\ablation\\lr_net1_11_gammavgg_relu4_3_SELU_2DA_2fea_ls_1g1d_mirror_20251031\\NPE\\images', type=str)  # HVI LEDNet FourierDiff
    #
    # parser.add_argument('--test_dir', type=str,default="D:\\File\\lowlight\\test\\test2",help='directory for clean images')
    # parser.add_argument('--test_dir', type=str,default="F:\\test_lime",help='directory for clean images')
    args = parser.parse_args()
    main(args)
    # parser.add_argument('-dirA', default='E:\\lowlight_dataset\\VE-LOL-L\\VE-LOL-L-Cap-Full\\VE-LOL-L-Cap-Normal_test', type=str)
    # parser.add_argument('-dirA', default='E:\\lowlight_dataset\\LOLdataset\\eval15\\high', type=str)
    # engan RetinexNet MBLLEN kind zero_dce zero_dce++ sci ruas LLFlow SGZ SNR_Net retinexformer URetinex  zeroIG PIE PairLIE
    # parser.add_argument('-dirB', default='E:\\results\\engan\\ve_lol', type=str)