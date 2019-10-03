"""

Calculate p-values by comparing predicted descriptor distributions with ground truth predictor distributions

Generate morphological comparison p-value tables

"""

import cv2
import os
import pandas as pd
import collections
from scipy.stats.mstats import mannwhitneyu, normaltest, ttest_ind
from skimage.measure import regionprops, label
from skimage import img_as_bool, io, color
from skimage.morphology import skeletonize
from skan import summarise
from branch_measurements import get_branch_meas

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)

path = "C:/Users/Christian/Desktop/Fourth_CV/Complete_images"


# Get image data

def morph_distribution(path, seg_name):


    gt_folder_path = path + os.sep + "Ground_Truth"
    seg_folder_path = path + os.sep + seg_name
    org_folder_path = path + os.sep + "Original"

    image_list = os.listdir(gt_folder_path)

    columns = ["Image", "Area", "Eccentricity", "Aspect Ratio", "Perimeter", "Solidity", "Number of branches",
               "Branch length", "Total branch length", "Curvature index", "Mean intensity"]

    df = pd.DataFrame(columns=columns)

    list_area = []
    list_ecc = []
    list_ar = []
    list_per = []
    list_sol = []

    list_nb = []
    list_tbl = []
    list_bl = []
    list_ci = []

    list_int = []


    # check for significant difference between two distributions
    def check_significance(gt_dist, seg_dist):

        normal_gt = normaltest(gt_dist)[1]
        normal_seg = normaltest(seg_dist)[1]

        # if both distributions are parametric use t-test, else use mann-whitney-u
        if normal_gt > 0.05 and normal_seg > 0.05:

            pvalue = ttest_ind(gt_dist, seg_dist)[1]

        else:
            pvalue = mannwhitneyu(gt_dist, seg_dist)[1]


        if pvalue > 0.05:
            return 0

        if 0.01 < pvalue < 0.05:
            return 1

        if 0.001 < pvalue < 0.01:
            return 2

        if pvalue < 0.001:
            return 3

        return pvalue

    for image in image_list:

        print(image)

        gt = cv2.imread(gt_folder_path + os.sep + image, cv2.IMREAD_GRAYSCALE)
        seg = cv2.imread(seg_folder_path + os.sep + image , cv2.IMREAD_GRAYSCALE)
        org = cv2.imread(org_folder_path + os.sep + image, cv2.IMREAD_GRAYSCALE)

        gt_nb, gt_bl, gt_tbl, gt_ci = get_branch_meas(gt_folder_path + os.sep + image)
        seg_nb, seg_bl, seg_tbl, seg_ci = get_branch_meas(seg_folder_path + os.sep + image)

        list_nb.append(check_significance(gt_nb, seg_nb))
        list_bl.append(check_significance(gt_bl, seg_bl))
        list_tbl.append(check_significance(gt_tbl, seg_tbl))
        list_ci.append(check_significance(gt_ci, seg_ci))

        # label image mask
        gt_labelled = label(gt)
        seg_labelled = label(seg)

        # Get region props of labelled images
        gt_reg_props = regionprops(label_image=gt_labelled, intensity_image=org, coordinates='xy')
        seg_reg_props = regionprops(label_image=seg_labelled, intensity_image=org, coordinates='xy')

        # compare shape descriptor distributions
        #################################

        # Intensity
        gt_int = [i.mean_intensity for i in gt_reg_props]
        seg_int = [i.mean_intensity for i in seg_reg_props]

        list_int.append(check_significance(gt_int, seg_int))

        # Area
        gt_area = [i.area for i in gt_reg_props]
        seg_area = [i.area for i in seg_reg_props]

        list_area.append(check_significance(gt_area, seg_area))

        # Eccentricity
        gt_ecc = [i.eccentricity for i in gt_reg_props]
        seg_ecc = [i.eccentricity for i in seg_reg_props]

        list_ecc.append(check_significance(gt_ecc, seg_ecc))

        # Aspect ratio
        gt_ar = [i.major_axis_length/i.minor_axis_length for i in gt_reg_props if i.minor_axis_length != 0]
        seg_ar = [i.major_axis_length/i.minor_axis_length for i in seg_reg_props if i.minor_axis_length != 0]

        list_ar.append(check_significance(gt_ar, seg_ar))

        # Perimeter
        gt_per = [i.perimeter for i in gt_reg_props]
        seg_per = [i.perimeter for i in seg_reg_props]

        list_per.append(check_significance(gt_per, seg_per))

        # Solidity
        gt_sol = [i.solidity for i in gt_reg_props]
        seg_sol = [i.solidity for i in seg_reg_props]

        list_sol.append(check_significance(gt_sol, seg_sol))


    df["Image"] = image_list
    df["Area"] = list_area
    df["Eccentricity"] = list_ecc
    df["Aspect Ratio"] = list_ar
    df["Perimeter"] = list_per
    df["Solidity"] = list_sol
    df["Number of branches"] = list_nb
    df["Branch length"] = list_bl
    df["Total branch length"] = list_tbl
    df["Curvature index"] = list_ci
    df["Mean intensity"] = list_int


    # raw data
    df.to_csv(path + os.sep + seg_name + "_Morph_Dist_comparison.csv")


# morph distribution comparison
seg_list = ["MitoSegNet", "Finetuned Fiji U-Net", "Ilastik", "Gaussian", "Hessian", "Laplacian"]

for seg in seg_list:
    morph_distribution(path, seg)
















