"""

Calculate energy distance between predicted descriptor distributions and ground truth predictor distributions

Generate morphological energy distance tables

"""

import cv2
from skimage.measure import regionprops, label
import os
from scipy.stats.mstats import mannwhitneyu, normaltest, ttest_ind
from scipy.stats import energy_distance
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from branch_measurements import get_branch_meas

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)

org_path = ""
path = org_path + os.sep + "Complete_images"
save_path = org_path + os.sep + "Morph_Energy_Distance_Extended"


# Get image data
#################################

def get_energy_distance(path, save_path, seg_name):

    gt_folder_path = path + os.sep + "Ground_Truth"
    raw_folder_path = path + os.sep + "Original"
    seg_folder_path = path + os.sep + seg_name

    gt_image_list = os.listdir(gt_folder_path)

    columns = ["Image", "Area", "Eccentricity", "Aspect Ratio", "Perimeter", "Solidity", "Number of branches",
               "Branch length", "Total branch length", "Curvature index", "Mean intensity"]

    df = pd.DataFrame(columns= columns)

    list_area = []
    list_ecc = []
    list_ar = []
    list_per = []
    list_sol = []

    list_nb = []
    list_bl = []
    list_tbl = []
    list_ci = []

    list_i = []

    for image in gt_image_list:

        print(image)

        gt = cv2.imread(gt_folder_path + os.sep + image, cv2.IMREAD_GRAYSCALE)
        seg = cv2.imread(seg_folder_path + os.sep + image, cv2.IMREAD_GRAYSCALE)
        org = cv2.imread(raw_folder_path + os.sep + image, cv2.IMREAD_GRAYSCALE)

        # label image mask
        gt_labelled = label(gt)
        seg_labelled = label(seg)

        # Get region props of labelled images
        gt_reg_props = regionprops(label_image=gt_labelled, intensity_image=org, coordinates='xy')
        seg_reg_props = regionprops(label_image=seg_labelled, intensity_image=org, coordinates='xy')

        # compare shape descriptor distributions

        # Area
        gt_area = [i.area for i in gt_reg_props]
        seg_area = [i.area for i in seg_reg_props]

        list_area.append(energy_distance(gt_area, seg_area))

        # Eccentricity
        gt_ecc = [i.eccentricity for i in gt_reg_props]
        seg_ecc = [i.eccentricity for i in seg_reg_props]

        list_ecc.append(energy_distance(gt_ecc, seg_ecc))

        # Aspect ratio
        gt_ar = [i.major_axis_length/i.minor_axis_length for i in gt_reg_props]
        seg_ar = [i.major_axis_length/i.minor_axis_length for i in seg_reg_props]

        list_ar.append(energy_distance(gt_ar, seg_ar))

        # Perimeter
        gt_per = [i.perimeter for i in gt_reg_props]
        seg_per = [i.perimeter for i in seg_reg_props]

        list_per.append(energy_distance(gt_per, seg_per))

        # Solidity
        gt_sol = [i.solidity for i in gt_reg_props]
        seg_sol = [i.solidity for i in seg_reg_props]

        list_sol.append(energy_distance(gt_sol, seg_sol))

        # branch descriptors
        gt_nb, gt_bl, gt_tbl, gt_ci = get_branch_meas(gt_folder_path + os.sep + image)
        seg_nb, seg_bl, seg_tbl, seg_ci = get_branch_meas(seg_folder_path + os.sep + image)

        list_nb.append(energy_distance(gt_nb, seg_nb))
        list_bl.append(energy_distance(gt_bl, seg_bl))
        list_tbl.append(energy_distance(gt_tbl, seg_tbl))
        list_ci.append(energy_distance(gt_ci, seg_ci))

        # Intensity
        gt_int = [i.mean_intensity for i in gt_reg_props]
        seg_int = [i.mean_intensity for i in seg_reg_props]

        list_i.append(energy_distance(gt_int, seg_int))

        def show(gt, seg):

            sb.distplot(gt, color="green")
            sb.distplot(seg, color="red")
            plt.show()

        def norm(gt, seg):

            print(normaltest(gt)[1])
            print(normaltest(seg)[1])

    df["Image"] = gt_image_list
    df["Area"] = list_area
    df["Eccentricity"] = list_ecc
    df["Aspect Ratio"] = list_ar
    df["Perimeter"] = list_per
    df["Solidity"] = list_sol
    df["Number of branches"] = list_nb
    df["Branch length"] = list_bl
    df["Total branch length"] = list_tbl
    df["Curvature index"] = list_ci
    df["Mean intensity"] = list_i

    # raw data
    df.to_csv(save_path + os.sep + seg_name + "_EnergyDistance.csv")


# morph distribution comparison
seg_list = ["MitoSegNet", "Ilastik", "Gaussian", "Hessian", "Laplacian"]

for seg in seg_list:
    get_energy_distance(path, save_path, seg)






