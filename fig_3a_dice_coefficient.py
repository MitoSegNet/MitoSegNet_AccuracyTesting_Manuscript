"""

Calculating the dice coefficient of segmentation predictions

Calculating statistics based on dice coefficient results

Plotting dice coefficient results

"""

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
from scipy.stats.mstats import normaltest, mannwhitneyu, kruskal
from scikit_posthocs import posthoc_dunn
from Plot_Significance import significance_bar
from effect_size import cohens_d

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)

# path where images are stored
path = "C:/Users/Christian/Desktop/Fourth_CV/Complete_images"

# name of folder in which ground truth images are stored
gt_folder = "Ground_Truth"

# list of segmentations against which to compare with ground truth
seg_list = ["MitoSegNet", "Finetuned Fiji U-Net", "Ilastik", "Gaussian", "Hessian", "Laplacian"]

# list of images to compare against each other (names have to be identical)
img_list = os.listdir(path + os.sep + gt_folder)

# empty data frame to store dice coefficient values
all_data = pd.DataFrame(columns=seg_list)

for folder in seg_list:

    dice_l = []
    for imgs in img_list:

        gt = cv2.imread(path + os.sep + gt_folder + os.sep + imgs, cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(path + os.sep + folder + os.sep + imgs, cv2.IMREAD_GRAYSCALE)

        dice = (np.sum(pred[gt == 255]) * 2.0) / (np.sum(pred) + np.sum(gt))

        dice_l.append(dice)

    all_data[folder] = dice_l

#print(all_data)
#all_data.to_csv(path + "/dice_coefficient_table.csv")


"""
# checking if data is normally distributed
for seg in all_data:
    print(seg, normaltest(all_data[seg]))
"""

# hypothesis testing
kwt = kruskal([all_data["MitoSegNet"], all_data["Finetuned Fiji U-Net"], all_data["Ilastik"], all_data["Gaussian"],
                  all_data["Hessian"], all_data["Laplacian"]])
#print(kwt)

dt = posthoc_dunn([all_data["MitoSegNet"], all_data["Finetuned Fiji U-Net"], all_data["Ilastik"], all_data["Gaussian"],
                  all_data["Hessian"], all_data["Laplacian"]])
#print(dt)
#dt.to_excel("dc_posthoc.xlsx")

#print(cohens_d(all_data["MitoSegNet"], all_data["Ilastik"]))


# pos_y and pos_x determine position of bar, p sets the number of asterisks, y_dist sets y distance of the asterisk to
# bar, and distance sets the distance between two or more asterisks

dist = 0.08
bar_y = 0.02

significance_bar(pos_y=1, pos_x=[0, 2], bar_y=bar_y, p=2, y_dist=bar_y, distance=dist)
significance_bar(pos_y=1.05, pos_x=[0, 3], bar_y=bar_y, p=3, y_dist=bar_y, distance=dist)
significance_bar(pos_y=1.1, pos_x=[0, 4], bar_y=bar_y, p=1, y_dist=bar_y, distance=dist)
significance_bar(pos_y=1.15, pos_x=[0, 5], bar_y=bar_y, p=2, y_dist=bar_y, distance=dist)

all_data.rename(columns={"Finetuned Fiji U-Net": "Finetuned\nFiji U-Net"}, inplace=True)

n = sb.boxplot(data=all_data, color="white", fliersize=0)
sb.swarmplot(data=all_data, color="black")

n.set_ylabel("Dice coefficient", fontsize=32)
n.tick_params(axis="x", labelsize=34, rotation=45)
n.tick_params(axis="y", labelsize=28)

plt.show()












