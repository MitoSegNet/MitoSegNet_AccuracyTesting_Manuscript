from skimage import img_as_bool, io, color
from skimage.morphology import skeletonize
from skan import summarise
import cv2
import os
from skimage.measure import label, regionprops

import warnings
warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore", UserWarning)


org_path = ""

img_list = os.listdir(org_path)

for image in img_list:

    read_lab_skel = img_as_bool(color.rgb2gray(io.imread(org_path + os.sep + image)))
    lab_skel = skeletonize(read_lab_skel).astype("uint8")

    branch_data = summarise(lab_skel)

    grouped_branch_data = branch_data.groupby(["skeleton-id"], as_index=False).mean()

    #print(branch_data)
    print(grouped_branch_data)

    img = cv2.imread(org_path + os.sep + image, cv2.IMREAD_GRAYSCALE)

    gt_label = label(img)
    gt_reg_props = regionprops(label_image=gt_label, coordinates='xy')

    print(len(grouped_branch_data), len(gt_reg_props))

    break

