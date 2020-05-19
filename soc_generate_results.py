
"""

soc: single object comparison

Calculate fold deviation between corresponding single object predicted descriptors and ground truth predictors and
percentage of false object detections

Generate tables (per image) with fold devations per object, per segmentation method

#######################

How are objects classified as corresponding:

If pixel coordinate of object in both ground truth and prediction is the same, object correspondence is assumed
(minimum of 1 pixel overlap).

Two new lists are created in which the duplicates of the ground truth labels (false split event) and the
segmentation labels (false merge) are found

Two additional lists in which all the objects with no correspondence was found are also created (missing seg_objects
(missing) or missing gt_objects (falsely added))

Comparing shape descriptors of every segmented object to object in mask (ground truth) and calculating
deviation in fold change (e.g. if gt = 1 and seg = 2, fold change is 2 (gt = 1 and seg = 0.5, fold change is also 2))

#######################

(gt_object, seg_object)

(value, None): missing
(None, value): falsely added
([value], value1, value2 ...): false split
(value1, value2 ..., [value]):  false merge

"""

import cv2
import numpy as np
from skimage.measure import regionprops, label
import skimage
import os
import pandas as pd
import itertools
import copy
import collections
import matplotlib.pyplot as plt
import seaborn as sb

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)


def create_data(path, save_path, save_path_error, seg_name):

    # Get image data
    #################################
    gt_path = path + os.sep + "Ground_Truth"
    seg_path = path + os.sep + seg_name

    gt_path_imgs = os.listdir(gt_path)
    org_path = path + os.sep + "Original"

    image_dictionary = {}

    for image in gt_path_imgs:
        image_dictionary.update({image: []})

    # loop through ground truth and segmentation folder
    for gt_img in gt_path_imgs:

        print(gt_img, "\n")

        gt = cv2.imread(gt_path + os.sep + gt_img, cv2.IMREAD_GRAYSCALE)
        seg = cv2.imread(seg_path + os.sep + gt_img, cv2.IMREAD_GRAYSCALE)
        org = cv2.imread(org_path + os.sep + gt_img, cv2.IMREAD_GRAYSCALE)

        # label image mask
        gt_labelled = label(gt)
        seg_labelled = label(seg)

        # Get region props of image data
        gt_reg_props = regionprops(label_image=gt_labelled, intensity_image=org)
        seg_reg_props = regionprops(label_image=seg_labelled, intensity_image=org)

        gt_props_dic = {}
        seg_props_dic = {}

        # generating dictionaries for gt and segmented objects using the labels as keys
        for i in gt_reg_props:
            gt_props_dic.update({i.label: i})

        for i in seg_reg_props:
            seg_props_dic.update({i.label: i})

        # Generate list containing information about object correspondence
        # only add coordinates of objects that have an area of greater than 10 (other objects are disregarded)
        gt_object_coords = [n.coords.tolist() for n in gt_reg_props if n.area >= 10]
        seg_object_coords = [n.coords.tolist() for n in seg_reg_props if n.area >= 10]

        # list of same length as object_coords, containing label information
        gt_object_labels = [n.label for n in gt_reg_props if n.area >= 10]
        seg_object_labels = [n.label for n in seg_reg_props if n.area >= 10]

        l = []
        # comparing object pixel coordinates to check which objects correspond to each other
        print("=== Comparing objects ===", "\n")
        for gt_label, gt_object in zip(gt_object_labels, gt_object_coords):

            progress = (gt_label - 1) / len(gt_object_coords) * 100
            print("%.2f" % progress, "%")

            # each gt_object is list containing coordinates of that object
            for gt_coordinates in gt_object:

                # [[(ground truth label, prediction label), ground truth reg_pros, predicted reg_props], ...
                for seg_label, seg_object in zip(seg_object_labels, seg_object_coords):

                    # if coordinates of gt object found in coordinate list of seg object, object is corresponding
                    # currently set to 1 pixel
                    if gt_coordinates in seg_object:
                        l.append([(gt_label, seg_label), gt_props_dic[gt_label], seg_props_dic[seg_label]])

                        break

        print("100 %")

        print("\n", "=== Object comparison completed ===", "\n")

        # sort list and remove duplicates of lists in list
        l.sort()
        # list l contains information of corresponding objects
        l = (list(k for k, _ in itertools.groupby(l)))

        # generating separate lists in which corresponding labels are added as well as regionprops of that object
        gt_l = []
        seg_l = []
        for count, i in enumerate(l):
            gt_l.append(i[0][0])
            seg_l.append(i[0][1])

        # find duplicates in gt_l (false split) and seg_l (false merge)
        gt_l_duplicates = ([item for item, count in collections.Counter(gt_l).items() if count > 1])
        seg_l_duplicates = ([item for item, count in collections.Counter(seg_l).items() if count > 1])

        # adding entries for missing seg_objects (missing) or missing gt_objects (falsely added)
        for count, gt in enumerate(gt_l):

            # 0 labels do not exist (except for background) and since labels do not start from 0, len(reg_props)
            # (nr of objects) > count -> prevents adding false labels

            if count not in gt_l:

                if len(gt_reg_props) >= count > 0:
                    l.insert(count, [(count, None)])

            if count not in seg_l:

                if len(seg_reg_props) >= count > 0:
                    l.insert(count, [(None, count)])

        # re-order list based on duplicate and merge events
        # create copy of list l in which split and merge entries are removed
        new_l = copy.copy(l)

        # create dictionaries in which entries for splits and merges can later be added
        dic_split = {}
        dic_merge = {}
        for i in gt_l_duplicates:
            dic_split.update({i: []})

        for i in seg_l_duplicates:
            dic_merge.update({i: []})

        # add split and merge entries to appending dictionaries and remove them from new_l
        gt_old_i = "None"
        seg_old_i = "None"

        for count, i in enumerate(l):

            if i[0][0] in gt_l_duplicates:

                new_l.remove(i)

                # add gt regprops only at beginning of each dictionary key
                if gt_old_i == "None" or gt_old_i != i[0][0]:
                    dic_split[i[0][0]].append(i[1])

                dic_split[i[0][0]].append([i[0][1], i[2]])

                gt_old_i = i[0][0]

            if i[0][1] in seg_l_duplicates:

                # if item i has already been deleted previously this line below will avoid a value error
                try:
                    new_l.remove(i)
                except ValueError:
                    continue

                if seg_old_i == "None" or i[0][1] != seg_old_i:
                    dic_merge[i[0][1]].append(i[2])

                dic_merge[i[0][1]].append([i[0][0], i[1]])

                seg_old_i = i[0][1]

        # checking if all elements after first element of list dic_merge[i] are lists, if not they are removed
        for i in dic_merge:

            # print(i, dic_merge[i])
            for count, n in enumerate(dic_merge[i]):
                if count > 0:
                    if isinstance(n, list):
                        pass
                    else:

                        dic_merge[i].pop(count)

        # Create data
        """

        Create tables with deviation from morphological properties of ground truth objects 

        i[1] = gt value
        i[2] = seg value

        area
        eccentricity 
        major_axis_length
        minor_axis_length
        perimeter 
        solidity    

        """

        dev_dic = {"objects": [], "area": [], "aspect ratio": [], "eccentricity": [], "perimeter": [], "solidity": [],
                   "mean intensity": []}

        # regular comparison (no split or merge events)

        for i in new_l:

            if len(i) > 1:

                # area
                dev_area = i[2].area / i[1].area

                if dev_area < 1:
                    dev_area = 1 / dev_area

                # aspect ratio
                asr_1 = i[1].major_axis_length / i[1].minor_axis_length
                asr_2 = i[2].major_axis_length / i[2].minor_axis_length

                """
                if i[2].minor_axis_length != 0:
                    asr_2 = i[2].major_axis_length / i[2].minor_axis_length

                else:

                    asr_2 = i[2].major_axis_length
                """

                dev_ar = asr_2 / asr_1

                if dev_ar < 1:
                    dev_ar = 1 / dev_ar

                #### necessary because values can be 0

                if i[1].eccentricity == 0 and i[2].eccentricity == 0:
                    dev_ecc = 1

                elif i[1].eccentricity == 0:
                    if i[2].eccentricity < 1:
                        dev_ecc = i[2].eccentricity + 1

                    else:
                        dev_ecc = i[2].eccentricity

                elif i[2].eccentricity == 0:
                    if i[1].eccentricity < 1:
                        dev_ecc = i[1].eccentricity + 1

                    else:
                        dev_ecc = i[1].eccentricity

                else:
                    dev_ecc = i[2].eccentricity / i[1].eccentricity

                    if dev_ecc < 1:
                        dev_ecc = 1 / dev_ecc


                ### alternative method to calculate orientation deviation
                dev_per = i[2].perimeter / i[1].perimeter

                if dev_per < 1:
                    dev_per = 1 / dev_per

                dev_sol = i[2].solidity / i[1].solidity

                if dev_sol < 1:
                    dev_sol = 1 / dev_sol

                dev_mean_int = i[2].mean_intensity / i[1].mean_intensity

                if dev_mean_int < 1:
                    dev_mean_int = 1 / dev_mean_int

                dev_dic["objects"].append(i[0])
                dev_dic["area"].append(dev_area)
                dev_dic["aspect ratio"].append(dev_ar)
                dev_dic["eccentricity"].append(dev_ecc)
                dev_dic["perimeter"].append(dev_per)
                dev_dic["solidity"].append(dev_sol)
                dev_dic["mean intensity"].append(dev_mean_int)


            # if no corresponding objects are found (either missing or falsely added)
            else:

                dev_dic["objects"].append(i[0])

                # excluding added and missing objects

                dev_dic["area"].append(None)
                dev_dic["aspect ratio"].append(None)
                dev_dic["eccentricity"].append(None)
                dev_dic["perimeter"].append(None)
                dev_dic["solidity"].append(None)
                dev_dic["mean intensity"].append(None)


        # comparison of objects with false split events
        for i in dic_split:

            o_l = []
            l_area = []
            l_ar = []
            l_ecc = []
            l_per = []
            l_sol = []
            l_int = []

            if len(dic_split[i]) != 0:

                for count, i2 in enumerate(dic_split[i]):

                    if count == 0:

                        o_l.append([i])
                        gt_area = i2.area
                        gt_ar = i2.major_axis_length / i2.minor_axis_length
                        gt_ecc = i2.eccentricity
                        gt_per = i2.perimeter
                        gt_sol = i2.solidity
                        gt_int = i2.mean_intensity

                    else:

                        o_l.append(i2[0])
                        l_area.append(i2[1].area)
                        l_ar.append(i2[1].major_axis_length / i2[1].minor_axis_length)
                        l_ecc.append(i2[1].eccentricity)
                        l_per.append(i2[1].perimeter)
                        l_sol.append(i2[1].solidity)
                        l_int.append(i2[1].mean_intensity)

                dev_area = np.average(l_area) / gt_area

                if dev_area < 1:
                    dev_area = 1 / dev_area

                dev_ar = np.average(l_ar) / gt_ar

                if dev_ar < 1:
                    dev_ar = 1 / dev_ar


                if gt_ecc == 0 and np.average(l_ecc) == 0:

                    dev_ecc = 1

                elif gt_ecc == 0:
                    if np.average(l_ecc) < 1:
                        dev_ecc = np.average(l_ecc) + 1

                    else:
                        dev_ecc = np.average(l_ecc)

                elif np.average(l_ecc) == 0:
                    if gt_ecc < 1:
                        dev_ecc = gt_ecc + 1

                    else:
                        dev_ecc = gt_ecc

                else:
                    dev_ecc = np.average(l_ecc) / gt_ecc
                    if dev_ecc < 1:
                        dev_ecc = 1 / dev_ecc

                dev_per = np.average(l_per) / gt_per

                if dev_per < 1:
                    dev_per = 1 / dev_per


                dev_sol = np.average(l_sol) / gt_sol

                if dev_sol < 1:
                    dev_sol = 1 / dev_sol

                dev_int = np.average(l_int) / gt_int

                if dev_int < 1:
                    dev_int = 1 / dev_int

                if len(o_l) > 1:
                    dev_dic["objects"].append(o_l)

                    dev_dic["area"].append(dev_area)
                    dev_dic["aspect ratio"].append(dev_ar)
                    dev_dic["eccentricity"].append(dev_ecc)
                    dev_dic["perimeter"].append(dev_per)
                    dev_dic["solidity"].append(dev_sol)
                    dev_dic["mean intensity"].append(dev_int)

        # comparison of objects with false merge events

        for i in dic_merge:

            o_l = []
            l_area = []
            l_ar = []
            l_ecc = []
            l_per = []
            l_sol = []
            l_int = []

            # only run it if any false merges have been found
            if len(dic_merge[i]) != 0:

                for count, i2 in enumerate(dic_merge[i]):

                    if count == 0:

                        seg_area = i2.area
                        seg_ar = i2.major_axis_length / i2.minor_axis_length
                        seg_ecc = i2.eccentricity
                        seg_per = i2.perimeter
                        seg_sol = i2.solidity
                        seg_int = i2.mean_intensity

                    else:

                        o_l.append(i2[0])

                        l_area.append(i2[1].area)
                        l_ar.append(i2[1].major_axis_length / i2[1].minor_axis_length)
                        l_ecc.append(i2[1].eccentricity)
                        l_per.append(i2[1].perimeter)
                        l_sol.append(i2[1].solidity)
                        l_int.append(i2[1].mean_intensity)

                o_l.append([i])


                dev_area = seg_area / np.average(l_area)

                if dev_area < 1:
                    dev_area = 1 / dev_area

                dev_ar = seg_ar / np.average(l_ar)

                if dev_ar < 1:
                    dev_ar = 1 / dev_ar

                if np.average(l_ecc) == 0 and seg_ecc == 0:
                    dev_ecc = 1

                elif seg_ecc == 0:
                    if np.average(l_ecc) < 1:
                        dev_ecc = np.average(l_ecc) + 1

                    else:
                        dev_ecc = np.average(l_ecc)

                elif np.average(l_ecc) == 0:
                    if seg_ecc < 1:
                        dev_ecc = seg_ecc + 1

                    else:
                        dev_ecc = seg_ecc

                else:
                    dev_ecc = seg_ecc / np.average(l_ecc)
                    if dev_ecc < 1:
                        dev_ecc = 1 / dev_ecc

                dev_per = seg_per / np.average(l_per)

                if dev_per < 1:
                    dev_per = 1 / dev_per

                dev_sol = seg_sol / np.average(l_sol)

                if dev_sol < 1:
                    dev_sol = 1 / dev_sol

                dev_int = seg_int / np.average(l_int)

                if dev_int < 1:
                    dev_int = 1 / dev_int

                if len(o_l) > 1:
                    dev_dic["objects"].append(o_l)
                    dev_dic["area"].append(dev_area)
                    dev_dic["aspect ratio"].append(dev_ar)
                    dev_dic["eccentricity"].append(dev_ecc)
                    dev_dic["perimeter"].append(dev_per)
                    dev_dic["solidity"].append(dev_sol)
                    dev_dic["mean intensity"].append(dev_int)

        # creating empty data frame for csv file (raw data) creation
        data = pd.DataFrame(index=list(range(0, len(dev_dic["objects"]))))

        for i in dev_dic:
            data[i] = dev_dic[i]

            image_dictionary[gt_img].append([dev_dic[i]])

        data.to_csv(save_path + os.sep + gt_img + ".csv")

        """
        (value, None): missing 
        (None, value): falsely added
        """

        # calculating number of falsely split (based on seg) / merged (based on seg) / added (based on seg) /
        # missing objects (based on gt)
        split = 0
        merge = 0

        added = 0
        missing = 0

        for obj in dev_dic["objects"]:

            try:

                if obj[0] == None:
                    added += 1
                if obj[1] == None:
                    missing += 1

                if None not in obj and len(obj) > 2:

                    if isinstance(obj[0], list):
                        split += (len(obj) - 1)

                    else:
                        merge += 1

            except IndexError:
                pass



def analyse_data(path, save_path_analysis, seg_name):

    path_data = os.listdir(path)

    average_dictionary = {"image": [], "area": [], "aspect ratio": [], "eccentricity": [],
                          "perimeter": [], "solidity": [], "mean intensity": []}

    for csv_file in path_data:

        data = pd.read_csv(path + os.sep + csv_file)

        new_area_l = []
        new_ar_l = []
        new_ecc_l = []
        new_per_l = []
        new_sol_l = []
        new_int_l = []

        for obj, area, ar, ecc, per, sol, mint in zip(data["objects"], data["area"], data["aspect ratio"],
                                                     data["eccentricity"], data["perimeter"], data["solidity"],
                                                     data["mean intensity"]):

            # take only single object correspondence into account: exclude false split, merge, false positive or negative
            if "None" not in obj and "[" not in obj:

                new_area_l.append(area)
                new_ar_l.append(ar)
                new_ecc_l.append(ecc)
                new_per_l.append(per)
                new_sol_l.append(sol)
                new_int_l.append(mint)

        average_dictionary["image"].append(csv_file)
        average_dictionary["area"].append(np.nanmean(new_area_l))
        average_dictionary["aspect ratio"].append(np.nanmean(new_ar_l))
        average_dictionary["eccentricity"].append(np.nanmean(new_ecc_l))
        average_dictionary["perimeter"].append(np.nanmean(new_per_l))
        average_dictionary["solidity"].append(np.nanmean(new_sol_l))
        average_dictionary["mean intensity"].append(np.nanmean(new_int_l))

    analysed_data = pd.DataFrame(index=list(range(0, len(path_data))))

    for i in average_dictionary:
        analysed_data[i] = average_dictionary[i]

    # removing first column
    analysed_data.drop(analysed_data.columns[[0]], axis=1, inplace=True)

    analysed_data.to_csv(save_path_analysis + os.sep + seg_name + "_analysed_data.csv")


#single object comparison + object error analysis
seg_list = ["MitoSegNet", "Ilastik", "Gaussian", "Hessian", "Laplacian"]

path = ""
save_path_analysis = path + os.sep + "Single_Object_Comparison_Average"
save_path_error = path + os.sep + "False_Object_Detection"

if not os.path.isdir(save_path_analysis):
    os.mkdir(save_path_analysis)

if not os.path.isdir(path + os.sep + "Single_Object_Comparison_Raw"):
    os.mkdir(path + os.sep + "Single_Object_Comparison_Raw")

if not os.path.isdir(save_path_error):
    os.mkdir(save_path_error)

for seg in seg_list:

    if not os.path.isdir(path + os.sep + "Single_Object_Comparison_Raw" + os.sep + seg):
        os.mkdir(path + os.sep + "Single_Object_Comparison_Raw"  + os.sep + seg)

    save_path = path + os.sep + "Single_Object_Comparison_Raw"  + os.sep + seg

    create_data(path, save_path, save_path_error, seg)
    analyse_data(save_path, save_path_analysis, seg)


