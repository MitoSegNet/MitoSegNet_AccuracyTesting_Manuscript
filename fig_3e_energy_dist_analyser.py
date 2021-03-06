"""

Analyse tables generated by energy_distance_generate_results.py

Statistical analysis

Create box plot

"""

import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from scipy.stats.mstats import normaltest, kruskal
from scikit_posthocs import posthoc_dunn
from Plot_Significance import significance_bar
from effect_size import cohens_d

import warnings
warnings.simplefilter("ignore", UserWarning)

path = ""

file_list = ['MitoSegNet_EnergyDistance.csv', 'Fiji_U-Net_EnergyDistance.csv', 'Ilastik_EnergyDistance.csv',
             'Gaussian_EnergyDistance.csv', 'Hessian_EnergyDistance.csv', 'Laplacian_EnergyDistance.csv']

seg_list = ["MitoSegNet", "Finetuned\nFiji U-Net", "Ilastik", "Gaussian", "Hessian", "Laplacian"]

all_data = pd.DataFrame(columns=seg_list)

for file, method in zip(file_list, all_data):

    if ".csv" in file:

        #print(file, method)
        table = pd.read_csv(path + "/" + file)

        # removing first column
        table.drop(table.columns[[0,1]], axis=1, inplace=True)

        # possibility to only read in one descriptor instead of all descriptors
        l = table.values.tolist()
        #l = table[descriptor].tolist()

        flat_list = [item for sublist in l for item in sublist]

        all_data[method] = flat_list #l
        #all_data[method] = l

"""
# check average values 
for seg in seg_list:
    print(seg, np.average(all_data[seg]))
print("\n")

# check normality p-value 
for seg in seg_list:
    print(seg, normaltest(all_data[seg]))
print("\n")
"""


print(kruskal(all_data["Gaussian"].tolist(), all_data["Hessian"].tolist(), all_data["Laplacian"].tolist(),
              all_data["Ilastik"].tolist(), all_data["MitoSegNet"].tolist(), all_data["Finetuned\nFiji U-Net"].tolist()))

dt = posthoc_dunn([all_data["Gaussian"].tolist(), all_data["Hessian"].tolist(), all_data["Laplacian"].tolist(),
              all_data["Ilastik"].tolist(), all_data["MitoSegNet"].tolist(), all_data["Finetuned\nFiji U-Net"].tolist()])

print(dt)
#dt.to_excel("ed_posthoc.xlsx")

# check effect sizes of segmentations against msn segmentation
for seg in seg_list:

    if seg != "MitoSegNet":
        print(seg, cohens_d(all_data[seg], all_data["MitoSegNet"]))


pos_y_start = 6
dist_bar_y = 0.2
significance_bar(pos_y=pos_y_start+0.5, pos_x=[0, 2], bar_y=dist_bar_y, p=1, y_dist=dist_bar_y, distance=0.11)
significance_bar(pos_y=pos_y_start+1.3, pos_x=[0, 3], bar_y=dist_bar_y, p=3, y_dist=dist_bar_y, distance=0.11)
significance_bar(pos_y=pos_y_start+2.1, pos_x=[0, 4], bar_y=dist_bar_y, p=1, y_dist=dist_bar_y, distance=0.11)
significance_bar(pos_y=pos_y_start+3.5, pos_x=[0, 5], bar_y=dist_bar_y, p=3, y_dist=dist_bar_y, distance=0.11)

sb.boxplot(data=all_data, color="white", fliersize=0)
sb.swarmplot(data=all_data, color="black", size=4)

plt.ylabel("Energy distance", size=32)
plt.yticks(fontsize=28)
plt.xticks(fontsize=28, rotation=45)

plt.show()




