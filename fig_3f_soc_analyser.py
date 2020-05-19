"""

Analyse tables generated by soc_generate_results.py

Statistical analysis

Create box plot

"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.stats import normaltest, kruskal
from scikit_posthocs import posthoc_dunn
from Plot_Significance import significance_bar
from effect_size import cohens_d

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)

path = ""

file_list = ['MitoSegNet_analysed_data.csv', 'Finetuned Fiji U-Net_analysed_data.csv', 'Ilastik_analysed_data.csv',
             'Gaussian_analysed_data.csv', 'Hessian_analysed_data.csv', 'Laplacian_analysed_data.csv']

seg_list = ["MitoSegNet", "Finetuned\nFiji U-Net", "Ilastik", "Gaussian", "Hessian", "Laplacian"]

all_data = pd.DataFrame(columns=seg_list)

for file, method_name in zip(file_list, all_data):

    if ".csv" in file:

        sheet =  pd.read_csv(path + "/" + file)

        # removing first column
        sheet.drop(sheet.columns[[0]], axis=1, inplace=True)

       # possibility to only read in one descriptor instead of all descriptors
        l = sheet.values.tolist()
        #l = sheet["area"].values.tolist()

        # flatten list of lists
        flat_list = [item for sublist in l for item in sublist]

        all_data[method_name] = flat_list
        #all_data[method_name] = l


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
#dt.to_excel("soc_posthoc.xlsx")

# check effect sizes of segmentations against msn segmentation
for seg in seg_list:
    if seg != "MitoSegNet":
        print(seg, cohens_d(all_data[seg], all_data["MitoSegNet"]))

bar_y = 0.03
significance_bar(pos_y=2.1, pos_x=[0, 2], bar_y=bar_y, p=3, y_dist=bar_y, distance=0.1)
significance_bar(pos_y=2.2, pos_x=[0, 3], bar_y=bar_y, p=3, y_dist=bar_y, distance=0.1)
significance_bar(pos_y=2.3, pos_x=[0, 4], bar_y=bar_y, p=1, y_dist=bar_y, distance=0.1)
significance_bar(pos_y=2.4, pos_x=[0, 5], bar_y=bar_y, p=1, y_dist=bar_y, distance=0.1)


n = sb.swarmplot(data=all_data, color="black", size=5)
sb.boxplot(data=all_data, color="white", fliersize=0)

n.set_ylabel("Average fold deviation", fontsize=32)
n.tick_params(axis="x", labelsize=34, rotation=45)
n.tick_params(axis="y", labelsize=28)

plt.show()

