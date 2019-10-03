"""

Plot validation history of all training sessions during cross validation

"""

import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sb

path = "C:/Users/Christian/Desktop/Fourth_CV/Validation_data/Cross_Val"

csv_list = os.listdir(path)

new_table = pd.DataFrame(columns=csv_list)
new_table2 = pd.DataFrame(columns=csv_list)

# choose which parameter to plot
param = "loss"
#param = "dice_coefficient"

for table in csv_list:

    data = pd.read_csv(path + os.sep + table)

    new_table[table] = data["val_"+param].tolist()
    new_table2[table] = data[param].tolist()

average_val = new_table.mean(axis=1)
std_val = new_table.std(axis=1)

average_train = new_table2.mean(axis=1)
std_train = new_table2.std(axis=1)

sb.lineplot(x=list(range(1,16)), y=average_val.values, color="blue", linewidth=3)
sb.lineplot(x=list(range(1,16)), y=average_train.values, color="red", linewidth=3)

plt.fill_between(list(range(1,16)),average_val.values-std_val.values,average_val.values+std_val.values,alpha=.2, color="blue")
plt.fill_between(list(range(1,16)),average_train.values-std_train.values,average_train.values+std_train.values,alpha=.2, color="red")

plt.ylabel("Average "+param, fontsize=32)
plt.xlabel("Epochs", fontsize=32)
plt.tick_params(axis="x", labelsize=26)
plt.tick_params(axis="y", labelsize=26)

plt.margins(x=0)

plt.legend(["Validation", "Training"], loc="upper right", fontsize=26)
plt.show()