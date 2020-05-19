"""

Plot validation history of final training session to generate fully trained MitoSegNet model

"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

filename = "Final_MitoSegNet_656_training_log.csv"
path = "/" + filename

table = pd.read_csv(path)

#param = "dice_coefficient"
param = "loss"

p2 = sb.lineplot(x=list(range(1,21)), y=table[param], color="red", linewidth=3)
p3 = sb.lineplot(x=list(range(1,21)), y=table["val_"+param], color="blue", linewidth=3)

p3.set_ylabel(param, fontsize=32)
p3.set_xlabel("Epochs", fontsize=32)
p3.tick_params(axis="x", labelsize=26)
p3.tick_params(axis="y", labelsize=26)

plt.legend(('Training', 'Validation'), prop={"size": 26}, loc="lower right")
plt.margins(x=0)
plt.show()

