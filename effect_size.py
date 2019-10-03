import numpy as np

# pooled standard deviation for calculation of effect size (cohen's d)
def cohens_d(data1, data2):

    p_std = np.sqrt(((len(data1)-1)*np.var(data1)+(len(data2)-1)*np.var(data2))/(len(data1)+len(data2)-2))
    cohens_d = np.abs(np.average(data1) - np.average(data2)) / p_std

    return cohens_d