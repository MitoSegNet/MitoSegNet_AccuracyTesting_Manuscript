from skimage import img_as_bool, io, color
from skimage.morphology import skeletonize
from skan import summarise
import collections

# get branche measurements: branch length, total branch length, number of branches, curvature index
def get_branch_meas(path):

    read_lab_skel = img_as_bool(color.rgb2gray(io.imread(path)))
    lab_skel = skeletonize(read_lab_skel).astype("uint8")

    branch_data = summarise(lab_skel)

    curve_ind = []
    for bd, ed in zip(branch_data["branch-distance"], branch_data["euclidean-distance"]):

        if ed != 0.0:
            curve_ind.append((bd - ed) / ed)
        else:
            curve_ind.append(bd - ed)

    branch_data["curvature-index"] = curve_ind

    grouped_branch_data_mean = branch_data.groupby(["skeleton-id"], as_index=False).mean()
    grouped_branch_data_sum = branch_data.groupby(["skeleton-id"], as_index=False).sum()

    counter = collections.Counter(branch_data["skeleton-id"])

    n_branches = []
    for i in grouped_branch_data_mean["skeleton-id"]:
        n_branches.append(counter[i])

    branch_len = grouped_branch_data_mean["branch-distance"].tolist()
    tot_branch_len = grouped_branch_data_sum["branch-distance"].tolist()
    curv_ind = grouped_branch_data_mean["curvature-index"].tolist()

    return n_branches, branch_len, tot_branch_len, curv_ind