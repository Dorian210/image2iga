# %%
import numpy as np
import pickle as pkl
from volVIC.virtual_image_correlation_energy import compute_distance_field


u_field = np.load("../out_fitting/u_field.npy")

with open("../out_fitting/problem.pkl", "rb") as file:
    pb = pkl.load(file)

d0 = compute_distance_field(pb.image_energies, saved_data=pb.saved_data_0)
dn = compute_distance_field(pb.image_energies)


# %%
def compute_mean_abs_distance(pb, d_list):
    d_int = 0
    surf = 0
    for patch in range(pb.mesh.connectivity.nb_patchs):
        d = d_list[patch].ravel()
        wdetJs = pb.image_energies[patch].wdetJs
        d_int += np.sum(np.abs(d) * wdetJs)
        surf += np.sum(wdetJs)
    return d_int / surf


compute_mean_abs_distance(pb, d0), compute_mean_abs_distance(pb, dn)

# %%
