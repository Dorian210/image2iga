# %%
import numpy as np
import pickle as pkl
from IGA_for_bsplyne import (
    IGAPatch,
    DirichletConstraintHandler,
    ProblemIGA,
    solve_sparse,
)

with open("BCC_cell_fitted.pkl", "rb") as file:
    splines, separated_ctrl_pts, connectivity = pkl.load(file)

l = 2.0
E, nu = 59e3, 0.33
patches = [IGAPatch(spl, ctrl, E, nu) for spl, ctrl in zip(splines, separated_ctrl_pts)]


constraints = DirichletConstraintHandler(3 * connectivity.nb_unique_nodes)

unique_ctrl_pts = connectivity.pack(connectivity.agglomerate(separated_ctrl_pts))
(top_nodes,) = np.where(np.isclose(unique_ctrl_pts[2], l))
top_inds = np.hstack(
    (
        top_nodes + 0 * connectivity.nb_unique_nodes,
        top_nodes + 1 * connectivity.nb_unique_nodes,
        top_nodes + 2 * connectivity.nb_unique_nodes,
    )
)
top_pos = unique_ctrl_pts[:, top_nodes]
ref_point = np.array([0.0, 0.0, l])
constraints.add_rigid_body_constraint(ref_point, top_inds, top_pos)

(bot_nodes,) = np.where(np.isclose(unique_ctrl_pts[2], -l))
bot_inds = np.hstack(
    (
        bot_nodes + 0 * connectivity.nb_unique_nodes,
        bot_nodes + 1 * connectivity.nb_unique_nodes,
        bot_nodes + 2 * connectivity.nb_unique_nodes,
    )
)
constraints.add_eqs_from_inds_vals(bot_inds, np.zeros_like(bot_inds))

C, _ = constraints.make_C_k()
C_ref = C[constraints.nb_dofs_init :, :]
C_u = C[: constraints.nb_dofs_init, :]

ref_inds = constraints.nb_dofs_init + np.arange(6)
theta = np.array([0.0, 0.0, np.pi / 16])
t = np.array([0.0, 0.0, -l / 8])
theta_t = np.hstack((theta, t))
constraints.add_eqs_from_inds_vals(ref_inds, theta_t)

dirichlet = constraints.create_dirichlet()

pb = ProblemIGA(patches, connectivity, dirichlet)

K, F = pb.lhs_rhs(verbose=True)

# Apply Dirichlet boundary conditions
lhs, rhs = pb.apply_dirichlet(K, F, verbose=True)

# Solve the system
dof = pb.solve_from_lhs_rhs(lhs, rhs, verbose=True)

# Cancel Dirichlet subspace
u = pb.dirichlet.u(dof)

res = F - K @ u
lamb = solve_sparse(C_ref @ C_ref.T, C_ref @ C_u.T @ res)
react_moments = lamb[:3]
react_forces = lamb[3:]

print(f"Top reaction moments : {react_moments}")
print(f"Top reaction forces : {react_forces}")

u_field = u.reshape((3, -1))

# u_field = pb.solve()

pb.save_paraview(u_field, "out_simu", "results", n_eval_per_elem=10)

with open("out_simu/saved_values.pkl", "wb") as file:
    pkl.dump((l, ref_point, t, theta), file)

# %%
