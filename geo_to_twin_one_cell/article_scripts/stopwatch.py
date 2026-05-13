# %%

jit_compiled = True
n = 10

print(f"Running with jit_compiled={jit_compiled} and n={n}")

import numba as nb

nb.config.DISABLE_JIT = not jit_compiled

from timeit import timeit

import numpy as np
import scipy.sparse as sps
from bsplyne import BSpline, MultiPatchBSplineConnectivity
from IGA_for_bsplyne import solve_sparse


from scipy.ndimage import gaussian_filter
import pickle as pkl
from PIL import Image, ImageSequence
from copy import deepcopy
from volVIC import Problem, Mesh, find_fg_bg

from IGA_for_bsplyne import IGAPatch, DirichletConstraintHandler, ProblemIGA


# %%


def make_geo():

    def ellipse(phi):  # x² - xy + y² = 3/2
        ellipse_x = np.sqrt(3 / 2) * np.cos(phi) - 1 / np.sqrt(2) * np.sin(phi)
        ellipse_y = np.sqrt(3 / 2) * np.cos(phi) + 1 / np.sqrt(2) * np.sin(phi)
        return np.array([ellipse_x, ellipse_y])

    xi = np.linspace(0, 1, 1000)
    phi = -np.pi / 3 + xi * np.pi / 3
    x_hat, y_hat = ellipse(phi)

    p = 2
    knot = np.array([0, 0, 0, 1, 1, 1])
    spline_curve = BSpline([p], [knot])
    spline_curve.orderElevation(None, [1])
    spline_curve.knotInsertion(None, [4])
    N = spline_curve.DN([xi], k=0)

    weights = np.hstack(([1e10], np.ones(N.shape[0] - 2), [1e10]))
    W = sps.diags(weights)

    x = solve_sparse(N.T @ W @ N, N.T @ W @ x_hat)
    y = solve_sparse(N.T @ W @ N, N.T @ W @ y_hat)

    l = 2
    r = 0.7 / 2

    degrees = [spline_curve.getDegrees()[0], 1, 1]
    knot_lin = np.array([0, 0, 1, 1])
    knots = [spline_curve.getKnots()[0], knot_lin, knot_lin]
    spline = BSpline(degrees, knots)

    X, Y, Z = np.zeros((3, *spline.getCtrlShape()))

    o = np.zeros_like(x)
    i = np.ones_like(x)
    X[:, 0, 0] = o
    Y[:, 0, 0] = o
    Z[:, 0, 0] = o
    X[:, 0, 1] = l * i
    Y[:, 0, 1] = l * i
    Z[:, 0, 1] = l * i
    X[:, 1, 0] = r * x
    Y[:, 1, 0] = r * y
    Z[:, 1, 0] = o
    X[:, 1, 1] = l * i
    Y[:, 1, 1] = l * i - r * y[::-1]
    Z[:, 1, 1] = l * i - r * x[::-1]

    ctrl_pts = np.array([X, Y, Z])

    ctrl_pts = spline.orderElevation(ctrl_pts, [0, 2, 2])

    ctrl_pts = spline.knotInsertion(ctrl_pts, [0, 3, 12])

    start = np.sqrt(2) * r / np.sqrt(3) + 1e-2
    greville = spline.bases[2].greville_abscissa()[1:-1]
    a, b = greville[0], greville[-1]
    coords = (l - 2 * start) * (greville - a) / (b - a) + start
    centers = coords[None] * np.ones((3, 1))
    disp = centers[:, None, None, :] - ctrl_pts[:, :, :, 1:-1]
    disp_needed = np.ones((3, 1, 1, 1)) * disp.sum(axis=0)[None] / 3
    ctrl_pts[:, :, :, 1:-1] += disp_needed
    ctrl_pts[1:, 0, :, :] = np.mean(ctrl_pts[1:, 0, :, :], axis=0)[None]
    ctrl_pts[:-1, -1, :, :] = np.mean(ctrl_pts[:-1, -1, :, :], axis=0)[None]

    X, Y, Z = ctrl_pts
    beam_pts = [[X, Y, Z], [X, Z, Y], [Y, X, Z], [Y, Z, X], [Z, X, Y], [Z, Y, X]]

    cell_pts = (
        [[x, y, z] for x, y, z in beam_pts]
        + [[-x, y, z] for x, y, z in beam_pts]
        + [[x, -y, z] for x, y, z in beam_pts]
        + [[-x, -y, z] for x, y, z in beam_pts]
        + [[x, y, -z] for x, y, z in beam_pts]
        + [[-x, y, -z] for x, y, z in beam_pts]
        + [[x, -y, -z] for x, y, z in beam_pts]
        + [[-x, -y, -z] for x, y, z in beam_pts]
    )

    separated_ctrl_pts = [np.array([x, y, z]) for x, y, z in cell_pts]
    connectivity = MultiPatchBSplineConnectivity.from_separated_ctrlPts(
        separated_ctrl_pts, eps=1e-5
    )
    splines = [spline] * len(separated_ctrl_pts)

    return splines, separated_ctrl_pts, connectivity


if jit_compiled:
    make_geo()

t_geo = timeit("make_geo()", globals=globals(), number=n) / n
print(f"Time to make geometry: {t_geo:.3f} seconds")

# %%


def fit_tomo():
    with open("../BCC_cell.pkl", "rb") as file:
        splines, separated_ctrl_pts, connectivity = pkl.load(file)

    mesh = Mesh(splines, separated_ctrl_pts, connectivity)

    voxel_size = 0.0108725
    mesh.unique_ctrl_pts /= voxel_size

    mesh.correct_orientation(axis=0, verbose=False)

    surf_mesh = mesh.extract_border()

    subset_mesh = surf_mesh.subset(np.arange(0, surf_mesh.connectivity.nb_patchs, 2))

    def load_tiff_stack(path):
        with Image.open(path) as img:
            return np.array([np.array(im) for im in ImageSequence.Iterator(img)])

    image = load_tiff_stack("../cropped_CT_scan.tiff")
    image = gaussian_filter(image, sigma=1)

    pb = Problem(
        deepcopy(subset_mesh),
        image,
        fg_bg_method="interp",
        width_dx=2.0,
        surf_dx=5.0,
        C1_mode="auto",
        disable_parallel=True,
        verbose=False,
    )

    pts = subset_mesh.unique_ctrl_pts
    (pts_to_lock,) = np.where(
        np.isclose(pts[0], pts[0].min())
        | np.isclose(pts[0], pts[0].max())
        | np.isclose(pts[1], pts[1].min())
        | np.isclose(pts[1], pts[1].max())
        | np.isclose(pts[2], pts[2].min())
        | np.isclose(pts[2], pts[2].max())
    )
    inds = np.hstack(
        (
            pts_to_lock + 0 * subset_mesh.connectivity.nb_unique_nodes,
            pts_to_lock + 1 * subset_mesh.connectivity.nb_unique_nodes,
            pts_to_lock + 2 * subset_mesh.connectivity.nb_unique_nodes,
        )
    )
    vals = np.zeros(inds.size)

    pb.constraints.add_eqs_from_inds_vals(inds, vals)
    pb.make_dirichlet()

    u_field, rho = pb.solve(
        eps=6.5e-2, max_iter=30, disable_parallel=True, verbose=False
    )

    u_vol_field = pb.propagate_displacement_to_volume_mesh(
        u_field, mesh, disable_parallel=True
    )

    pts = mesh.unique_ctrl_pts
    (pts_to_lock,) = np.where(
        np.isclose(pts[0], pts[0].min())
        | np.isclose(pts[0], pts[0].max())
        | np.isclose(pts[1], pts[1].min())
        | np.isclose(pts[1], pts[1].max())
        | np.isclose(pts[2], pts[2].min())
        | np.isclose(pts[2], pts[2].max())
    )
    u_vol_field[:, pts_to_lock] = 0.0

    vol_mesh = deepcopy(mesh)

    vol_mesh.unique_ctrl_pts += u_vol_field
    vol_mesh.unique_ctrl_pts *= voxel_size

    return vol_mesh


if jit_compiled:
    fit_tomo()

t_fit = timeit("fit_tomo()", globals=globals(), number=n) / n
print(f"Time to fit tomography: {t_fit:.3f} seconds")

# %%


def get_mechanical_response():
    with open("../BCC_cell_fitted.pkl", "rb") as file:
        splines, separated_ctrl_pts, connectivity = pkl.load(file)

    l = 2.0
    E, nu = 59e3, 0.33
    patches = [
        IGAPatch(spl, ctrl, E, nu) for spl, ctrl in zip(splines, separated_ctrl_pts)
    ]

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

    ref_inds = constraints.nb_dofs_init + np.arange(6)
    theta = np.array([0.0, 0.0, np.pi / 16])
    t = np.array([0.0, 0.0, -l / 8])
    theta_t = np.hstack((theta, t))
    constraints.add_eqs_from_inds_vals(ref_inds, theta_t)

    (bot_nodes,) = np.where(np.isclose(unique_ctrl_pts[2], -l))
    bot_inds = np.hstack(
        (
            bot_nodes + 0 * connectivity.nb_unique_nodes,
            bot_nodes + 1 * connectivity.nb_unique_nodes,
            bot_nodes + 2 * connectivity.nb_unique_nodes,
        )
    )
    constraints.add_eqs_from_inds_vals(bot_inds, np.zeros_like(bot_inds))

    dirichlet = constraints.create_dirichlet()

    pb = ProblemIGA(patches, connectivity, dirichlet)

    u_field = pb.solve()

    return u_field


if jit_compiled:
    get_mechanical_response()

t_meca = timeit("get_mechanical_response()", globals=globals(), number=n) / n
print(f"Time to get mechanical response: {t_meca:.3f} seconds")
# %%
