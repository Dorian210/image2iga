# %%
import numpy as np
import scipy.sparse as sps
import pickle as pkl
import pyvista as pv
import meshio
import tempfile
import os
from tqdm import trange
from CGAL.CGAL_Polyhedron_3 import Polyhedron_3
from CGAL.CGAL_Mesh_3 import (
    Polyhedral_mesh_domain_3,
    Mesh_3_parameters,
    Default_mesh_criteria,
    make_mesh_3,
)
from scipy.sparse.linalg import minres

from IGA_for_bsplyne import Dirichlet


l = 2.0
E, nu = 59e3, 0.33
surf_n_nodes_per_b_spline_elem = 10
vol_mesh_surf_h = l / 20
vol_mesh_bulk_h = 2 * vol_mesh_surf_h


def make_surf_mesh(
    splines, separated_ctrl_pts, connectivity, n_nodes_per_b_spline_elem
):
    bd_connectivity, bd_splines, full_to_bd = connectivity.extract_exterior_borders(
        splines
    )
    unique_ctrl_pts = connectivity.pack(connectivity.agglomerate(separated_ctrl_pts))
    bd_unique_ctrl_pts = unique_ctrl_pts[..., full_to_bd]
    bd_separated_ctrl_pts = bd_connectivity.separate(
        bd_connectivity.unpack(bd_unique_ctrl_pts)
    )

    meshes = []
    for patch in trange(bd_connectivity.nb_patchs, desc="IGA patchs to surface meshes"):
        spline = bd_splines[patch]
        ctrl_pts = bd_separated_ctrl_pts[patch]
        XI = spline.linspace(n_nodes_per_b_spline_elem)
        points = spline(ctrl_pts, XI)
        inds = np.arange(np.prod(points.shape[1:])).reshape(points.shape[1:])
        A = inds[:-1, :-1].ravel()
        B = inds[1:, :-1].ravel()
        C = inds[1:, 1:].ravel()
        D = inds[:-1, 1:].ravel()
        triangle = np.vstack(
            (np.stack((A, B, C), axis=-1), np.stack((C, D, A), axis=-1))
        )
        mesh = meshio.Mesh(points.reshape((3, -1)).T, {"triangle": triangle})
        meshes.append(mesh)

    mesh_pv = pv.merge([pv.from_meshio(m) for m in meshes])
    surf_mesh = pv.to_meshio(mesh_pv)

    points = surf_mesh.points
    cells = surf_mesh.cells_dict["triangle"]
    A, B, C = points[cells].transpose((1, 0, 2))
    area = np.linalg.norm(np.cross(B - A, C - A), axis=1) / 2.0
    problematic = area < 1e-10
    cells = cells[~problematic]
    surf_mesh = meshio.Mesh(points, {"triangle": cells})

    return surf_mesh


def mesh_from_surface(meshio_surf, facet_size, cell_size, facet_angle=30):

    offSurfFile = tempfile.NamedTemporaryFile(suffix=".off", delete=False).name
    verts, faces = meshio_surf.points, meshio_surf.cells_dict["triangle"]
    with open(offSurfFile, "wb") as f:
        f.write(f"OFF\n{len(verts)} {len(faces)} 0\n\n".encode())
        # loop over nodes
        fmt_v = b"%f %f %f\n"
        for i in range(verts.shape[0]):
            f.write(fmt_v % (verts[i, 0], verts[i, 1], verts[i, 2]))
        # loop over triangles
        fmt_f = b"3 %d %d %d\n"
        for i in range(faces.shape[0]):
            f.write(fmt_f % (faces[i, 0], faces[i, 1], faces[i, 2]))
        f.write(b" ")
    print("off file created")

    # Create input polyhedron as an offset file
    polyhedron = Polyhedron_3(offSurfFile)
    os.remove(offSurfFile)
    print("polyhedron created")
    print(f"polyèdre fermé : {polyhedron.is_closed()}")

    # Create domain
    domain = Polyhedral_mesh_domain_3(polyhedron)
    params = Mesh_3_parameters()

    # // Mesh criteria
    # Mesh_criteria criteria(facet_angle=30, facet_size=0.1,
    #                        facet_distance=0.025,
    #                        cell_radius_edge_ratio=2, cell_size=0.1)
    # Mesh criteria (no cell_size set)
    criteria = Default_mesh_criteria()
    criteria.facet_angle(facet_angle).facet_size(facet_size).cell_size(cell_size)
    # Mesh generation
    c3t3 = make_mesh_3(domain, criteria, params)
    print("mesh created")

    meshFile = tempfile.NamedTemporaryFile(suffix=".mesh", delete=False).name
    c3t3.output_to_medit(meshFile)

    feMesh = meshio.read(meshFile)
    os.remove(meshFile)

    return feMesh


def clean_unstable_tetras(mesh):
    cells = mesh.cells_dict["tetra"]
    face_conn = np.array(
        [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
    )  # which node to put in each face : node to face
    faces = np.sort(cells[:, face_conn], axis=2)
    face_ids, inverse = np.unique(
        faces.reshape((-1, 3)), axis=0, return_inverse=True
    )  # unique[inverse].reshape((-1, 4, 3)) == faces
    faces_indices = inverse.reshape((-1, 4))
    n_cells = cells.shape[0]
    n_faces = face_ids.shape[0]

    # Creating an inverse table : for each face, what are the adjacent tetrahedra ?
    # Since each face has at most 2 tetrahedra, we can do this quickly:
    face_to_cells = -np.ones((n_faces, 2), dtype="int")
    cells_ids = np.repeat(np.arange(n_cells), 4)
    face_to_cells[faces_indices.flat, 0] = cells_ids
    face_to_cells[faces_indices.flat[::-1], 1] = cells_ids[::-1]

    # Selecting the shared faces
    shared = face_to_cells[:, 0] != face_to_cells[:, 1]
    c1, c2 = face_to_cells[shared, 0], face_to_cells[shared, 1]

    # Creating a graph where the vertices are the tetrahedra
    adj = sps.coo_matrix((np.ones(len(c1)), (c1, c2)), shape=(n_cells, n_cells))

    # Extracting the connected components (by faces !)
    n_components, labels = sps.csgraph.connected_components(adj, directed=False)
    print(
        f"Dual mesh contains {n_components} components. Keeping only the largest in the primal mesh."
    )

    # Keeping the biggest component
    largest_label = np.argmax(np.bincount(labels))
    final_cells = cells[labels == largest_label]

    cells_dict = mesh.cells_dict
    cells_dict["tetra"] = final_cells
    new_mesh = meshio.Mesh(mesh.points, cells_dict)
    return new_mesh


def keep_only_tetra(mesh):
    n_pts = mesh.points.shape[0]
    mask = np.zeros(n_pts, dtype="bool")
    mask[mesh.cells_dict["tetra"].ravel()] = True
    inds = -np.ones(n_pts, dtype="int")
    inds[mask] = np.arange(np.count_nonzero(mask))
    points = mesh.points[mask]
    cells_dict = {"tetra": inds[mesh.cells_dict["tetra"]]}
    new_mesh = meshio.Mesh(points, cells_dict)
    return new_mesh


with open("../BCC_cell_fitted.pkl", "rb") as file:
    splines, separated_ctrl_pts, connectivity = pkl.load(file)
surf_mesh = make_surf_mesh(
    splines, separated_ctrl_pts, connectivity, surf_n_nodes_per_b_spline_elem
)
print("Surface extraction completed")
vol_mesh = mesh_from_surface(
    surf_mesh, facet_size=vol_mesh_surf_h, cell_size=vol_mesh_bulk_h
)
print("Volumetric meshing done")
vol_mesh = clean_unstable_tetras(vol_mesh)
vol_mesh = keep_only_tetra(vol_mesh)
print("Mesh cleaned")
vol_mesh.write("tmp.vtk")
print("Finite elements mesh saved")

# %%


def make_K(mesh, E, nu):
    points = mesh.points
    cells = mesh.cells_dict["tetra"]
    n_nodes = points.shape[0]
    n_cells = cells.shape[0]
    I = np.eye(3)

    # 1. Geometry and Volumes
    p = points[cells]
    M = np.ones((n_cells, 4, 4))
    M[:, :, 1:] = p
    vols = np.linalg.det(M) / 6.0
    G = np.linalg.inv(M)[:, 1:, :]  # Shape: (n_cells, 3, 4)
    del p, M
    print("Tetra mapping inverted")

    # 2. Construction of eps_op such that eps = eps_op : U
    # eps_op is of dim (n_cells, 3, 3, 4, 3)
    eps_op = 0.5 * (
        G[:, :, None, :, None] * I[None, None, :, None, :]
        + G[:, None, :, :, None] * I[None, :, None, None, :]
    )
    del G
    print("Epsilon operator constructed")

    # 3. Construction of the 4th order Hooke's tensor C (dim 3x3x3x3)
    lmbda = (E * nu) / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    C = (
        2 * mu * I[:, None, :, None] * I[None, :, None, :]
        + lmbda * I[:, :, None, None] * I[None, None, :, :]
    )
    print("Hooke's tensor created")

    # 4. Construction of sigma_op such that sigma = sigma_op : U
    # sigma_op is of dim (n_cells, 3, 3, 4, 3)
    sigma_op = np.einsum("jkno,inolm->ijklm", C, eps_op)
    print("Sigma operator constructed")

    # 5. Elementary stiffness Ke of dim (n_cells, 4, 3, 4, 3)
    Ke = (
        np.einsum("inojk,inolm->ijklm", sigma_op, eps_op)
        * vols[:, None, None, None, None]
    )
    del eps_op, sigma_op, vols
    print("Elementary stiffness computed")

    # 6. Assembly of K formated as [ux0, ux1, ..., uy0, uy1, ..., uz0, uz1, ...]
    cells_dofs = np.stack(
        (cells, cells + n_nodes, cells + 2 * n_nodes), axis=-1
    )  # shape (n_cells, 4, 3)
    rows = np.broadcast_to(
        cells_dofs[:, :, :, None, None], (n_cells, 4, 3, 4, 3)
    ).ravel()
    cols = np.broadcast_to(
        cells_dofs[:, None, None, :, :], (n_cells, 4, 3, 4, 3)
    ).ravel()
    data = Ke.ravel()
    K = sps.coo_matrix((data, (rows, cols)), shape=(3 * n_nodes, 3 * n_nodes))
    print("Global stiffness assembled")

    return K


vol_mesh = meshio.read("mesh_for_finite_elements.vtk")
K = make_K(vol_mesh, E, nu)

ndof = K.shape[0]
F = np.zeros(ndof)

# %%

n_nodes = vol_mesh.points.shape[0]

from IGA_for_bsplyne import Dirichlet

dirichlet = Dirichlet.eye(ndof + 6)

K_augmented = sps.bmat([[K, None], [None, sps.coo_matrix((6, 6))]])
F_augmented = np.hstack((F, np.zeros(6)))

(bot_nodes,) = np.where(vol_mesh.points[:, 2] <= -l)
bot_inds = np.hstack(
    (bot_nodes + 0 * n_nodes, bot_nodes + 1 * n_nodes, bot_nodes + 2 * n_nodes)
)
dirichlet.set_u_inds_vals(bot_inds, np.zeros(bot_inds.size))

(top_nodes,) = np.where(vol_mesh.points[:, 2] >= l)
ref_point = np.array([0.0, 0.0, l])

slaves = np.hstack(
    (top_nodes + 0 * n_nodes, top_nodes + 1 * n_nodes, top_nodes + 2 * n_nodes)
)

theta_t_ids = np.arange(ndof, ndof + 6)
n_top = top_nodes.size
refs_x = np.broadcast_to([theta_t_ids[1], theta_t_ids[2], theta_t_ids[3]], (n_top, 3))
refs_y = np.broadcast_to([theta_t_ids[0], theta_t_ids[2], theta_t_ids[4]], (n_top, 3))
refs_z = np.broadcast_to([theta_t_ids[0], theta_t_ids[1], theta_t_ids[5]], (n_top, 3))
references = np.vstack((refs_x, refs_y, refs_z))

top_points = vol_mesh.points[top_nodes].T  # (3, n)
x, y, z = top_points - ref_point[:, None]
ones = np.ones_like(x)
coefs_x = np.stack((z, -y, ones), axis=-1)
coefs_y = np.stack((-z, x, ones), axis=-1)
coefs_z = np.stack((y, -x, ones), axis=-1)
coefs = np.vstack((coefs_x, coefs_y, coefs_z))

dirichlet.slave_reference_linear_relation(slaves, references, coefs)

K_d = dirichlet.C.T @ K_augmented @ dirichlet.C
F_d = dirichlet.C.T @ (F_augmented - K_augmented @ dirichlet.k)

theta = np.array([0.0, 0.0, np.pi / 16])
t = np.array([0.0, 0.0, -l / 8])
theta_t = np.hstack((theta, t))

C_lagrangien = sps.hstack([sps.coo_matrix((6, F_d.size - 6)), sps.eye(6)])

scale = np.mean(K_d.diagonal())
K_lag = sps.bmat([[K_d, scale * C_lagrangien.T], [scale * C_lagrangien, None]])
F_lag = np.hstack((F_d, scale * theta_t))

dof_d_lamb, info = minres(K_lag, F_lag, rtol=1e-14, show=True)

dof_d = dof_d_lamb[:-6]
lamb_physique = dof_d_lamb[-6:] * scale

react_moments = lamb_physique[:3]
react_forces = lamb_physique[3:]

print(f"Top reaction moments : {react_moments}")
print(f"Top reaction forces : {react_forces}")

u_theta_t = dirichlet.C @ dof_d + dirichlet.k

u = u_theta_t[:-6]
theta_t_retrieved = u_theta_t[-6:]

theta_retrieved = theta_t_retrieved[:3]
t_retrieved = theta_t_retrieved[3:]

print(f"Retrieved value of theta imposed : {theta_retrieved}")
print(f"Retrieved value of t imposed : {t_retrieved}")


# %%
def compute_von_mises(mesh, u, E, nu):
    points = mesh.points
    cells = mesh.cells_dict["tetra"]
    n_cells = cells.shape[0]
    I = np.eye(3)

    # 1. u at elements
    u_el = u[cells]  # Shape (n_cells, 4, 3)
    print("Displacement at elements extracted")

    # 2. Geometry (Gradient operator)
    p = points[cells]
    M = np.ones((n_cells, 4, 4))
    M[:, :, 1:] = p
    G = np.linalg.inv(M)[:, 1:, :]  # Shape (n_cells, 3, 4)
    print("Tetra mapping inverted")

    # 3. Epsilon (strain) : eps = 0.5 * (grad U + grad U^T)
    grad_u_T = G @ u_el  # Shape (n_cells, 3, 3)
    eps = 0.5 * (grad_u_T.transpose((0, 2, 1)) + grad_u_T)
    print("Strain computed")

    # 4. Hooke's law for Sigma (stress)
    lmbda = (E * nu) / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    trace_eps = np.trace(eps, axis1=1, axis2=2)
    # sigma = 2*mu*eps + lambda*tr(eps)*I
    sigma = 2 * mu * eps + lmbda * trace_eps[:, None, None] * I[None, :, :]
    print("Stress computed")

    # 5. Von Mises
    # s = sigma - 1/3 * tr(sigma) * I
    trace_sigma = np.trace(sigma, axis1=1, axis2=2)
    s = sigma - (1 / 3.0) * trace_sigma[:, None, None] * I[None, :, :]
    # von_mises = sqrt(3/2 * s : s)
    von_mises = np.sqrt(1.5 * (s * s).sum(axis=2).sum(axis=1))
    print("Von Mises calculated")

    return von_mises


U = u.reshape((3, -1)).T
vm = compute_von_mises(vol_mesh, U, E, nu)

res_mesh = meshio.Mesh(
    vol_mesh.points,
    vol_mesh.cells_dict,
    point_data={"U": U},
    cell_data={"von_mises": [vm]},
)

res_mesh.write("results_fe.vtk")

# %%
