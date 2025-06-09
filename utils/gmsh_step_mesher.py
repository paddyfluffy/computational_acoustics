import gmsh
import pickle
import os
import numpy as np
from mpi4py import MPI
from dolfinx.io.gmshio import model_to_mesh
from utils.mesh_utils import create_mesh

def save_pickle(data, path):
    """Save a Python object to a pickle file."""
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved pickle to {path}")

def load_pickle(path):
    """Load a Python object from a pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)

def get_surface_normal(surface_tag):
    """Get the surface normal using Gmsh's geometric model (not mesh)."""
    try:
        uv = [0.5, 0.5]  # Parametric midpoint
        normal = gmsh.model.getNormal(surface_tag, uv)
        norm_array = np.array(normal)
        norm_array /= np.linalg.norm(norm_array)
        return norm_array
    except Exception as e:
        print(f"Warning: Could not get normal for surface {surface_tag}: {e}")
        return None

def classify_orientation(normal):
    """Classify surface based on its normal vector."""
    if normal is None:
        return "unknown"
    if np.isclose(np.abs(normal[2]), 1.0, atol=0.1):
        return "horizontal"
    elif np.isclose(np.abs(normal[2]), 0.0, atol=0.1):
        return "vertical"
    else:
        return "angled"


def load_step_and_create_individual_surface_groups(step_file, mesh_size_meters=1.0e-3, show_meshing_info=False, data_pkl_path=None):

    terminal = 1 if show_meshing_info else 0
    step_dir = os.path.dirname(step_file)
    step_base_name = os.path.splitext(os.path.basename(step_file))[0]

    XDMF_FILE = os.path.join(step_dir, f"{step_base_name}.xdmf")
    MSH_FILE = os.path.join(step_dir, f"{step_base_name}.msh")
    DATA_PKL = os.path.join(step_dir, f"{step_base_name}_data.pkl") if not data_pkl_path else data_pkl_path

    print(f"XDMF_FILE: {XDMF_FILE}")
    print(f"MSH_FILE: {MSH_FILE}")

    gmsh.initialize()
    gmsh.option.setString("Geometry.OCCTargetUnit", "M")
    gmsh.option.setNumber("General.Terminal", terminal)  # Disable meshing info if not needed
    gmsh.model.add("step_import")
    gmsh.model.occ.importShapes(step_file)
    gmsh.model.occ.synchronize()

    gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size_meters)

    volume_to_surface_phys_ids = {}
    surface_orientations = {}

    volumes = gmsh.model.getEntities(dim=3)
    print(f"Found {len(volumes)} volumes.")

    for (dim, vol_tag) in volumes:
        # Volume physical group
        gmsh.model.addPhysicalGroup(dim, [vol_tag], vol_tag)
        gmsh.model.setPhysicalName(dim, vol_tag, f"Volume_{vol_tag}")

        boundaries = gmsh.model.getBoundary([(dim, vol_tag)], oriented=False, recursive=False)
        surface_phys_ids = []

        for (surf_dim, surf_tag) in boundaries:
            phys_tag = 100000 + surf_tag  # Avoid clashes
            gmsh.model.addPhysicalGroup(surf_dim, [surf_tag], phys_tag)
            gmsh.model.setPhysicalName(surf_dim, phys_tag, f"Surface_{surf_tag}_of_Volume_{vol_tag}")
            surface_phys_ids.append(phys_tag)

            # Get and classify normal
            normal = get_surface_normal(surf_tag)
            orientation = classify_orientation(normal)

            surface_orientations[phys_tag] = {
                "normal": normal.tolist() if normal is not None else None,
                "orientation": orientation
            }

        volume_to_surface_phys_ids[vol_tag] = surface_phys_ids

    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(3)
    gmsh.write(MSH_FILE)
    gmsh.finalize()

    meshWritten = create_mesh(
        MPI.COMM_SELF,
        meshPath=MSH_FILE,
        xdmfPath=XDMF_FILE,
        gdim=3,
        mode="w",
        name="mesh"
    )

    if meshWritten:
        print(f"Mesh written to {XDMF_FILE}")
    else:
        print("Error writing mesh")

    data_dict = {
        "volume": volume_to_surface_phys_ids,
        "orientation": surface_orientations,
        "xdmf_path": XDMF_FILE,
    }

    save_pickle(data_dict, DATA_PKL)

    return DATA_PKL


if __name__ == "__main__":
    step_file = "/home/acoustics/meshes/4_5_2p5.step"
    from pathlib import Path

    step_path = Path("/home/acoustics/meshes/4_5_2p5.step")
    base = step_path.with_suffix("")
    data_pkl_path = load_step_and_create_individual_surface_groups(step_file, 0.5, True)

    data_dict = load_pickle(data_pkl_path)
    volume_dict = data_dict["volume"]
    orientation_dict = data_dict["orientation"]

    print("\nVolume to Surface Physical Groups:")
    for vol, surfaces in volume_dict.items():
        print(f"Volume {vol}: Surfaces {surfaces}")

    print("\nSurface Orientations:")
    for surf_id, data in orientation_dict.items():
        print(f"Surface {surf_id}: Orientation = {data['orientation']}, Normal = {data['normal']}")