from mpi4py import MPI
from dolfinx.io import gmshio, XDMFFile

# Functions to read Mesh
def mpi_print(s, comm: MPI.Comm):
    print(f"Rank {comm.rank}: {s}")

def read_xdmf_data(xdmfPath: str, comm: MPI.Comm,gdim:int, name:str = "mesh"):
    from dolfinx.io import XDMFFile
    from dolfinx.mesh import GhostMode
    
    with XDMFFile(comm, xdmfPath, "r") as xdmf:
        mesh = xdmf.read_mesh(name=name, ghost_mode=GhostMode.none)
        mesh.topology.create_connectivity(gdim-1, gdim)

        ct = xdmf.read_meshtags(mesh, name=f"{name}_cells")
        ft = xdmf.read_meshtags(mesh, name=f"{name}_facets")
    return mesh, ct, ft

    
def create_mesh(comm: MPI.Comm, meshPath: str, xdmfPath: str, gdim: int,mode: str = "w", name: str = "mesh"):
    """Create a DOLFINx from a Gmsh model and output to file.

    Args:
        comm: MPI communicator top create the mesh on.
        meshPath: Gmsh .msh Path.
        xdmfPath: XDMF filename.
        gdim: Geometry dimension of the mesh.
        mode: XDMF file mode. "w" (write) or "a" (append).
        name: Name (identifier) of the mesh to add.
    """
    from dolfinx.io import XDMFFile
    meshWritten = False
    msh, ct, ft = gmshio.read_from_msh(meshPath, comm, gdim=gdim)
    ft.name = f"{msh.name}_facets"
    ct.name = f"{msh.name}_cells"
    with XDMFFile(msh.comm, xdmfPath, mode) as file:
        msh.topology.create_connectivity(gdim-1, gdim)

        file.write_mesh(msh)
        file.write_meshtags(
            ct, msh.geometry, geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{msh.name}']/Geometry"
        )
        file.write_meshtags(
            ft, msh.geometry, geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{msh.name}']/Geometry"
        )
        meshWritten = True

    return meshWritten
