import numpy as np
import stl
from scipy.spatial import Delaunay


class STLMesh:
    def __init__(
        self,
        vertices=None,
        faces=None,
        area=None,
        planar=None,
        normal_dir=None,
    ):
        self.vertices = vertices
        self.faces = faces
        self.area = area
        self.normal_dir = normal_dir
        self.planar = planar
        self.status = {}
        self.update()

    def update_status(self):
        if self.vertices is None:
            self.status["vertices"] = False
        elif self.vertices.shape[1] == 3 and len(self.vertices.shape) == 2:
            self.status["vertices"] = True
        else:
            self.status["vertices"] = False
        if self.faces is None:
            self.status["faces"] = False
        elif self.vertices.shape[1] == 3 and len(self.faces.shape) == 2:
            self.status["faces"] = True
        else:
            self.status["faces"] = False
        if self.area is None:
            self.status["area"] = False
        elif isinstance(self.area, float):
            self.status["area"] = True
        else:
            self.status["area"] = False
        if self.normal_dir is None:
            self.status["normal_dir"] = False
        elif isinstance(self.normal_dir, int):
            self.status["normal_dir"] = True
        else:
            self.status["normal_dir"] = False
        if self.planar is None:
            self.status["planar"] = False
        elif isinstance(self.planar, bool):
            self.status["planar"] = True
        else:
            self.status["planar"] = False

    def update(self):
        self.update_status()
        if (
            self.status["vertices"]
            and not self.status["normal_dir"]
            and self.status["planar"]
        ):
            print("\t\tUpdating normal_dir to")
            for i in range(3):
                A = np.where(self.vertices[:, i] == self.vertices[0, i])[0]
                if len(A) == self.vertices.shape[0]:
                    self.normal_dir = i
                    self.status["normal_dir"] = True
                    print(f"\t\t\t{i}")
                    break
        if (
            self.status["vertices"]
            and not self.status["faces"]
            and self.status["planar"]
        ):
            print("\t\tUpdating faces")
            points = np.zeros((self.vertices.shape[0], 2))
            count = 0
            for i in range(3):
                if not i == self.normal_dir:
                    points[:, count] = self.vertices[:, i]
                    count += 1
            tri = Delaunay(points)
            self.faces = np.array(tri.simplices)
            self.status["faces"] = True

        if self.status["vertices"] and self.status["faces"]:
            print("\t\tUpdating area")
            self.calc_area()
            self.status["area"] = True

    def to_stl(self):
        stlObj = stl.mesh.Mesh(
            np.zeros(self.faces.shape[0], dtype=stl.mesh.Mesh.dtype)
        )
        for i, f in enumerate(self.faces):
            for j in range(3):
                stlObj.vectors[i][j] = self.vertices[f[j], :]
        return stlObj

    def from_stl(self, stlObj):
        self.vertices = np.unique(np.reshape(stlObj.vectors, (-1, 3)), axis=0)
        self.faces = np.zeros((stlObj.vectors.shape[0], 3), dtype=int)
        for i in range(stlObj.vectors.shape[0]):
            for j in range(stlObj.vectors.shape[1]):
                ind = np.argwhere(
                    np.linalg.norm(
                        self.vertices - stlObj.vectors[i, j], axis=1
                    )
                    == 0
                )[0][0]
                self.faces[i, j] = ind

    def calc_area(self):
        stlObj = self.to_stl()
        stlObj.update_areas()
        self.area = np.sum(stlObj.areas)

    def from_mesh_list(self, mesh_list):
        if len(mesh_list) > 1:
            self.planar = False
            self.update_status()

        offset = int(0)
        self.area = 0
        for patch_mesh in mesh_list:
            patch_mesh.faces += offset
            offset += len(patch_mesh.vertices)
            if self.vertices is None:
                self.vertices = patch_mesh.vertices
            else:
                self.vertices = np.vstack((self.vertices, patch_mesh.vertices))
            if self.faces is None:
                self.faces = patch_mesh.faces
            else:
                self.faces = np.vstack((self.faces, patch_mesh.faces))
            self.area += patch_mesh.area
        self.update()

    def rotate(self, theta=0, normal_dir=1):
        stlObj = self.to_stl()
        normal = [0, 0, 0]
        normal[normal_dir] = 1
        stlObj.rotate(normal, theta)
        return self.from_stl(stlObj)

    def translate(self, vector=np.array([0, 0, 0])):
        self.vertices += vector

    def save(self, filename):
        stlObj = self.to_stl()
        stlObj.save(f"{filename}", mode=stl.Mode.ASCII)
