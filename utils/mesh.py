# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os
from pathlib import Path
from typing import Optional

import torch

import utils.vector_ops as vecmath

######################################################################################
# Utility functions
######################################################################################


def _find_mat(materials, name):
    for mat in materials:
        if mat["name"] == name:
            return mat
    return materials[0]  # Materials 0 is the default


######################################################################################
# Create mesh object from objfile
######################################################################################


def load_obj(filename: os.PathLike):
    filename = Path(filename)

    # Read entire file
    with open(filename, "r") as f:
        lines = f.readlines()

    # Load materials
    # all_materials = [
    #     {
    #         'name' : '_default_mat',
    #         'bsdf' : 'pbr',
    #         'kd'   : texture.Texture2D(torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32, device='cuda')),
    #         'ks'   : texture.Texture2D(torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device='cuda'))
    #     }
    # ]
    # if mtl_override is None:
    #     for line in lines:
    #         if len(line.split()) == 0:
    #             continue
    #         if line.split()[0] == 'mtllib':
    #             all_materials += material.load_mtl(os.path.join(obj_path, line.split()[1]), clear_ks) # Read in entire material library
    # else:
    #     all_materials += material.load_mtl(mtl_override)

    # load vertices
    vertices, texcoords, normals = [], [], []
    for line in lines:
        if len(line.split()) == 0:
            continue

        prefix = line.split()[0].lower()
        if prefix == "v":
            vertices.append([float(v) for v in line.split()[1:]])
        elif prefix == "vt":
            val = [float(v) for v in line.split()[1:]]
            texcoords.append([val[0], 1.0 - val[1]])
        elif prefix == "vn":
            normals.append([float(v) for v in line.split()[1:]])

    # load faces
    activeMatIdx = None
    faces, tfaces, nfaces, mfaces = [], [], [], []
    for line in lines:
        if len(line.split()) == 0:
            continue

        prefix = line.split()[0].lower()
        if prefix == "usemtl":  # Track used materials
            # mat = _find_mat(all_materials, line.split()[1])
            # if not mat in used_materials:
            #     used_materials.append(mat)
            # activeMatIdx = used_materials.index(mat)
            pass
        elif prefix == "f":  # Parse face
            vs = line.split()[1:]
            nv = len(vs)
            vv = vs[0].split("/")
            v0 = int(vv[0]) - 1
            t0 = int(vv[1]) - 1 if len(vv) > 1 and vv[1] != "" else -1
            n0 = int(vv[2]) - 1 if len(vv) > 2 and vv[2] != "" else -1
            for i in range(nv - 2):  # Triangulate polygons
                vv = vs[i + 1].split("/")
                v1 = int(vv[0]) - 1
                t1 = int(vv[1]) - 1 if len(vv) > 1 and vv[1] != "" else -1
                n1 = int(vv[2]) - 1 if len(vv) > 2 and vv[2] != "" else -1
                vv = vs[i + 2].split("/")
                v2 = int(vv[0]) - 1
                t2 = int(vv[1]) - 1 if len(vv) > 1 and vv[1] != "" else -1
                n2 = int(vv[2]) - 1 if len(vv) > 2 and vv[2] != "" else -1
                mfaces.append(activeMatIdx)
                faces.append([v0, v1, v2])
                tfaces.append([t0, t1, t2])
                nfaces.append([n0, n1, n2])
    assert len(tfaces) == len(faces) and len(nfaces) == len(faces)

    # Create an "uber" material by combining all textures into a larger texture
    # if len(used_materials) > 1:
    #     uber_material, texcoords, tfaces = material.merge_materials(used_materials, texcoords, tfaces, mfaces)
    # else:
    #     uber_material = used_materials[0]

    vertices = torch.tensor(vertices, dtype=torch.float32, device="cuda")
    texcoords = torch.tensor(texcoords, dtype=torch.float32, device="cuda") if len(texcoords) > 0 else None
    normals = torch.tensor(normals, dtype=torch.float32, device="cuda") if len(normals) > 0 else None

    faces = torch.tensor(faces, dtype=torch.int64, device="cuda")
    tfaces = torch.tensor(tfaces, dtype=torch.int64, device="cuda") if texcoords is not None else None
    nfaces = torch.tensor(nfaces, dtype=torch.int64, device="cuda") if normals is not None else None

    return Mesh(vertices, faces, normals, nfaces, texcoords, tfaces)


######################################################################################
# Save mesh object to objfile
######################################################################################


def write_obj(
    filename: os.PathLike,
    v_pos: torch.Tensor,
    t_pos_idx: torch.Tensor,
    v_nrm: Optional[torch.Tensor] = None,
    t_nrm_idx: Optional[torch.Tensor] = None,
    v_tex: Optional[torch.Tensor] = None,
    t_tex_idx: Optional[torch.Tensor] = None,
):
    obj_file = Path(filename)
    with open(obj_file, "w") as f:
        f.write("g default\n")

        # v_pos = mesh.v_pos.detach().cpu().numpy() if mesh.v_pos is not None else None
        # v_nrm = mesh.v_nrm.detach().cpu().numpy() if mesh.v_nrm is not None else None
        # v_tex = mesh.v_tex.detach().cpu().numpy() if mesh.v_tex is not None else None

        # t_pos_idx = mesh.t_pos_idx.detach().cpu().numpy() if mesh.t_pos_idx is not None else None
        # t_nrm_idx = mesh.t_nrm_idx.detach().cpu().numpy() if mesh.t_nrm_idx is not None else None
        # t_tex_idx = mesh.t_tex_idx.detach().cpu().numpy() if mesh.t_tex_idx is not None else None

        for v in v_pos:
            f.write("v {} {} {} \n".format(v[0].item(), v[1].item(), v[2].item()))

        if v_tex is not None:
            assert len(t_pos_idx) == len(t_tex_idx)
            for v in v_tex:
                f.write("vt {} {} \n".format(v[0].item(), 1.0 - v[1]))

        if v_nrm is not None:
            assert len(t_pos_idx) == len(t_nrm_idx)
            for v in v_nrm:
                f.write("vn {} {} {}\n".format(v[0], v[1], v[2]))

        # faces
        f.write("s 1 \n")
        f.write("g pMesh1\n")

        # Write faces
        for i in range(len(t_pos_idx)):
            f.write("f ")
            for j in range(3):
                f.write(
                    " {}/{}/{}".format(
                        str(t_pos_idx[i][j].item() + 1),
                        "" if v_tex is None else str(t_tex_idx[i][j].item() + 1),
                        "" if v_nrm is None else str(t_nrm_idx[i][j].item() + 1),
                    )
                )
            f.write("\n")

    # if save_material:
    #     mtl_file = os.path.join(folder, "mesh.mtl")
    #     print("Writing material: ", mtl_file)
    #     material.save_mtl(mtl_file, mesh.material)


def auto_normals(imesh):
    i0 = imesh.t_pos_idx[:, 0].long()
    i1 = imesh.t_pos_idx[:, 1].long()
    i2 = imesh.t_pos_idx[:, 2].long()

    v0 = imesh.v_pos[i0, :]
    v1 = imesh.v_pos[i1, :]
    v2 = imesh.v_pos[i2, :]

    face_normals = torch.linalg.cross(v1 - v0, v2 - v0)

    # Splat face normals to vertices
    v_nrm = torch.zeros_like(imesh.v_pos)
    v_nrm.scatter_add_(0, i0[:, None].repeat(1, 3), face_normals)
    v_nrm.scatter_add_(0, i1[:, None].repeat(1, 3), face_normals)
    v_nrm.scatter_add_(0, i2[:, None].repeat(1, 3), face_normals)

    # Normalize, replace zero (degenerated) normals with some default value
    v_nrm = torch.where(
        vecmath.dot(v_nrm, v_nrm) > 1e-20,
        v_nrm,
        torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device="cuda"),
    )
    v_nrm = vecmath.safe_normalize(v_nrm)

    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(v_nrm))

    return Mesh(v_nrm=v_nrm, t_nrm_idx=imesh.t_pos_idx, base=imesh)


def compute_tangents(imesh):
    vn_idx = [None] * 3
    pos = [None] * 3
    tex = [None] * 3
    for i in range(0, 3):
        pos[i] = imesh.v_pos[imesh.t_pos_idx[:, i]]
        tex[i] = imesh.v_tex[imesh.t_tex_idx[:, i]]
        vn_idx[i] = imesh.t_nrm_idx[:, i]

    tangents = torch.zeros_like(imesh.v_nrm)

    # Compute tangent space for each triangle
    uve1 = tex[1] - tex[0]
    uve2 = tex[2] - tex[0]
    pe1 = pos[1] - pos[0]
    pe2 = pos[2] - pos[0]

    nom = pe1 * uve2[..., 1:2] - pe2 * uve1[..., 1:2]
    denom = uve1[..., 0:1] * uve2[..., 1:2] - uve1[..., 1:2] * uve2[..., 0:1]

    # Avoid division by zero for degenerated texture coordinates
    tang = nom / torch.where(denom > 0.0, torch.clamp(denom, min=1e-6), torch.clamp(denom, max=-1e-6))

    # Update all 3 vertices
    for i in range(0, 3):
        idx = vn_idx[i][:, None].repeat(1, 3)
        tangents.scatter_add_(0, idx.long(), tang)  # tangents[n_i] = tangents[n_i] + tang

    # Normalize and make sure tangent is perpendicular to normal
    tangents = vecmath.safe_normalize(tangents)
    tangents = vecmath.safe_normalize(tangents - vecmath.dot(tangents, imesh.v_nrm) * imesh.v_nrm)

    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(tangents))

    return Mesh(v_tng=tangents, t_tng_idx=imesh.t_nrm_idx, base=imesh)


class Mesh(object):
    def __init__(
        self,
        v_pos: torch.Tensor = None,
        t_pos_idx: torch.Tensor = None,
        v_nrm: Optional[torch.Tensor] = None,
        t_nrm_idx: Optional[torch.Tensor] = None,
        v_tex: Optional[torch.Tensor] = None,
        t_tex_idx: Optional[torch.Tensor] = None,
        v_tng: Optional[torch.Tensor] = None,
        t_tng_idx: Optional[torch.Tensor] = None,
        base=None,
    ):
        self.v_pos = v_pos
        self.v_nrm = v_nrm
        self.v_tex = v_tex
        self.v_tng = v_tng
        self.t_pos_idx = t_pos_idx.int() if t_pos_idx is not None else None
        self.t_nrm_idx = t_nrm_idx.int() if t_nrm_idx is not None else None
        self.t_tex_idx = t_tex_idx.int() if t_tex_idx is not None else None
        self.t_tng_idx = t_tng_idx.int() if t_tng_idx is not None else None

        if base is not None:
            self.copy_none(base)

    def copy_none(self, other):
        if self.v_pos is None:
            self.v_pos = other.v_pos
        if self.t_pos_idx is None:
            self.t_pos_idx = other.t_pos_idx
        if self.v_nrm is None:
            self.v_nrm = other.v_nrm
        if self.t_nrm_idx is None:
            self.t_nrm_idx = other.t_nrm_idx
        if self.v_tex is None:
            self.v_tex = other.v_tex
        if self.t_tex_idx is None:
            self.t_tex_idx = other.t_tex_idx
        if self.v_tng is None:
            self.v_tng = other.v_tng
        if self.t_tng_idx is None:
            self.t_tng_idx = other.t_tng_idx

    def clone(self):
        out = Mesh(base=self)
        if out.v_pos is not None:
            out.v_pos = out.v_pos.clone().detach()
        if out.t_pos_idx is not None:
            out.t_pos_idx = out.t_pos_idx.clone().detach()
        if out.v_nrm is not None:
            out.v_nrm = out.v_nrm.clone().detach()
        if out.t_nrm_idx is not None:
            out.t_nrm_idx = out.t_nrm_idx.clone().detach()
        if out.v_tex is not None:
            out.v_tex = out.v_tex.clone().detach()
        if out.t_tex_idx is not None:
            out.t_tex_idx = out.t_tex_idx.clone().detach()
        if out.v_tng is not None:
            out.v_tng = out.v_tng.clone().detach()
        if out.t_tng_idx is not None:
            out.t_tng_idx = out.t_tng_idx.clone().detach()
        return out

    def to(self, device):
        self.v_pos = self.v_pos.to(device) if self.v_pos is not None else None
        self.v_nrm = self.v_nrm.to(device) if self.v_nrm is not None else None
        self.v_tex = self.v_tex.to(device) if self.v_tex is not None else None
        self.v_tng = self.v_tng.to(device) if self.v_tng is not None else None
        self.t_pos_idx = self.t_pos_idx.to(device) if self.t_pos_idx is not None else None
        self.t_nrm_idx = self.t_nrm_idx.to(device) if self.t_nrm_idx is not None else None
        self.t_tex_idx = self.t_tex_idx.to(device) if self.t_tex_idx is not None else None
        self.t_tng_idx = self.t_tng_idx.to(device) if self.t_tng_idx is not None else None

        return self

    def cuda(self):
        self.v_pos = self.v_pos.cuda() if self.v_pos is not None else None
        self.v_nrm = self.v_nrm.cuda() if self.v_nrm is not None else None
        self.v_tex = self.v_tex.cuda() if self.v_tex is not None else None
        self.v_tng = self.v_tng.cuda() if self.v_tng is not None else None
        self.t_pos_idx = self.t_pos_idx.cuda() if self.t_pos_idx is not None else None
        self.t_nrm_idx = self.t_nrm_idx.cuda() if self.t_nrm_idx is not None else None
        self.t_tex_idx = self.t_tex_idx.cuda() if self.t_tex_idx is not None else None
        self.t_tng_idx = self.t_tng_idx.cuda() if self.t_tng_idx is not None else None

        return self

    @staticmethod
    def load_obj(filename: os.PathLike):
        return load_obj(filename)

    def write_obj(self, filename: os.PathLike):
        write_obj(
            filename,
            self.v_pos,
            self.t_pos_idx,
            self.v_nrm,
            self.t_nrm_idx,
            self.v_tex,
            self.t_tex_idx,
        )

    def aabb(mesh):
        """Compute the axis-aligned bounding box of the mesh.

        Args:
            mesh (Mesh): Mesh object

        Returns:
            (torch.Tensor, torch.Tensor): Min and max corners of the AABB
        """
        return torch.min(mesh.v_pos, dim=0).values, torch.max(mesh.v_pos, dim=0).values
