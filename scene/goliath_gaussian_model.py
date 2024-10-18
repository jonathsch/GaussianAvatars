#
# Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual
# property and proprietary rights in and to this software and related documentation.
# Any commercial use, reproduction, disclosure or distribution of this software and
# related documentation without an express license agreement from Toyota Motor Europe NV/SA
# is strictly prohibited.
#

from pathlib import Path
import numpy as np
import torch
from flame_model.flame import FlameHead

from .gaussian_model import GaussianModel
from utils.graphics_utils import compute_face_orientation
from roma import rotmat_to_unitquat, quat_xyzw_to_wxyz
from utils.mesh import load_obj


class GoliathGaussianModel(GaussianModel):
    def __init__(
        self,
        sh_degree: int,
    ):
        super().__init__(sh_degree)

        self.faces = torch.from_numpy(
            np.load(
                "/mnt/cluster/pegasus/jschmidt/goliath/m--20230306--0707--AXE977--pilot--ProjectGoliath--Head/kinematic_tracking/faces.npy"
            )
        ).cuda()
        self.vertices_dict = None

        # binding is initialized once the mesh topology is known
        if self.binding is None:
            self.binding = torch.arange(len(self.faces)).cuda()
            self.binding_counter = torch.ones(len(self.faces), dtype=torch.int32).cuda()

    def load_meshes(self, train_meshes, test_meshes, tgt_train_meshes, tgt_test_meshes):
        if self.vertices_dict is None:

            meshes = {**train_meshes, **test_meshes}
            tgt_meshes = {**tgt_train_meshes, **tgt_test_meshes}
            pose_meshes = meshes if len(tgt_meshes) == 0 else tgt_meshes

            self.num_timesteps = max(pose_meshes) + 1  # required by viewers

            T = self.num_timesteps

            self.vertices_dict = torch.zeros(T, meshes[0].shape[0], 3)  # (T, V, 3)

            for i, mesh in pose_meshes.items():
                self.vertices_dict[i] = torch.as_tensor(
                    mesh, dtype=torch.float32
                ).cuda()
            self.vertices_dict = self.vertices_dict.cuda()

        else:
            # NOTE: not sure when this happens
            import ipdb

            ipdb.set_trace()
            pass

    def select_mesh_by_timestep(self, timestep, original=False):
        self.timestep = timestep

        verts = self.vertices_dict[timestep]

        self.update_mesh_properties(verts)

    def update_mesh_properties(self, verts: torch.Tensor):
        faces = self.faces
        triangles = verts[faces]

        # position
        self.face_center = triangles.mean(dim=-2).squeeze(0)

        # orientation and scale
        self.face_orien_mat, self.face_scaling = compute_face_orientation(
            verts.squeeze(0), faces.squeeze(0), return_scale=True
        )
        # self.face_orien_quat = matrix_to_quaternion(self.face_orien_mat)  # pytorch3d (WXYZ)
        self.face_orien_quat = quat_xyzw_to_wxyz(
            rotmat_to_unitquat(self.face_orien_mat)
        )  # roma

        # for mesh rendering
        self.verts = verts
        self.faces = faces

    def training_setup(self, training_args):
        super().training_setup(training_args)

    def save_ply(self, path):
        super().save_ply(path)

        # npz_path = Path(path).parent / "flame_param.npz"
        # flame_param = {k: v.cpu().numpy() for k, v in self.flame_param.items()}
        # np.savez(str(npz_path), **flame_param)

    def load_ply(self, path, **kwargs):
        super().load_ply(path)

        if not kwargs["has_target"]:
            # When there is no target motion specified, use the finetuned FLAME parameters.
            # This operation overwrites the FLAME parameters loaded from the dataset.
            npz_path = Path(path).parent / "flame_param.npz"
            flame_param = np.load(str(npz_path))
            flame_param = {
                k: torch.from_numpy(v).cuda() for k, v in flame_param.items()
            }

            self.flame_param = flame_param
            self.num_timesteps = self.flame_param["expr"].shape[
                0
            ]  # required by viewers

        if "motion_path" in kwargs and kwargs["motion_path"] is not None:
            # When there is a motion sequence specified, load only dynamic parameters.
            motion_path = Path(kwargs["motion_path"])
            flame_param = np.load(str(motion_path))
            flame_param = {
                k: torch.from_numpy(v).cuda()
                for k, v in flame_param.items()
                if v.dtype == np.float32
            }

            self.flame_param = {
                # keep the static parameters
                "shape": self.flame_param["shape"],
                "static_offset": self.flame_param["static_offset"],
                # update the dynamic parameters
                "translation": flame_param["translation"],
                "rotation": flame_param["rotation"],
                "neck_pose": flame_param["neck_pose"],
                "jaw_pose": flame_param["jaw_pose"],
                "eyes_pose": flame_param["eyes_pose"],
                "expr": flame_param["expr"],
                "dynamic_offset": flame_param["dynamic_offset"],
            }
            self.num_timesteps = self.flame_param["expr"].shape[
                0
            ]  # required by viewers

        if "disable_fid" in kwargs and len(kwargs["disable_fid"]) > 0:
            mask = (self.binding[:, None] != kwargs["disable_fid"][None, :]).all(-1)

            self.binding = self.binding[mask]
            self._xyz = self._xyz[mask]
            self._features_dc = self._features_dc[mask]
            self._features_rest = self._features_rest[mask]
            self._scaling = self._scaling[mask]
            self._rotation = self._rotation[mask]
            self._opacity = self._opacity[mask]
