import argparse
import logging
import numpy as np
import os
from skimage import measure
import sys
import struct

import daisy

# import segway.tasks.launchable_daisy_task.LaunchableTask
import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from launchable_daisy_task import LaunchableDaisyTask

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MeshingTask")


def getHierarchicalMeshPath(object_id, hierarchical_size):

    assert object_id != 0

    level_dirs = []
    num_level = 0
    while object_id > 0:
        level_dirs.append(int(object_id % hierarchical_size))
        object_id = int(object_id / hierarchical_size)
    num_level = len(level_dirs) - 1
    level_dirs = [str(lv) for lv in reversed(level_dirs)]
    return os.path.join(str(num_level), *level_dirs)


def downscale_block(in_data, factor, add_boundary_data=True):

    dims = len(factor)

    # in_shape = daisy.Coordinate(in_data.shape[-dims:])
    # assert in_shape.is_multiple_of(factor), "%s has to be multipes of %s" % (in_shape, factor)

    n_channels = len(in_data.shape) - dims
    if n_channels >= 1:
        factor = (1,)*n_channels + factor

    if add_boundary_data:
        assert dims == 3
        in_data = np.concatenate(
            [in_data, in_data[:, :, -1:]], axis=2)
        in_data = np.concatenate(
            [in_data, in_data[:, -1:, :]], axis=1)
        in_data = np.concatenate(
            [in_data, in_data[-1:, :, :]], axis=0)

    if in_data.dtype == np.uint64:
        if add_boundary_data:
            slices = tuple(slice(0, None, k) for k in factor)
        else:
            slices = tuple(slice(k//2, None, k) for k in factor)
        out_data = in_data[slices]

    else:
        out_data = skimage.measure.block_reduce(in_data, factor, np.mean)

    return out_data


def save_mesh_as_precomputed(file, vertices, triangles):
    """Store a mesh in Neuroglancer pre-computed format.
    :param file: a file-like object opened in binary mode (its ``write`` method
        will be called with :class:`bytes` objects).
    :param numpy.ndarray vertices: the list of vertices of the mesh. Must be
        convertible to an array of size Nx3 and type ``float32``. Coordinates
        will be interpreted by Neuroglancer in nanometres.
    :param numpy.ndarray triangles: the list of triangles of the mesh. Must be
        convertible to an array of size Mx3 and ``uint32`` data type.
    :raises AssertionError: if the inputs do not match the constraints above
    """
    # https://github.com/HumanBrainProject/neuroglancer-scripts/blob/47babe5f1d4b379a603ab21b07f6ece3ffc1fc0d/src/neuroglancer_scripts/mesh.py
    vertices = np.asarray(vertices)
    assert vertices.ndim == 2
    triangles = np.asarray(triangles)
    assert triangles.ndim == 2
    assert triangles.shape[1] == 3
    # if not np.can_cast(vertices.dtype, "<f"):
    #     logger.debug("Vertex coordinates will be converted to float32")
    file.write(struct.pack("<I", vertices.shape[0]))
    file.write(vertices.astype("<f").tobytes(order="C"))
    # file.write(triangles.astype("<I", casting="safe").tobytes(order="C"))
    file.write(triangles.astype("<I").tobytes(order="C"))


class MeshingTask(LaunchableDaisyTask):

    def _init(self, config):

        self.worker_script_file = os.path.realpath(__file__)

        self.input_file = config['input_file']
        self.label_key = config['label_key']
        self.output_dir = config['output_dir']
        self.block_size = config['block_size']
        self.roi_offset = config['roi_offset']
        self.roi_shape = config['roi_shape']
        self.context = config['context']
        self.downsample = config['downsample']
        self.hierarchical_path_size = config['hierarchical_path_size']

        self.block_size = tuple(self.block_size)
        self.context = tuple(self.context)

        assert self.context == (0, 0, 0), "Context for meshing is not supported yet"

        self.ds = daisy.open_ds(self.input_file, self.label_key)

        if self.roi_offset is not None or self.roi_shape is not None:
            assert self.roi_offset is not None and self.roi_shape is not None
            self.roi = daisy.Roi(tuple(self.roi_offset), tuple(self.roi_shape))
        else:
            self.roi = self.ds.roi

        self.voxel_size = self.ds.voxel_size
        self.roi = self.roi.snap_to_grid(self.voxel_size, 'grow')

        self.output_dir_frag_mesh = os.path.join(self.output_dir, "mesh")
        if not os.path.exists(self.output_dir_frag_mesh):
            logger.info("Creating mesh dir %s" % self.output_dir_frag_mesh)
            os.makedirs(self.output_dir_frag_mesh, exist_ok=True)

    def schedule_blockwise(self):

        assert len(self.block_size) == 3

        write_roi = daisy.Roi((0, 0, 0), self.block_size)
        read_roi = write_roi.grow(self.context, self.context)
        total_roi = self.roi.grow(self.context, self.context)

        config = {
            'input_file': self.input_file,
            'label_key': self.label_key,
            'output_dir': self.output_dir,
            'block_size': self.block_size,
            'roi_offset': self.roi.get_offset(),
            'roi_shape': self.roi.get_shape(),
            'context': self.context,
            'downsample': self.downsample,
            'hierarchical_path_size': self.hierarchical_path_size,
            # 'block_size': self.block_size,
        }

        self.write_config(config)

        self._run_daisy(
            total_roi=total_roi,
            read_roi=read_roi,
            write_roi=write_roi,
            )

    def worker_function(self, block):

        ndarray = self.ds[block.read_roi].to_ndarray()
        voxel_size = self.voxel_size

        if self.downsample is not None:
            assert len(self.downsample) == 3
            ndarray = downscale_block(ndarray, self.downsample)
            voxel_size = [
                voxel_size[0]*self.downsample[0],
                voxel_size[1]*self.downsample[1],
                voxel_size[2]*self.downsample[2],
                ]

        labels = np.unique(ndarray)
        offset = block.read_roi.get_offset()
        # offset += tuple(voxel_size)
        # offset += tuple([40, 0, 0])
        # offset += tuple([40, 64, 64])

        for object_id in labels:

            if object_id == 0:
                # assuming background
                continue

            mask = ndarray == object_id
            logger.debug('Generating isosurface for %s' % (object_id))
            verts, faces, normals, values = measure.marching_cubes_lewiner(
                mask, 0.5, spacing=voxel_size, step_size=1)

            verts = verts + offset
            verts = np.flip(verts, 1)  # data is in zyx but neuroglancer speaks xyz

            # faces = faces + 1

            self.precomputed_write_fn(
                object_id,
                verts,
                normals,
                faces,
                )

    def arlo_write_fn(
            self,
            object_id,
            verts,
            normals,
            faces,
            ):

        save_name = os.path.join(self.output_dir, 'mesh_%i.obj' % (object_id))
        obj = open(save_name, mode='w')
        for item in verts:
            obj.write("v {0} {1} {2}\n".format(item[2],item[1],item[0]))
        for item in normals:
            obj.write("vn {0} {1} {2}\n".format(item[2],item[1],item[0]))
        for item in faces:
            obj.write("f {0}//{0} {1}//{1} {2}//{2}\n".format(item[2],item[1],item[0]))
        obj.close()

    def precomputed_write_fn(
            self,
            object_id,
            verts,
            normals,
            faces,
            ):

        if self.hierarchical_path_size:
            save_name = os.path.join(
                self.output_dir_frag_mesh,
                getHierarchicalMeshPath(object_id, self.hierarchical_path_size))
            os.makedirs(os.path.dirname(save_name), exist_ok=True)
        else:
            save_name = os.path.join(self.output_dir_frag_mesh, '%i' % (object_id))

        obj = open(save_name, mode='wb')
        save_mesh_as_precomputed(obj, verts, faces)
        obj.close()


if __name__ == "__main__":

    task = MeshingTask()

    if len(sys.argv) > 1 and sys.argv[1] == 'run_worker':
        task.run_worker(sys.argv[2])

    else:
        ap = argparse.ArgumentParser()
        ap.add_argument("input_file", type=str, help='Input hdf/zarr volume')
        ap.add_argument("label_key", type=str, help='Label key to use')
        ap.add_argument("output_dir", type=str, help='Directory to save meshes to')

        ap.add_argument(
            "--block_size", type=int, help='zyx in nm, should align to the superfragment block size',
            nargs='+')
        ap.add_argument(
            "--roi_offset", type=int, help='',
            nargs='+', default=None)
        ap.add_argument(
            "--roi_shape", type=int, help='',
            nargs='+', default=None)
        ap.add_argument(
            "--context", type=int, help='',
            nargs='+', default=[0, 0, 0])
        # ap.add_argument(
        #     "--downsample", type=int, help='',
        #     default=1)
        ap.add_argument(
            "--downsample", type=int, help='',
            nargs='+', default=None)
        ap.add_argument(
            "--hierarchical_path_size", type=int, help='',
            default=10000)

        config = task.parse_args(ap)

        task.init(config)

        task.schedule_blockwise()
