import argparse
import logging
import os
import sys
import numpy as np
import skimage
# import copy

import daisy

from segway.tasks.launchable_daisy_task import LaunchableDaisyTask

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ScalePyramidTask")


def downscale_block(in_array, out_array, scale_factor, block):

    scale_factor = tuple(scale_factor)
    dims = len(scale_factor)
    in_data = in_array.to_ndarray(block.read_roi, fill_value=0)

    in_shape = daisy.Coordinate(in_data.shape[-dims:])
    assert in_shape.is_multiple_of(scale_factor)

    n_channels = len(in_data.shape) - dims
    if n_channels >= 1:
        scale_factor = (1,)*n_channels + scale_factor

    if in_data.dtype == np.uint64:
        slices = tuple(slice(k//2, None, k) for k in scale_factor)
        out_data = in_data[slices]
    else:
        out_data = skimage.measure.block_reduce(in_data, scale_factor, np.mean)

    out_array[block.write_roi] = out_data

    return 0


def calculateNearIsotropicDimensions(voxel_size, max_voxel_count):

    dims = len(voxel_size)
    # assert dims == 3

    voxel_count = 1
    vol_size = [k for k in voxel_size]
    voxel_dims = [1 for k in voxel_size]

    while voxel_count < max_voxel_count:
        # for i in range(dims-1, -1, -1):
        for i in range(0, dims):
            if voxel_count >= max_voxel_count:
                continue
            if vol_size[i] == min(vol_size):
                vol_size[i] *= 2
                voxel_count *= 2
                voxel_dims[i] *= 2

    return voxel_dims


def calculateNextIsotropicScaleFactor(voxel_size,
        max_downsampling=2,
        downsampling_mult=2,
        # preferred_axis=[0],
        preferred_axis=[],
        ):

    dims = len(voxel_size)
    voxel_size = [k for k in voxel_size]
    factors = [1 for k in voxel_size]

    while True:

        if max(voxel_size)/min(voxel_size) < downsampling_mult:
            max_voxel_size = max(voxel_size)
            for i in range(dims-1, -1, -1):
                if i in preferred_axis:
                    continue
                if factors[i] >= max_downsampling:
                    continue
                if max_voxel_size / voxel_size[i] < downsampling_mult:
                    factors[i] *= downsampling_mult
                    voxel_size[i] *= downsampling_mult

        for i in range(dims-1, -1, -1):
            if factors[i] >= max_downsampling:
                continue
            if max(voxel_size) / voxel_size[i] >= downsampling_mult:
                factors[i] *= downsampling_mult
                voxel_size[i] *= downsampling_mult

        if max(factors) >= max_downsampling:
            break

    return daisy.Coordinate(factors)


class ScalePyramidTask(LaunchableDaisyTask):

    def _init(self, config):

        self.worker_script_file = os.path.realpath(__file__)

        try:
            self.in_ds = daisy.open_ds(self.in_file, self.in_ds_name)
        except:
            print("ERROR: Dataset %s not found in %s" % (
                self.in_ds_name, self.in_file))
            exit(1)

        voxel_size = self.in_ds.voxel_size

        if self.in_ds.n_channel_dims == 0:
            num_channels = 1
        elif self.in_ds.n_channel_dims == 1:
            num_channels = self.in_ds.shape[0]
        else:
            raise RuntimeError(
                "more than one channel not yet implemented, sorry...")

        self.ds_roi = self.in_ds.roi

        sub_roi = None
        if self.roi_offset is not None or self.roi_shape is not None:
            assert self.roi_offset is not None and self.roi_shape is not None
            self.schedule_roi = daisy.Roi(
                tuple(self.roi_offset), tuple(self.roi_shape))
            sub_roi = self.schedule_roi
        else:
            self.schedule_roi = self.in_ds.roi

        if self.scale_factor is not None:
            try:
                self.scale_factor = daisy.Coordinate(self.scale_factor)
            except Exception:
                self.scale_factor = daisy.Coordinate(
                    (self.scale_factor,)*self.write_size.dims())
        else:
            self.scale_factor = calculateNextIsotropicScaleFactor(voxel_size)

        out_voxel_size = voxel_size*self.scale_factor

        if self.chunk_shape_voxel is None:
            self.chunk_shape_voxel = calculateNearIsotropicDimensions(
                out_voxel_size, self.max_voxel_count)
            print(out_voxel_size)
            print(self.chunk_shape_voxel)
        self.chunk_shape_voxel = daisy.Coordinate(self.chunk_shape_voxel)

        self.schedule_roi = self.schedule_roi.snap_to_grid(
            out_voxel_size,
            mode='grow')
        out_ds_roi = self.ds_roi.snap_to_grid(
            out_voxel_size,
            mode='grow')

        self.write_size = self.chunk_shape_voxel*out_voxel_size

        scheduling_block_size = self.write_size
        if self.scheduling_block_size_mult is not None:
            scheduling_block_size = scheduling_block_size * tuple(self.scheduling_block_size_mult)
        self.write_roi = daisy.Roi((0, 0, 0), scheduling_block_size)

        if sub_roi is not None:
            # with sub_roi, the coordinates are absolute
            # so we'd need to align total_roi to the write size too
            self.schedule_roi = self.schedule_roi.snap_to_grid(
                self.write_size, mode='grow')
            out_ds_roi = out_ds_roi.snap_to_grid(
                self.write_size, mode='grow')

        print("out_ds_roi:", out_ds_roi)
        print("schedule_roi:", self.schedule_roi)
        print("write_size:", self.write_size)
        print("self.in_ds.voxel_size:", self.in_ds.voxel_size)
        print("out_voxel_size:", out_voxel_size)
        # exit(0)

        delete = self.overwrite == 2

        self.out_ds = daisy.prepare_ds(
            self.out_file,
            self.out_ds_name,
            total_roi=out_ds_roi,
            voxel_size=out_voxel_size,
            write_size=self.write_size,
            dtype=self.in_ds.dtype,
            num_channels=num_channels,
            force_exact_write_size=True,
            compressor={'id': 'zlib', 'level': 3},
            delete=delete,
            )

    def schedule_blockwise(self):

        assert len(self.chunk_shape_voxel) == 3

        logger.info(
            "Rechunking %s/%s to %s/%s with chunk_shape_voxel %s (write_size %s, scheduling_bs %s)"
            % (self.in_file, self.in_ds_name, self.out_file, self.out_ds_name, self.chunk_shape_voxel, self.write_size, self.write_roi))
        logger.info("ROI: %s" % self.schedule_roi)

        write_roi = self.write_roi
        read_roi = write_roi
        total_roi = self.schedule_roi

        self.write_config()

        self._run_daisy(
            total_roi=total_roi,
            read_roi=read_roi,
            write_roi=write_roi,
            check_fn=lambda b: self.check_fn(b),
            )

    def worker_function(self, block):
        return downscale_block(self.in_ds, self.out_ds, self.scale_factor, block)

    def check_fn(self, block):

        write_roi = self.out_ds.roi.intersect(block.write_roi)
        if write_roi.empty():
            return True
        if self.completion_db.count({'block_id': block.block_id}) >= 1:
            return True
        return False


if __name__ == "__main__":

    task = ScalePyramidTask()

    if len(sys.argv) > 1 and sys.argv[1] == 'run_worker':
        task.run_worker(sys.argv[2])

    else:
        ap = argparse.ArgumentParser(
            description="Create a scale pyramide for a zarr/N5 container.")
        ap.add_argument("in_file", type=str, help='The input container')
        ap.add_argument("in_ds_name", type=str, help='The name of the dataset')
        ap.add_argument("out_file", type=str, help='The input container')
        ap.add_argument("out_ds_name", type=str, help='The name of the dataset')
        ap.add_argument(
            "--chunk_shape_voxel", type=int, help='The size of a chunk in voxels',
            nargs='+', default=None)
        ap.add_argument(
            "--max_voxel_count", type=int, help='zyx size in pixel',
            default=256*1024)
        ap.add_argument(
            "--scale_factor", type=int, help='zyx size in pixel',
            nargs='+', default=None)
        ap.add_argument(
            "--roi_offset", type=int, help='',
            nargs='+', default=None)
        ap.add_argument(
            "--roi_shape", type=int, help='',
            nargs='+', default=None)
        ap.add_argument(
            "--scheduling_block_size_mult", type=int,
            help='zyx size in pixel, must be multiples of write_size',
            nargs='+', default=None)

        config = task.parse_args(ap)

        task.init(config)

        task.schedule_blockwise()
