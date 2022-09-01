import argparse
import logging
import os
import sys

import daisy

# import segway.tasks.launchable_daisy_task.LaunchableTask
import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from launchable_daisy_task import LaunchableDaisyTask

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RechunkTask")


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


class RechunkTask(LaunchableDaisyTask):

    def _init(self, config):

        self.worker_script_file = os.path.realpath(__file__)

        self.input_ds = daisy.open_ds(self.input_file, self.input_dataset)

        if self.roi_offset is not None or self.roi_shape is not None:
            assert self.roi_offset is not None and self.roi_shape is not None
            self.roi = daisy.Roi(tuple(self.roi_offset), tuple(self.roi_shape))
        else:
            self.roi = self.input_ds.roi

        self.voxel_size = self.input_ds.voxel_size

        if self.write_size is None:
            self.write_size = calculateNearIsotropicDimensions(
                self.voxel_size, self.max_voxel_count)
            print(self.voxel_size)
            print(self.write_size)

        self.write_size = daisy.Coordinate(tuple(self.write_size))

        self.chunk_size = self.write_size*self.voxel_size
        self.roi = self.roi.snap_to_grid(self.voxel_size, 'grow')

        ds_roi = self.input_ds.roi.snap_to_grid(self.chunk_size, 'grow')

        scheduling_block_size = self.chunk_size
        if self.scheduling_block_size_mult is not None:
            scheduling_block_size = scheduling_block_size * tuple(self.scheduling_block_size_mult)
        self.write_roi = daisy.Roi((0, 0, 0), scheduling_block_size)

        n_channels = 1
        multiple_channels = self.input_ds.n_channel_dims
        if multiple_channels:
            assert multiple_channels == 1
            n_channels = self.input_ds.data.shape[0]
        # print(self.input_ds.data.shape)
        # print(n_channels); exit(0)

        self.output_ds = daisy.prepare_ds(
            self.output_file, self.output_dataset,
            ds_roi,
            self.input_ds.voxel_size,
            self.input_ds.dtype,
            num_channels=n_channels,
            write_size=self.chunk_size,
            force_exact_write_size=True,
            compressor={'id': 'zlib', 'level': 3},
            )

    def schedule_blockwise(self):

        assert len(self.chunk_size) == 3

        logger.info("Rechunking %s/%s to %s/%s with write_size %s (chunk_size %s, bs %s)"
            % (self.input_file, self.input_dataset, self.output_file, self.output_dataset, self.chunk_size, self.write_size, self.write_roi))
        logger.info("ROI: %s" % self.roi)

        # write_roi = daisy.Roi((0, 0, 0), self.chunk_size)
        write_roi = self.write_roi
        read_roi = write_roi
        total_roi = self.roi

        config = {
            'input_file': self.input_file,
            'input_dataset': self.input_dataset,
            'output_file': self.output_file,
            'output_dataset': self.output_dataset,
            'write_size': self.write_size,
            'roi_offset': self.roi_offset,
            'roi_shape': self.roi_shape,
            'max_voxel_count': self.max_voxel_count,
        }

        self.write_config(config)

        self._run_daisy(
            total_roi=total_roi,
            read_roi=read_roi,
            write_roi=write_roi,
            check_fn=lambda b: self.check_fn(b),
            fit='shrink',
            )

    def worker_function(self, block):
        self.output_ds[block.write_roi] = self.input_ds[block.write_roi]

    def check_fn(self, block):

        write_roi = self.output_ds.roi.intersect(block.write_roi)
        if write_roi.empty():
            return True
        if self.completion_db.count({'block_id': block.block_id}) >= 1:
            return True
        return False


if __name__ == "__main__":

    task = RechunkTask()

    if len(sys.argv) > 1 and sys.argv[1] == 'run_worker':
        task.run_worker(sys.argv[2])

    else:
        ap = argparse.ArgumentParser()
        ap.add_argument("input_file", type=str, help='Input hdf/zarr volume')
        ap.add_argument("input_dataset", type=str, help='')
        ap.add_argument("output_file", type=str, help='')
        ap.add_argument("output_dataset", type=str, help='')
        ap.add_argument(
            "--write_size", type=int, help='zyx size in pixel',
            nargs='+', default=None)
        ap.add_argument(
            "--scheduling_block_size_mult", type=int,
            help='zyx size in pixel, must be multiples of write_size',
            nargs='+', default=None)
        ap.add_argument(
            "--max_voxel_count", type=int, help='zyx size in pixel',
            default=256*1024)
        ap.add_argument(
            "--roi_offset", type=int, help='',
            nargs='+', default=None)
        ap.add_argument(
            "--roi_shape", type=int, help='',
            nargs='+', default=None)

        config = task.parse_args(ap)

        task.init(config)

        task.schedule_blockwise()
