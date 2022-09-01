import argparse
import logging
import os
import sys
import numpy as np

import cv2

import daisy

from segway.tasks.launchable_daisy_task import LaunchableDaisyTask

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MakeZarrFromTiffTask")


def calculateNearIsotropicDimensions(voxel_size, max_voxel_count):

    dims = len(voxel_size)
    # assert dims == 3

    voxel_count = 1
    vol_size = [k for k in voxel_size]
    voxel_dims = [1 for k in voxel_size]

    while voxel_count < max_voxel_count:
        for i in range(dims-1, -1, -1):
            if voxel_count >= max_voxel_count:
                continue
            if vol_size[i] == min(vol_size):
                vol_size[i] *= 2
                voxel_count *= 2
                voxel_dims[i] *= 2

    return voxel_dims


class MakeZarrFromTiffTask(LaunchableDaisyTask):

    def _init(self, config):

        self.worker_script_file = os.path.realpath(__file__)

        self.voxel_size = daisy.Coordinate(tuple(self.voxel_size))

        if self.roi_offset is not None or self.roi_shape is not None:
            assert self.roi_offset is not None and self.roi_shape is not None
            self.roi = daisy.Roi(tuple(self.roi_offset), tuple(self.roi_shape))
        else:
            # self.roi = self.input_ds.roi
            assert False

        if self.write_size is None:
            self.write_size = calculateNearIsotropicDimensions(
                self.voxel_size, self.max_voxel_count)

        self.write_size = daisy.Coordinate(tuple(self.write_size))

        self.chunk_size = self.write_size*self.voxel_size

        print(f'voxel_size: {self.voxel_size}')
        print(f'max_voxel_count: {self.max_voxel_count}')
        print(f'calculated write_size: {self.write_size}')
        print(f'calculated chunk_size: {self.chunk_size}')

        self.roi = self.roi.snap_to_grid(self.voxel_size, 'grow')
        ds_roi = self.roi.snap_to_grid(self.chunk_size, 'grow')

        self.bad_sections = set(self.bad_sections)

        # n_channels = 1
        # multiple_channels = self.input_ds.n_channel_dims
        # if multiple_channels:
        #     assert multiple_channels == 1
        #     n_channels = self.input_ds.data.shape[0]
        # print(self.input_ds.data.shape)
        # print(n_channels); exit(0)

        print("Preparing zarr_out with ROI %s and chunk_size %s" % (ds_roi, self.chunk_size))

        self.zarr_out = daisy.prepare_ds(
            self.output_file, self.output_dataset,
            ds_roi,
            self.voxel_size,
            np.uint8,
            # num_channels=n_channels,
            write_size=self.chunk_size,
            force_exact_write_size=True,
            # compressor={'id': 'lz4'},
            compressor={'id': 'zlib', 'level': 3},
            )

    def schedule_blockwise(self):

        # align scheduling block to tiff file size
        x_size = self.x_tile_size*self.voxel_size[2]
        y_size = self.y_tile_size*self.voxel_size[2]
        write_roi = daisy.Roi((0, 0, 0), (self.chunk_size[0], y_size, x_size))
        read_roi = write_roi
        total_roi = self.roi

        self.write_config()

        self._run_daisy(
            total_roi=total_roi,
            read_roi=read_roi,
            write_roi=write_roi,
            check_fn=lambda b: self.check_fn(b),
            )

    def worker_function(self, block):

        print("Processing ", block)

        roi = block.write_roi

        z0, y0, x0 = roi.get_begin() / self.voxel_size

        if self.y_folder_block_size:
            y_folder = int(y0 / self.y_folder_block_size) * self.y_folder_block_size
            row = int((y0 % self.y_folder_block_size) / self.y_tile_size) + 1

        col = int(x0 / self.x_tile_size) + 1
        abs_row = int(y0 / self.y_tile_size) + 1

        sections = range((roi.get_begin()/self.voxel_size)[0], (roi.get_end()/self.voxel_size)[0])

        arr = np.zeros(roi.get_shape() / self.voxel_size, dtype=np.uint8)
        # go through and fill block
        for num in sections:
            # dir_name = "%s/%.4d" % (self.aligned_dir_path, num)
            if self.section_dir_name_format:
                dir_name = self.aligned_dir_path + '/' + self.section_dir_name_format.format(num)
            else:
                dir_name = "%s/%d" % (self.aligned_dir_path, num)  # standard format without 0 padding

            tif = None
            if self.y_folder_block_size:
                tif = "%s/%s/c%.2dr%.2d.tif" % (dir_name, y_folder, col, row)

            if tif is None or not os.path.exists(tif):
                # use default non-parallel format
                tif = "%s/c%.2dr%.2d.tif" % (dir_name, col, abs_row)

            if not os.path.exists(tif):
                if not self.missing_ok and (num not in self.bad_sections):
                    raise RuntimeError("Cannot read %s" % tif)
                continue

            print(tif)

            i = cv2.imread(tif, 0)
            if i.shape != (self.y_tile_size, self.x_tile_size):
                raise RuntimeError("%s is malformed" % tif)

            arr[num-sections[0], :, :] = i

        print("Writing block %s to ZARR.." % block)
        self.zarr_out[roi] = arr

    def check_fn(self, block):

        write_roi = self.zarr_out.roi.intersect(block.write_roi)
        if write_roi.empty():
            return True
        if self.completion_db.count({'block_id': block.block_id}) >= 1:
            return True
        return False


if __name__ == "__main__":

    task = MakeZarrFromTiffTask()

    if len(sys.argv) > 1 and sys.argv[1] == 'run_worker':
        task.run_worker(sys.argv[2])

    else:
        ap = argparse.ArgumentParser()
        # ap.add_argument("input_file", type=str, help='Input hdf/zarr volume')
        ap.add_argument("aligned_dir_path", type=str, help='E.g.: /n/groups/htem/temcagt/datasets/cb2/intersection/aligned_links')
        ap.add_argument("y_tile_size", type=int, help='In pixel')
        ap.add_argument("x_tile_size", type=int, help='In pixel')
        ap.add_argument(
            "voxel_size", type=int, help='In ZYX',
            nargs='+')
        ap.add_argument("output_file", type=str, help='')
        ap.add_argument("output_dataset", type=str, help='')
        ap.add_argument("--section_dir_name_format",
            type=str,
            help='Use Python string.format() to get the directory of each section. By default the script looks for an unpadded sequence of section numbers, e.g., 0, 1, 2, 3, etc.. E.g., "section_{:04d}" will modify this sequence to section_0000, section_0001, etc...',
            default=None)
        ap.add_argument(
            "--write_size", type=int, help='zyx size in pixel',
            nargs='+', default=None)
        ap.add_argument(
            "--max_voxel_count", type=int, help='zyx size in pixel',
            # default=256*1024)
            default=1024*1024)
        ap.add_argument(
            "--roi_offset", type=int, help='In nanometer',
            nargs='+', default=None)
        ap.add_argument(
            "--roi_shape", type=int, help='In nanometer',
            nargs='+', default=None)
        ap.add_argument(
            "--bad_sections", type=int, help='Space separated, e.g., "1 34 66"',
            nargs='+', default=[])
        ap.add_argument("--y_folder_block_size", type=int, help='', default=None)
        ap.add_argument("--missing_ok", type=int, help='Override error when there are missing tiffs. Better to put manually missing/bad sections using the bad_sections argument', default=0)

        config = task.parse_args(ap)

        task.init(config)

        task.schedule_blockwise()
