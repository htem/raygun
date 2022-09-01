# import copy
import logging
import daisy
import numpy as np

from gunpowder.nodes.batch_filter import BatchFilter
# from gunpowder.coordinate import Coordinate

logger = logging.getLogger(__name__)


def upsample(a, factor):

    for d, f in enumerate(factor):
        a = np.repeat(a, f, axis=d)

    return a


def get_mask_data_in_roi(mask, roi, target_voxel_size):

    roi = daisy.Roi(roi.get_begin(), roi.get_shape())

    # print(f'roi: {roi}')
    # print(f'mask.voxel_size: {mask.voxel_size}')
    # print(f'target_voxel_size: {target_voxel_size}')

    assert mask.voxel_size.is_multiple_of(target_voxel_size), (
        "Can not upsample from %s to %s" % (mask.voxel_size, target_voxel_size))

    aligned_roi = roi.snap_to_grid(mask.voxel_size, mode='grow')
    aligned_data = mask.to_ndarray(aligned_roi, fill_value=0)

    if mask.voxel_size == target_voxel_size:
        return aligned_data

    factor = mask.voxel_size/target_voxel_size

    upsampled_aligned_data = upsample(aligned_data, factor)

    upsampled_aligned_mask = daisy.Array(
        upsampled_aligned_data,
        roi=aligned_roi,
        voxel_size=target_voxel_size)

    return upsampled_aligned_mask.to_ndarray(roi)


class ReplaceSectionsNode(BatchFilter):
    '''
    '''

    def __init__(
            self,
            key,
            delete_section_list=[],
            replace_section_list=[],
            mask_file=None,
            mask_dataset=None,
            ):

        self.key = key
        self.delete_section_list = delete_section_list
        self.replace_section_list = replace_section_list
        self.mask_file = mask_file
        self.mask_dataset = mask_dataset

        if self.mask_file:
            self.mask = daisy.open_ds(self.mask_file, self.mask_dataset)

    def process(self, batch, request):

        array = batch.arrays[self.key]
        roi = array.spec.roi
        # print(f'array.spec.roi: {array.spec.roi}')

        if self.mask_file:
            # mask_data = batch.arrays[self.mask_key].data
            mask_data = get_mask_data_in_roi(self.mask, roi, array.spec.voxel_size)
            array.data *= mask_data

        z_begin = int(roi.get_begin()[0] / array.spec.voxel_size[0])
        z_end = int(roi.get_end()[0] / array.spec.voxel_size[0])

        for z in self.delete_section_list:

            if z >= z_begin and z < z_end:
                z -= z_begin
                array.data[z] = 0

        for z, z_replace in self.replace_section_list:

            if ((z >= z_begin and z < z_end) and
                    (z_replace >= z_begin and z_replace < z_end)):
                z -= z_begin
                z_replace -= z_begin
                array.data[z] = array.data[z_replace]
