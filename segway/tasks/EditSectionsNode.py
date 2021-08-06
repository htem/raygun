# import copy
import logging

from gunpowder.nodes.batch_filter import BatchFilter
# from gunpowder.coordinate import Coordinate

logger = logging.getLogger(__name__)


class ReplaceSectionsNode(BatchFilter):
    '''
    '''

    def __init__(
            self,
            key,
            delete_section_list=[],
            replace_section_list=[]
            ):

        self.key = key
        self.delete_section_list = delete_section_list
        self.replace_section_list = replace_section_list

    def process(self, batch, request):

        array = batch.arrays[self.key]
        roi = array.spec.roi

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
