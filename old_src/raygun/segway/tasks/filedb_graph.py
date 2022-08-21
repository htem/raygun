import os
import itertools

import numpy as np

from daisy import Roi, Coordinate, Block

class FileDBGraph:

    def __init__(
            self,
            filepath,
            blocksize,
            roi_offset,
            attr_postpend=None,
            ):

        self.filepath = filepath
        self.attr_postpend = attr_postpend
        self.blocksize = tuple(blocksize)
        self.roi_offset = Coordinate(roi_offset)

    def create_attributes(self, attributes, thresholds=[None]):

        if isinstance(attributes, str):
            attributes = [attributes]

        for threshold in thresholds:
            for attribute in attributes:
                print(attribute)
                if threshold is not None:
                    attribute += '_%d' % int(threshold*100)
                new_dir = os.path.join(self.filepath, attribute)
                print("Creating", new_dir)
                os.makedirs(new_dir, exist_ok=True)

    def get_attribute_path(self, attr, block_id, threshold=None):
        if self.attr_postpend:
            attr += '_%s' % self.attr_postpend
        if threshold:
            attr += '_%d' % int(threshold*100)
        p = os.path.join(
            self.filepath,
            attr,
            "%d.npz" % block_id)
        return p

    def exists_attribute(self, attr, block, threshold=None):
        return os.path.exists(self.get_attribute_path(attr, block.block_id, threshold))

    def load_attribute(self, attr, block, threshold=None, attr_name=None):

        if attr_name is None:
            attr_name = attr

        attr_data = []

        block_ids = self.enumerate_id_sub_blocks(block, ensure_aligned=True)
        for block_id in block_ids:
            attr_path = self.get_attribute_path(attr, block_id, threshold)
            # print(attr_path)
            try:
                d = np.load(attr_path)[attr_name]
                attr_data.append(d)
            except:
                print("Couldn't load", attr_path)
                pass

        ret = np.block(attr_data)
        # print(ret); exit()
        return ret

    def write_attribute(self, data, attr, block, threshold=None, attr_name=None):

        if attr_name is None:
            attr_name = attr
        block_ids = self.enumerate_id_sub_blocks(block, ensure_aligned=True)
        assert len(block_ids) == 1  # support single block write for now
        for block_id in block_ids:
            attr_path = self.get_attribute_path(attr, block_id, threshold)
            np.savez_compressed(attr_path, **{attr_name:data})

    def enumerate_id_sub_blocks(self, block, ensure_aligned=True):
        # for index in 
        indices = self.get_chunks(block.read_roi, self.blocksize, self.roi_offset, ensure_aligned=ensure_aligned)
        block_ids = []
        for i in indices:
            block_ids.append(Block.index2id(i))
        return block_ids

    def get_chunks(self, roi, chunk_size, roi_offset=None, ensure_aligned=True):
        '''Get a list of chunk indices and a list of chunk ROIs for each chunk
        that overlaps with the given ROI.'''

        # print(roi_offset)
        if roi_offset is not None:
            roi = roi.shift(-roi_offset)

        chunk_roi = roi.snap_to_grid(chunk_size, mode='grow')
        if ensure_aligned:
            # print(chunk_roi)
            # print(roi)
            assert chunk_roi == roi

        # print(chunk_size)
        chunks = chunk_roi/chunk_size
        # print(chunks); exit()

        chunk_indices = itertools.product(*[
            range(chunks.get_begin()[d], chunks.get_end()[d])
            for d in range(chunks.dims())
        ])

        # for e in chunk_indices: print(e)
        # print(chunk_indices); exit()

        return chunk_indices



