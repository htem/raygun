import os

class SubsegmentGraph:

    def __init__(
            self,
            filepath,
            blocksize,
            chunk,
            thresholds,
            merge_function,
            ):

        # lut_dir_subseg = os.path.join(filepath, self.lut_dir)
        self.filepath_local = filepath

        super_lut_dir = 'super_%dx%dx%d_%s' % (
            chunk[0], chunk[1], chunk[2],
            merge_function)

        self.filepath_subseg = os.path.join(filepath, super_lut_dir)

        self.thresholds = thresholds
        self.blocksize = blocksize

    def create_attributes(self, attributes):

        for threshold in self.thresholds:
            threshold_dir = self.filepath_subseg + '_%d' % int(threshold*100)

            for attribute in attributes:
                os.makedirs(os.path.join(threshold_dir, attribute), exist_ok=True)

    # def get_subseg_attribute_path(self, attr, threshold):

    #     return os.path.join(
    #         self.filepath_subseg + '_%d' % int(threshold*100),
    #         attr)

    def exists_attribute_subseg(self, attr, threshold, blockid):

        p = os.path.join(
            self.filepath_subseg + '_%d' % int(threshold*100),
            attr,
            "%d.npz" % blockid)

        return os.path.exists(p)

    def load_attribute(self, attr, blockid):



