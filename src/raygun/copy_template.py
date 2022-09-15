from shutil import copytree
from os.path import isdir, join
from fnmatch import filter
import sys

def include_patterns(inc_patterns, ig_patterns):
    """Factory function that can be used with copytree() ignore parameter.

    Arguments define a sequence of glob-style patterns
    that are used to specify what files to NOT ignore.
    Creates and returns a function that determines this for each directory
    in the file hierarchy rooted at the source directory when used with
    shutil.copytree().
    """
    def _ignore_patterns(path, names):
        keep = set(name for pattern in inc_patterns
                            for name in filter(names, pattern))
        ignore = set(name for name in names
                        if name not in keep and not isdir(join(path, name)) or any(ig_pattern in name for ig_pattern in ig_patterns))
        return ignore
    return _ignore_patterns

def copy_template(source, destination):
    copytree(source, destination, ignore=include_patterns(['*_conf.json', 'retrain.sh'], ['tensorboard', 'models', 'tensorboards', 'snapshots', 'daisy_logs', 'log', '.n5', '.zarr']))

if __name__ == '__main__':
    copy_template(sys.argv[1], sys.argv[2])
    