from abc import abstractmethod
import gunpowder as gp


class BaseDataPipe(object):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def get_source(self, path, src_names, src_specs=None):
        if path.endswith(".zarr") or path.endswith(".n5"):
            source = gp.ZarrSource(  # add the data source
                path,
                src_names,  # which dataset to associate to the array key
                src_specs,  # meta-information
            )
        elif path.endswith(".h5") or path.endswith(".hdf"):
            source = gp.Hdf5Source(  # add the data source
                path,
                src_names,  # which dataset to associate to the array key
                src_specs,  # meta-information
            )
        else:
            raise NotImplemented(
                f"Datasource type of {path} not implemented yet. Feel free to contribute its implementation!"
            )
        return source

    def prenet_pipe(self, mode: str = "train"):
        # Make pre-net datapipe
        prenet_pipe = self.source
        if mode == "train":
            sections = [
                gp.RandomLocation(),
                "reject",
                "resample",
                "preprocess",
                "augment",
                "unsqueeze",
                "stack",
            ]
        elif mode == "predict":
            sections = ["reject", "resample", "preprocess", "unsqueeze", "stack"]
        elif mode == "test":
            sections = [gp.RandomLocation(), "reject", "resample", "preprocess", "unsqueeze", gp.Stack(1)]
        else:
            raise ValueError(f"mode={mode} not implemented.")

        for section in sections:
            if (
                isinstance(section, str)
                and hasattr(self, section)
                and getattr(self, section) is not None
            ):
                prenet_pipe += getattr(self, section)
            elif isinstance(section, gp.nodes.BatchFilter):
                prenet_pipe += section

        return prenet_pipe

    @abstractmethod
    def postnet_pipe(self):
        raise NotImplementedError()
