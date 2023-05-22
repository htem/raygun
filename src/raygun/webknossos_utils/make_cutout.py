# %%

import os
import sys
import daisy
import webknossos as wk
from webknossos.dataset import COLOR_CATEGORY
from webknossos.dataset.properties import (
    DatasetViewConfiguration,
    LayerViewConfiguration,
)

def make_cutout(
    roi,
    out_file,
    out_dataset,
    src_file="/n/data3/hms/neurobio/htem/temcagt/datasets/cb2/zarr_volume/cb2_v3.n5",
    src_dataset="volumes/raw_mipmap/s1",
):
    src = daisy.open_ds(src_file, src_dataset)
    roi = roi.snap_to_grid(src.voxel_size, mode="closest")
    out = daisy.prepare_ds(
        out_file,
        out_dataset,
        roi,
        src.voxel_size,
        src.dtype,
        # write_roi=roi,
        # num_channels=src.num_channels,
        # compressor=src.compressor,
    )
    out[roi] = src.to_ndarray(roi)
    # print("Server command:")
    # print(f"\tpython -m http.server --directory {out_file}/{out_dataset}/ --bind catmaid2.hms.harvard.edu 9999")
    return 1


def convert_to_webknossos(
        file,
        dataset,
        name=None,
        wk_directory="/home/tmn7/wkw-test/wkw2/binaryData/harvard-htem/"
):
    if name is None:
        name = os.path.basename(file).split(".")[0]
    ds = daisy.open_ds(file, dataset)
    voxel_size = ds.voxel_size
    offset = ds.roi.get_offset()
    data = ds.to_ndarray(ds.roi)
    try:
        wk_ds = wk.Dataset(dataset_path=f"{os.path.dirname(file)}/webknossos/{name}", name=name, voxel_size=voxel_size)
        # wk_ds = wk.Dataset(dataset_path=wk_url, name=name, voxel_size=voxel_size)
        wk_ds.default_view_configuration = DatasetViewConfiguration()
        layer = wk_ds.add_layer("raw", COLOR_CATEGORY, dtype_per_layer=data.dtype)
        layer.add_mag(1, compress=True).write(data, absolute_offset=offset)
        layer.default_view_configuration = LayerViewConfiguration()
    except IndexError:
        pass
    
    print(f"Successfully converted {file} to webknossos format. To move to webknossos server, run:")
    print(f"\tscp -r {os.path.dirname(file)}/webknossos/{name} catmaid2.hms.harvard.edu:{wk_directory}")
    print("\t\tOR, on the server node:")
    print(f"\tmv {os.path.dirname(file)}/webknossos/{name} {wk_directory}")


def upload_raw_to_webknossos(
        file,
        dataset,
        name=None,
        wk_url="http://catmaid2.hms.harvard.edu:9000",
        wk_token="Q9OpWh1PPwHYfH9BsnoM2Q"
):
    if name is None:
        name = os.path.basename(file).split(".")[0]
    ds = daisy.open_ds(file, dataset)
    voxel_size = ds.voxel_size
    offset = ds.roi.get_offset()
    # shape = ds.roi.get_shape()
    data = ds.to_ndarray(ds.roi)
    # data = data.transpose((2, 1, 0))
    with wk.webknossos_context(token=wk_token, url=wk_url):
        try:
            wk_ds = wk.Dataset(dataset_path=f"{os.path.dirname(file)}/webknossos/{name}", name=name, voxel_size=voxel_size)
            # wk_ds = wk.Dataset(dataset_path=wk_url, name=name, voxel_size=voxel_size)
            wk_ds.default_view_configuration = DatasetViewConfiguration()
            layer = wk_ds.add_layer("raw", COLOR_CATEGORY, dtype_per_layer=data.dtype)
            layer.add_mag(1, compress=True).write(data, absolute_offset=offset)
            layer.default_view_configuration = LayerViewConfiguration()
        except IndexError:
            pass
        # remote_dataset = upload_dataset(wk_ds, name)
    #     remote_dataset = wk_ds.upload(name)
    #     # remote_dataset = wk_ds.upload()
    #     url = remote_dataset.url
    # print(f"Successfully uploaded {url}")

# %%
if __name__ == "__main__":
    roi = daisy.Roi((7160, 262320, 400064), (4520, 6160, 6160))
    out_file = "/n/groups/htem/temcagt/datasets/cb2/segmentation/jeff/cb2_gt/synapse_gt/syn_area_val_0.n5"
    out_dataset = "volumes/raw"
    make_cutout(roi, out_file, out_dataset)
    convert_to_webknossos(out_file, out_dataset)
    # upload_raw_to_webknossos(out_file, out_dataset, name="syn_area_val_00")
    # sys.exit()
    
#%%


# python -m http.server --directory /n/groups/htem/temcagt/datasets/cb2/segmentation/jeff/cb2_gt/synapse_gt/syn_area_val_1.n5/ --bind catmaid2.hms.harvard.edu 9999
# %%
# import socket
# import ssl
# import http.server, ssl, socketserver, os

# directory = "/n/groups/htem/temcagt/datasets/cb2/segmentation/jeff/cb2_gt/synapse_gt/syn_area_val_1.n5/"
# hostname = 'catmaid2.hms.harvard.edu'
# port = 9999
# # context = ssl.create_default_context()

# # with socket.create_connection((hostname, port)) as sock:
# #     with context.wrap_socket(sock, server_hostname=hostname) as ssock:
# #         print(ssock.version())
# os.chdir(directory)

# context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
# # context.load_cert_chain("cert.pem") # PUT YOUR cert.pem HERE

# server_address = (hostname, port) # CHANGE THIS IP & PORT

# handler = http.server.SimpleHTTPRequestHandler
# with socketserver.TCPServer(server_address, handler) as httpd:
#     httpd.socket = context.wrap_socket(httpd.socket, server_side=True)
#     httpd.serve_forever()
