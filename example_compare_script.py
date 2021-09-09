from compare import Compare

src = "/n/groups/htem/ESRF_id16a/tomo_ML/ReducedAnglesXray/jlr54_tests/volumes/mCTX_17keV_30nm_512c_first256/mCTX_17keV_30nm_512c_first256.zarr/volumes"
out_path = "/n/groups/htem/ESRF_id16a/tomo_ML/ReducedAnglesXray/jlr54_tests/volumes/mCTX_17keV_30nm_512c_first256/"

comp = Compare(src, out_path=out_path)
comp.compare()