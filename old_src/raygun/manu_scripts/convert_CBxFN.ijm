
basePath = "/n/groups/htem/ESRF_id16a/LTP/cb_FN_lobX_feb2021/jlr54/fullres/";
in_file = "cbX_FN_90nm_fullres-fuse1.xml";
out_n5 = "cbX_FN_90nm_fullres-fuse1.n5";
out_xml = "cbX_FN_90nm_fullres-fuse1-n5.xml";

// resave HDF5 as N5
run("As N5 ...",
    "select=" + basePath + in_file + " resave_angle=[All angles] resave_channel=[All channels] " +
    "resave_illumination=[All illuminations] resave_tile=[All tiles] resave_timepoint=[All Timepoints] " + 
    "compression=[Gzip] subsampling_factors=[{ {1,1,1}, {2,2,2}, {4,4,4}, {8,8,8}, {16,16,16}, {32,32,32}, {64,64,64} }] " + 
    "n5_block_sizes=[{ {64,64,64}, {64,64,64}, {64,64,64}, {64,64,64}, {64,64,64}, {64,64,64}, {64,64,64} }] " +
    "output_xml=" + basePath + out_xml + " output_n5=" + basePath + out_n5 + " write_xml write_data ");

// quit after we are finished
eval("script", "System.exit(0);");
