// // ~/fiji --headless --console -macro /n/groups/htem/users/jlr54/raygun/manu_scripts/fuse_CBxFN.ijm '/n/groups/htem/ESRF_id16a/LTP/cb_FN_lobX_feb2021/jlr54/fullres/ cbX_FN_90nm_fullres.xml cbX_FN_90nm_fullres-fuse1.xml'

// // read dataset path, number of tiles as commandline arguments
// args = getArgument()
// args = split(args, " ");

// basePath = args[0];
// if (!endsWith(basePath, File.separator))
// {
//     basePath = basePath + File.separator;
// }
// in_file = args[1];
// out_file = args[2];

// ~/fiji  --headless -macro /n/groups/htem/users/jlr54/raygun/manu_scripts/fuse_CBxFN.ijm
basePath = "/n/groups/htem/ESRF_id16a/LTP/cb_FN_lobX_feb2021/jlr54/fullres/";
in_file = "cbX_FN_90nm_fullres.xml";
out_file = "cbX_FN_90nm_fullres-fuse1.xml";

// fuse dataset, save as HDF5
run("Fuse dataset ...",
    "select="+basePath+in_file+" process_angle=[All angles] process_channel=[All channels] " +
    "process_illumination=[All illuminations] process_tile=[All tiles] process_timepoint=[All Timepoints]" + 
    " bounding_box=[All Views] downsampling=1 pixel_type=[16-bit unsigned integer] interpolation=[Linear Interpolation]" +
    " image=[Cached] blend produce=[All views together] fused_image=[Save as new XML Project (HDF5)] " +
    "subsampling_factors=[{ {1,1,1}, {2,2,2}, {4,4,4}, {8,8,8}, {16,16,16}, {32,32,32} }] " + 
    "hdf5_chunk_sizes=[{ {16,16,16}, {16,16,16}, {16,16,16}, {16,16,16}, {16,16,16}, {16,16,16} }] use_deflate_compression " +
    "export_path=" + basePath + out_file);

// quit after we are finished
eval("script", "System.exit(0);");