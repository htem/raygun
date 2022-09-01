# /n/groups/htem/tier2/cb3/sections/170326130159_cb3_0511/intersection/tiles_bridge9

# for i in `seq 200 511`; do
#     path=/n/groups/htem/tier2/cb3/sections/*0${i}/intersection/tiles_bridge9
#     if [ ! -d $path ]; then
#         echo Section $i missing!
#         continue
#     fi
#     ln -s $path $i
# done
# for i in `seq 100 199`; do
#     path=/n/groups/htem/tier2/cb3/sections/*0${i}/intersection/tiles_with_201_511
#     if [ ! -d $path ]; then
#         echo Section $i missing!
#         continue
#     fi
#     ln -s $path $i
# done
# for i in `seq 10 99`; do
#     path=/n/groups/htem/tier2/cb3/sections/*00${i}/intersection/tiles_with_201_511
#     if [ ! -d $path ]; then
#         echo Section $i missing!
#         continue
#     fi
#     ln -s $path $i
# done
# for i in `seq 0 9`; do
#     path=/n/groups/htem/tier2/cb3/sections/*000${i}/intersection/tiles_with_201_511
#     if [ ! -d $path ]; then
#         echo Section $i missing!
#         continue
#     fi
#     ln -s $path $i
# done

aligned_dir_path="/n/groups/htem/temcagt/datasets/cb3/cb3_intersections_std_dir_structure"
output_file='/n/groups/htem/temcagt/datasets/cb3/zarr/cb3.zarr'
output_dataset="volumes/raw"
voxel_size='40 4 4'
write_size='16 256 256'
roi_offset='0 0 0'
# volume size
# Z: 1-511 = 510*40 = 20400
# Y: r1-93 = 92*4*2048 = 753664
# X: c1-149 = 148*4*2048 = 1212416
roi_shape='20400 753664 1212416'
y_tile_size=2048
x_tile_size=2048
bad_sections='0 1 2 4 9 12 25 48 54 68 70 75 76 84 91 106 107 116 121 122 123 137 163 178 209 215 217 219 220 222 226 228 229 232 233 235 236 237 238 241 243 245 249 271 274 277 278 291 296 301 303 305 319 320 322 328 330 331 332 333 334 335 336 340 341 342 343 344 348 353 354 355 356 357 361 362 363 370 371 372 374 375 377 380 381 382 390 391 392 396 397 398 399 400 401 402 403 411 412 418 419 428 431 443 444 448 449 450 451 452 454 455 458 459 460 461 462 463 464 465 466 467 468 469 471 472 473 474 475 476 477 478 479 480 486 487 488 489 495 496 499 506 200'

python segway/tasks/make_zarr_from_tiff/task_make_zarr_from_tiff.py $aligned_dir_path $y_tile_size $x_tile_size $voxel_size $output_file $output_dataset --write_size $write_size --roi_offset $roi_offset --roi_shape $roi_shape --bad_sections $bad_sections --no_launch_workers 1

