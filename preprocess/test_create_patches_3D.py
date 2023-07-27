import os
import shutil
import sys
sys.path.append('..')
from unittest import mock
from utils.testing_utils import *
from gen_artificial_data import *
from create_patches_3D import setup, seg_and_patch_3D

_c = Colorcodes()

patching_args = mock.MagicMock()
patching_args.depth = None
patching_args.step_size = 96
patching_args.patch_size = 96
patching_args.slice_mode = 'all'
patching_args.patch = True
patching_args.seg = True
patching_args.stitch = True
patching_args.verbose = True
patching_args.no_auto_skip = True
patching_args.save_mask = True
patching_args.preset = None
patching_args.patch_level = 0
patching_args.downscale = 4
patching_args.process_list = None
patching_args.clip_min = 0.0
patching_args.clip_max = 255.0
patching_args.black_thresh = -1 # TODO: make black thresholding work better with phantom data?
patching_args.mthresh = 15
patching_args.sthresh = 100
patching_args.a_h = 0
patching_args.a_t = 0
patching_args.contour_fn = 'all'
patching_args.area_thresh = 0.0
patching_args.thresh_mode = 'fixed'
patching_args.normalize = False


#### test gray 3d
# If currently nonexistent, generate some gray phantom data with which to test the pipeline on
print('Testing Gray 3D Tiff image patching')
os.makedirs('../test/source/gray', exist_ok=True)
if not os.path.exists('../test/source/gray/gray_polygons_19'):
    print('Regenerating TIFF gray test data!')
    gen_data('polygons', 20, '../test/source/gray/gray', mode='L', n_obj=25, d=360, w=360, h=360, size=32)
    gen_data('spheres', 20, '../test/source/gray/gray', mode='L', n_obj=25, d=360, w=360, h=360, size=32)
    create_csv(['spheres', 'polygons'], [20, 20], ['gray', 'gray'], '../test/source/gray/gray_phantom_clinical_list.csv')
gray_phantom = load_generated_img('../test/source/gray/gray_polygons_19')
print(f'Loaded image shape: {gray_phantom.shape}')

# Remove results of old test runs
if os.path.exists('../test/save_dir/gray/patch_96_step_96_3D_all_fixed/'):
    shutil.rmtree('../test/save_dir/gray/patch_96_step_96_3D_all_fixed/')
if os.path.exists('../test/save_dir/gray/patch_96_step_96_2D_single_fixed/'):
    shutil.rmtree('../test/save_dir/gray/patch_96_step_96_2D_single_fixed/')

patching_args.source = '../test/source/gray'
patching_args.save_dir = '../test/save_dir/gray'
patching_args.patch_mode = '3D'
directories, parameters, process_list = setup(patching_args)
print("Running segmentation and patching...")

with HiddenPrints():
    seg_times, patch_times = seg_and_patch_3D(**directories,
                                            **parameters,
                                            patch_size=patching_args.patch_size,
                                            step_size=patching_args.step_size,
                                            step_size_z=patching_args.step_z,
                                            depth = patching_args.depth,
                                            seg = patching_args.seg,
                                            save_mask=patching_args.save_mask,
                                            stitch=patching_args.stitch,
                                            patch_level=patching_args.patch_level,
                                            patch=patching_args.patch,
                                            patch_mode=patching_args.patch_mode,
                                            slice_mode=patching_args.slice_mode,
                                            process_list = process_list,
                                            auto_skip=patching_args.no_auto_skip,
                                            verbose=patching_args.verbose)

success = Conditional()
success.file_check('../test/save_dir/gray/patch_96_step_96_3D_all_fixed/masks/gray_polygons_19/gray_polygons_19_zlevel_0.png')
success.file_check('../test/save_dir/gray/patch_96_step_96_3D_all_fixed/patches/gray_polygons_19_patches.h5')
success.check(len(os.listdir('../test/save_dir/gray/patch_96_step_96_3D_all_fixed/stitches/')) >= 1)
if success.eval():
    print(_c.green + _c.bold + 'TIFF Gray 3D Success!' + _c.reset)
else:
    print(_c.red + _c.bold + 'TIFF Gray 3D Failure!' + _c.reset)

## test gray 2d
## TODO: Test for all 2D slice (akin to all wsi)?
# gray_2d_phantom = None
# prev_max = 0
# for i in range(gray_phantom.shape[0]):
#     avg = np.mean(gray_phantom[i])
#     if avg > prev_max:
#         gray_2d_phantom = gray_phantom[i]
#         prev_max = avg
print('Testing Gray 2D Tiff image patching')

patching_args.source = '../test/source/gray'
patching_args.save_dir = '../test/save_dir/gray'
patching_args.patch_mode = '2D'
patching_args.step_z = 16
patching_args.slice_mode = 'single'
directories, parameters, process_list = setup(patching_args)
print("Running segmentation and patching...")
with HiddenPrints():
    seg_times, patch_times = seg_and_patch_3D(**directories,
                                            **parameters,
                                            patch_size=patching_args.patch_size,
                                            step_size=patching_args.step_size,
                                            step_size_z=patching_args.step_z,
                                            depth = patching_args.depth,
                                            seg = patching_args.seg,
                                            save_mask=patching_args.save_mask,
                                            stitch=patching_args.stitch,
                                            patch_level=patching_args.patch_level,
                                            patch=patching_args.patch,
                                            patch_mode=patching_args.patch_mode,
                                            slice_mode=patching_args.slice_mode,
                                            process_list = process_list,
                                            auto_skip=patching_args.no_auto_skip,
                                            verbose=patching_args.verbose)

success.reset()
success.file_check('../test/save_dir/gray/patch_96_step_96_2D_single_fixed/masks/gray_polygons_19/gray_polygons_19_zlevel_0.png')
success.file_check('../test/save_dir/gray/patch_96_step_96_2D_single_fixed/patches/gray_polygons_19_patches.h5')
success.check(len(os.listdir('../test/save_dir/gray/patch_96_step_96_2D_single_fixed/stitches/')) >= 1)
if success.eval():
    print(_c.green + _c.bold + 'TIFF Gray 2D Success!' + _c.reset)
else:
    print(_c.red + _c.bold + 'TIFF Gray 2D Failure!' + _c.reset)

# test color 3d
# If currently nonexistent, generate some rgb phantom data with which to test the pipeline on
print('Testing Color 3D DCM image patching')
os.makedirs('../test/source/color', exist_ok=True)
if not os.path.exists('../test/source/color/color_polygons_19'):
    print('Regenerating DCM rgb test data!')
    gen_data('polygons', 20, '../test/source/color/color', mode='RGB', n_obj=25, d=360, w=360, h=360, size=32, filetype='dcm')
    gen_data('spheres', 20, '../test/source/color/color', mode='RGB', n_obj=25, d=360, w=360, h=360, size=32, filetype='dcm')
    create_csv(['spheres', 'polygons'], [20, 20], ['color', 'color'], '../test/source/color/color_phantom_clinical_list.csv')
color_phantom = load_generated_img('../test/source/color/color_polygons_19')
print(f'Loaded image shape: {color_phantom.shape}')

if os.path.exists('../test/save_dir/color/patch_96_step_96_3D_all_fixed/'):
    shutil.rmtree('../test/save_dir/color/patch_96_step_96_3D_all_fixed/')
# if os.path.exists('../test/save_dir/color/patch_96_step_96_2D_single_fixed/'):
#     shutil.rmtree('../test/save_dir/color/patch_96_step_96_2D_single_fixed/')

patching_args.source = '../test/source/color'
patching_args.save_dir = '../test/save_dir/color'
patching_args.patch_mode = '3D'
patching_args.slice_mode = 'all'
directories, parameters, process_list = setup(patching_args)
print("Running segmentation and patching...")
with HiddenPrints():
    seg_times, patch_times = seg_and_patch_3D(**directories,
                                            **parameters,
                                            patch_size=patching_args.patch_size,
                                            step_size=patching_args.step_size,
                                            step_size_z=patching_args.step_z,
                                            depth = patching_args.depth,
                                            seg = patching_args.seg,
                                            save_mask=patching_args.save_mask,
                                            stitch=patching_args.stitch,
                                            patch_level=patching_args.patch_level,
                                            patch=patching_args.patch,
                                            patch_mode=patching_args.patch_mode,
                                            slice_mode=patching_args.slice_mode,
                                            process_list = process_list,
                                            auto_skip=patching_args.no_auto_skip,
                                            verbose=patching_args.verbose)

success.reset()
success.file_check('../test/save_dir/color/patch_96_step_96_3D_all_fixed/masks/color_polygons_19/color_polygons_19_zlevel_0.png')
success.file_check('../test/save_dir/color/patch_96_step_96_3D_all_fixed/patches/color_polygons_19_patches.h5')
success.check(len(os.listdir('../test/save_dir/color/patch_96_step_96_3D_all_fixed/stitches/')) >= 1)
if success.eval():
    print(_c.green + _c.bold + 'DCM Color 3D Success!' + _c.reset)
else:
    print(_c.red + _c.bold + 'DCM Color 3D Failure!' + _c.reset)

# ## test color 2d

# TODO: finish color 2D testing once it can work
# patching_args.source = '../test/source/color'
# patching_args.save_dir = '../test/save_dir/color'
# patching_args.patch_mode = '2D'
# patching_args.step_z = 10
# patching_args.slice_mode = 'single'
# directories, parameters, process_list = setup(patching_args)
# seg_times, patch_times = seg_and_patch_3D(**directories,
#                                           **parameters,
#                                           patch_size=patching_args.patch_size,
#                                           step_size=patching_args.step_size,
#                                           step_size_z=patching_args.step_z,
#                                           depth = patching_args.depth,
#                                           seg = patching_args.seg,
#                                           save_mask=patching_args.save_mask,
#                                           stitch=patching_args.stitch,
#                                           patch_level=patching_args.patch_level,
#                                           patch=patching_args.patch,
#                                           patch_mode=patching_args.patch_mode,
#                                           slice_mode=patching_args.slice_mode,
#                                           process_list = process_list,
#                                           auto_skip=patching_args.no_auto_skip,
#                                           verbose=patching_args.verbose)

# assert os.path.exists('../test/save_dir/color/patch_96_step_96_2D_all_fixed/masks/color_phantom/color_phantom_zlevel_0.png')
# assert os.path.exists('../test/save_dir/color/patch_96_step_96_2D_all_fixed/patches/color_phantom_patches.h5')
# print('Color 2D Success!')