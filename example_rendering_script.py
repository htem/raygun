import sys
sys.path.append('/n/groups/htem/users/jlr54/raygun/')
from noise2gun import *

#specify, setup and get n2g object:
data_path = '/n/groups/htem/ESRF_id16a/tomo_ML/ReducedAnglesXray/CARE/mCTX/450p_stacks/mCTX_17keV_30nm_512c_first256.zarr/volumes'
# TODO: CHANGE PATH TO BE FROM TEST VOLUME^ (probably need to make zarr)

model_name = 'noise2gun_mCTX30nm_450p_20210824'

n2g = Noise2Gun(train_source=data_path,
                voxel_size=gp.Coordinate((30, 30, 30)),
                out_path=data_path,
                model_name = model_name,
                model_path = '/n/groups/htem/ESRF_id16a/tomo_ML/ReducedAnglesXray/raygun/n2g/models/',
                tensorboard_path = '/n/groups/htem/ESRF_id16a/tomo_ML/ReducedAnglesXray/raygun/n2g/tensorboard/'+model_name,
                num_epochs = 30000,
                init_learning_rate=0.0004)

#render
n2g.render_full()

print('Rendering done.')