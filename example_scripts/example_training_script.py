# This is an example of a wrapper script one can use to train a Noise2Void UNet
# on a dataset using Noise2Gun.
# It is helpful to save scripts, such as this one, alongside the data in order
# to keep a complete record of the training/rendering process performed.
# Can be ran uninterrupted on server with: "nohup python training_script.py &"
# and monitored with: "watch tail nohup.out"
import sys
sys.path.append('/n/groups/htem/users/jlr54/raygun/')
from noise2gun import *

#specify, setup and get n2g object:
data_path = '/n/groups/htem/ESRF_id16a/tomo_ML/ReducedAnglesXray/CARE/mCTX/450p_stacks/mCTX_17keV_30nm_512c_first256.zarr/volumes'

model_name = 'noise2gun_mCTX30nm_450p_20210824'

n2g = Noise2Gun(train_source=data_path,
                voxel_size=gp.Coordinate((30, 30, 30)),
                out_path=data_path,
                model_name = model_name,
                model_path = '/n/groups/htem/ESRF_id16a/tomo_ML/ReducedAnglesXray/raygun/n2g/models/',
                tensorboard_path = '/n/groups/htem/ESRF_id16a/tomo_ML/ReducedAnglesXray/raygun/n2g/tensorboard/'+model_name,
                num_epochs = 30000,
                init_learning_rate=0.0004)

#train
n2g.build_training_pipeline()
n2g.train()

print('Training done.')