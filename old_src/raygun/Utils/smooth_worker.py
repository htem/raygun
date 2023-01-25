import os
import runpy
import daisy
import torch
import numpy as np

from raygun.launchable_daisy_task import LaunchableDaisyTask

class SmoothPredictTask(LaunchableDaisyTask):

    def _init(self, config):
        self.worker_script_file = os.path.realpath(__file__)
        
        self.src_path_name = self.src_path.split('/')[-1][:-3]
    
        self.cuda_available = torch.cuda.is_available() #TODO: Get CUDA working
        if not self.cuda_available:
            print('Cuda not available.')
        # else:
        #     torch.multiprocessing.set_start_method('spawn', force=True)

        #Load Model
        script_dict = runpy.run_path(self.script_path)
        cycleGun = script_dict['cycleGun']
        if side_length is None:
            side_length = cycleGun.side_length

        #Set Dataset to Render
        self.source = daisy.open_ds(self.src_path, self.src_name)
        side = side.upper()

        if len(cycleGun.model_name.split('-')) > 1:
            parts = cycleGun.model_name.split('-')
            label_prefix = parts[0] + parts[1][1:]
        else:
            label_prefix = cycleGun.model_name
        label_prefix += f'_seed{script_dict["seed"]}'

        read_size = side_length #e.g. 64
        read_roi = daisy.Roi([0,0,0], self.source.voxel_size * read_size)
        write_size = read_size

        if self.crop: #e.g. 16
            write_size -= self.crop*2 #e.g. --> 32

        if self.smooth:
            if self.cuda_available:
                self.window = torch.cuda.FloatTensor(self.tukey(write_size, alpha=self.alpha, sym=False))
            else:
                self.window = torch.FloatTensor(self.tukey(write_size, alpha=self.alpha, sym=False)).share_memory_()
            self.window = (self.window[:, None, None] * self.window[None, :, None]) * self.window[None, None, :]
            chunk_size = int(write_size * (self.alpha / 2)) #e.g. 8
            self.write_pad = (self.source.voxel_size * write_size * (self.alpha / 2)) / 2  #e.g. voxel_size * 4
            write_size -= write_size * (self.alpha / 2) #e.g. --> 24
            write_roi = daisy.Roi(self.source.voxel_size * self.crop + self.write_pad, self.source.voxel_size * write_size)
        else:
            write_roi = daisy.Roi(self.source.voxel_size * self.crop, self.source.voxel_size * write_size)
            chunk_size = int(write_size)

        compressor = {  'id': 'blosc', 
                        'clevel': 3,
                        'cname': 'blosclz',
                        'blocksize': chunk_size
                        }
        
        net = 'netG1' if side == 'A' else 'netG2'
        other_net = 'netG1' if side == 'B' else 'netG2'

        this_checkpoint = f'{cycleGun.model_path}{cycleGun.model_name}_checkpoint_{self.checkpoint}'
        print(f"Processing checkpoint {this_checkpoint}...")
        cycleGun.checkpoint = this_checkpoint
        cycleGun.load_saved_model(this_checkpoint, self.cuda_available)
        if self.cuda_available:
            self.generator = getattr(cycleGun.model, net).cuda()
        else:
            self.generator = getattr(cycleGun.model, net).cpu()
        self.generator.eval()

        label_dict = {}
        label_dict['fake'] = f'volumes/{label_prefix}_checkpoint{self.checkpoint}_{net}{self.ds_suffix}'
        label_dict['cycled'] = f'volumes/{label_prefix}_checkpoint{self.checkpoint}_{net}{other_net}{self.ds_suffix}'
        
        total_roi = self.source.roi
        if self.total_roi_crop:
            total_roi = total_roi.grow(self.source.voxel_size * -self.total_roi_crop, self.source.voxel_size * -self.total_roi_crop)

        needs_erase = os.path.exists(f'{self.src_path}/{label_dict["fake"]}')
        self.destination = daisy.prepare_ds(
            self.src_path, 
            label_dict['fake'],
            total_roi,
            self.source.voxel_size,
            self.source.dtype,
            write_size=write_roi.get_shape() if not self.smooth else self.write_pad*2,
            compressor=compressor,
            delete=True,
            # force_exact_write_size=True
            )
        if needs_erase:
            self.destination[self.destination.roi] = 0
        
        assert False, "Fn needs to be implemented by subclass"

    def worker_function(self, block:daisy.Roi):        
        this_write = block.write_roi
        data = self.source.to_ndarray(block.read_roi)
        if self.cuda_available:
            data = torch.cuda.FloatTensor(data).unsqueeze(0).unsqueeze(0)
        else:
            data = torch.FloatTensor(data).unsqueeze(0).unsqueeze(0)
        data -= np.iinfo(self.source.dtype).min
        data /= np.iinfo(self.source.dtype).max
        data *= 2
        data -= 1
        out = self.generator(data).detach().squeeze()
        del data
        if self.crop:
            out = out[self.crop:-self.crop, self.crop:-self.crop, self.crop:-self.crop]
        out += 1.0
        out /= 2
        out *= 255 #TODO: This is written assuming dtype = np.uint8
        if self.smooth:
            out *= self.window
            this_write = this_write.grow(self.write_pad, self.write_pad)
        if self.cuda_available:
            out = out.cpu().numpy()#.astype(np.uint8)
        else:
            out = out.numpy()#.astype(np.uint8)
        self.destination[this_write] = np.uint8(self.destination.to_ndarray(this_write) + out)
        del out

    def schedule_blockwise(self):
        assert False, "Fn needs to be implemented by subclass"
