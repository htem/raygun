# !conda activate n2v
import numpy as np
from matplotlib import pyplot as plt
import os
import zarr
from skimage import metrics as skimet
import pandas as pd
from datetime import datetime

import gunpowder as gp
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.Logger('Compare', 'INFO')

class Compare():
    def __init__(self, 
                src_path, # 'path/to/data.zarr/volumes'
                gt_name='gt', # 'gt_dataset_name'
                ds_names=None, # ['dataset_1_name', 'dataset_2_name', ...]
                out_path=None,
                batch_size=1,
                metric_list=None,
                vizualize=False
    ):
        self.src_path = src_path
        self.gt_name = gt_name
        if ds_names is None:
            data = zarr.open(self.src_path)
            self.ds_names = [key for key in data.keys() if key!=self.gt_name]
        else:
            self.ds_names = ds_names
        self.out_path = out_path
        self.batch_size = batch_size
        if metric_list is None:
            self.metric_list = ['normalized_root_mse', 
                                'peak_signal_noise_ratio',
                                'structural_similarity',
                                # 'normalized_mutual_information'
                                ]
        else:
            self.metric_list = metric_list
        self.vizualize = vizualize
        self.make_pipes()        
        self.batch = None


    def make_pipes(self):
        self.gt = gp.ArrayKey('GT')
        self.array_dict = {self.gt: self.gt_name}
        self.normalizers = gp.Normalize(self.gt)
        for ds in self.ds_names:
            setattr(self, ds, gp.ArrayKey(ds.upper()))
            self.array_dict[getattr(self, ds)] = ds
            self.normalizers += gp.Normalize(getattr(self, ds))
        
        # setup data source
        self.source = gp.ZarrSource(
                    self.src_path,
                    self.array_dict,
                    {array: gp.ArraySpec(interpolatable=True) for array in self.array_dict}
        )

        # get voxel sizes for datasets
        self.voxel_sizes = {}
        data_file = zarr.open(self.src_path)
        for key, name in self.array_dict.items():
            self.voxel_sizes[key] = self.source._Hdf5LikeSource__read_spec(key, data_file, name).voxel_size

       # setup a cache
        self.cache = gp.PreCache(num_workers=os.cpu_count())
        
        # add a RandomLocation node to the pipeline to randomly select a sample
        self.random_location = gp.RandomLocation()

        # make Comparator for full dataset comparisons:
        self.comparator = Comparator(self.array_dict, self.gt, self.metric_list)


    def __get_batch_show(self, batch, i=0):
        gt_data = batch[self.gt].data[i].squeeze()
        if len(gt_data.shape) == 3:
            mid = gt_data.shape // 2
            gt_data = gt_data[mid]
        else:
            mid = False
        datas = [gt_data]
        labels = [self.array_dict[self.gt]]
        for array, name in self.array_dict.items():
            if array is not self.gt:
                labels.append(name)
                if mid:
                    datas.append(batch[array].data[i].squeeze()[mid])
                else:
                    datas.append(batch[array].data[i].squeeze())
        return datas, labels        


    def batch_show(self, batch=None, i=0):
        if batch is None:
            batch = self.batch
        datas, labels = self.__get_batch_show(batch, i=i)
        cols = len(datas)
        fig, axes = plt.subplots(1, cols, figsize=(30,30*cols))        
        for i, (data, label) in enumerate(zip(datas, labels)):
            axes[i].imshow(data, cmap='gray', vmin=0, vmax=1)
            axes[i].set_title(label)
        # self.results.plot.bar()
        fig, axs = plt.subplots(1, len(self.metric_list), sharey=True, figsize=(len(self.metric_list)*3, len(self.results.columns)//2))
        for ax, metric in zip(axs, self.metric_list):
            self.results.loc[metric].plot.barh(ax=ax, title=metric)


    def patch_compare(self, patch_size=gp.Coordinate((64,64,64))):
        pipeline =  (self.source + 
                    self.normalizers + 
                    self.random_location)
        
        request = gp.BatchRequest()
        for array in self.array_dict:
            request.add(array, self.voxel_sizes[array] * patch_size, voxel_size=self.voxel_sizes[array])
        
        with gp.build(pipeline):
            self.batch = pipeline.request_batch(request)
        
        results_dict = {}
        gt_data = self.batch[self.gt].data
        for array, name in self.array_dict.items():
            if array is not self.gt:
                this_data = self.batch[array].data
                these_results = {}
                for metric in self.metric_list:
                    these_results[metric] = getattr(skimet, metric)(gt_data, this_data)
                results_dict[name] = these_results
        self.results = pd.DataFrame(results_dict)
        
        if self.vizualize:
            self.batch_show()
        
        print(self.results)
        return self.results, self.batch


    def compare(self, patch_size=gp.Coordinate((64,64,64))): #TODO: FIX MISSING REGIONS ROIS
        scan_request = gp.BatchRequest()
        for array in self.array_dict:
            scan_request.add(array, self.voxel_sizes[array] * patch_size)
        scan = gp.Scan(scan_request, num_workers=self.batch_size)#os.cpu_count())
        
        pipeline =  (self.source + 
                    self.normalizers + 
                    # self.cache +
                    self.comparator +
                    scan
                    )
        
        with gp.build(pipeline):
            pipeline.request_batch(gp.BatchRequest())
        
        self.results = self.comparator.get_results()
        
        if self.vizualize:
            self.results.plot.bar()
        
        print(self.results)
        if self.out_path is not None:
            self.results.to_csv(self.out_path + 'comparator_results_' + datetime.now().strftime("%Y%m%d%H%M") + '.csv')
        return self.results


class Comparator(gp.BatchFilter):

    def __init__(self, array_dict, gt_key, metric_list=None):
        self.array_dict = array_dict #TODO: JUST INFER FROM BATCH
        self.gt = gt_key
        if metric_list is None:
            self.metric_list = ['normalized_root_mse', 
                                'peak_signal_noise_ratio',
                                'structural_similarity',
                                # 'normalized_mutual_information'
                                ]
        else:
            self.metric_list = metric_list
        self.results = None
        self.iter = 0


    def prepare(self, request):
        return request.copy() #TODO: determine if this is necessary


    def process(self, batch, request):
        results_dict = {}
        gt_data = batch[self.gt].data
        for array, name in self.array_dict.items():
            if array is not self.gt:
                this_data = batch[array].data
                these_results = {}
                for metric in self.metric_list:
                    these_results[metric] = getattr(skimet, metric)(gt_data, this_data)
                results_dict[name] = these_results
        
        if self.results is None:
            self.results = pd.DataFrame(results_dict)
        else:
            self.results += pd.DataFrame(results_dict) # TODO: DETERMINE BEST WAY TO COMBINE ACROSS SCANS...
        self.iter += 1

        return
    
    def get_results(self):
        return self.results / self.iter # TODO: DETERMINE BEST WAY TO COMBINE ACROSS SCANS... (currently average)