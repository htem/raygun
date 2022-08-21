import numpy as np
import daisy
from daisy import Coordinate
import sys
sys.path.insert(0, '/n/groups/htem/temcagt/datasets/cb2/segmentation/tri/repos/funlib.show.neuroglancer')
sys.path.insert(0, '/n/groups/htem/Segmentation/tmn7/segwaytool.proofreading')
import segwaytool.proofreading
import segwaytool.proofreading.neuron_db_server
from database_synapses import SynapseDatabase
from database_superfragments import SuperFragmentDatabase, SuperFragment
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from funlib.segment.arrays import replace_values
from scipy.interpolate import interpn
import matplotlib.cm as cm
from scipy.ndimage.filters import gaussian_filter
from scipy.stats import gaussian_kde
from scipy import stats
from PIL import Image
import random
import json
from animation import *

class SynapsesPlots(object):
    """docstring for SynapsesPlots"""
    def __init__(self,
                 db_name,
                 db_host,
                 db_name_n,
                 db_host_n,
                 neuron_id,
                 voxel_size,
                 upper_left_xyz,
                 lower_right_xyz):

        
        super(SynapsesPlots, self).__init__()
        self.db_name = db_name
        self.db_host = db_host
        self.db_name_n = db_name_n
        self.db_host_n = db_host_n
        self.neuron_id = neuron_id
        self.voxel_size = voxel_size
        self.upper_left_xyz = upper_left_xyz
        self.lower_right_xyz = lower_right_xyz

        self.syn_db, self.sf_db, self.neuron_db = self.__connect_DBs()
        self.sup_ds, self.sfs_dict = self._get_superfragments()
        self.roi = self.__get_snapped_roi()
        self.shape_array = self._generate_shape_array()

    def __connect_DBs(self):

        syn_db = SynapseDatabase(
            db_name= self.db_name,
            db_host= self.db_host,
            db_col_name='synapses',
            )

        sf_db = SuperFragmentDatabase(
            db_name= self.db_name,
            db_host= self.db_host,
            db_col_name='superfragments',
            )

        neuron_db = segwaytool.proofreading.neuron_db_server.NeuronDBServer(
                    db_name= self.db_name_n,
                    host= self.db_host_n,
                    )
        neuron_db.connect()

        return syn_db, sf_db, neuron_db


    def _get_superfragments(self):

        superfragments = self.neuron_db.get_neuron(self.neuron_id).to_json()['segments']
        superfragments_list = [int(item) for item in superfragments]
        super_fragments_file = "/n/vast/htem/Segmentation/cb2_v3/output.zarr"
        super_fragments_dataset = "volumes/super_1x2x2_segmentation_0.400_mipmap/s4"
        sup_ds = daisy.open_ds(super_fragments_file, super_fragments_dataset, 'r')
        sfs_dict = self.sf_db.read_superfragments(sf_ids=superfragments_list)
        
        return sup_ds, sfs_dict


    def __get_snapped_roi(self):

        self.upper_left_xyz = (self.upper_left_xyz[0]*self.voxel_size[2],
                      self.upper_left_xyz[1]*self.voxel_size[1],
                      self.upper_left_xyz[2]*self.voxel_size[0])

        self.lower_right_xyz = (self.lower_right_xyz[0]*self.voxel_size[2],
                      self.lower_right_xyz[1]*self.voxel_size[1],
                      self.lower_right_xyz[2]*self.voxel_size[0])

        ul_zyx = [
                    self.upper_left_xyz[2],
                    self.upper_left_xyz[1],
                    self.upper_left_xyz[0],
                ]
        lr_zyx = [
                    self.lower_right_xyz[2],
                    self.lower_right_xyz[1],
                    self.lower_right_xyz[0],
                ]

        ul_zyx = daisy.Coordinate(ul_zyx)
        lr_zyx = daisy.Coordinate(lr_zyx)

        roi = daisy.Roi(ul_zyx, lr_zyx-ul_zyx)
        #roi = roi.snap_to_grid(sup_ds.voxel_size, 'grow')
        roi = roi.snap_to_grid((80, 2048, 2048), 'grow')
        print("ROI : ", roi)

        return roi


    def __to_xyz(self,
                 zyx):

        return (zyx[2]/self.sup_ds.voxel_size[2], 
                zyx[1]/self.sup_ds.voxel_size[1],
                zyx[0]/self.sup_ds.voxel_size[0])


    def query_synapses_by_score(self, score_threshold):

        synapses = []
        for sf in self.sfs_dict:
            synapses.extend(sf['syn_ids'])

        query = { '$and' : [{'id' : { '$in' : synapses }}, 
                {'score': { '$gt': score_threshold }} ]}

        syn_cur = self.syn_db.synapses.find(query)
        items = []
        for item in syn_cur:
            items.append(item)


        return items

    def __build_location_dict(self, syn_dict):
        loc_dict_xyz = dict()
        for syn in syn_dict:
            loc_dict_xyz[syn['id']] = self.__to_xyz([syn['z'],syn['y'],syn['x']])

        return loc_dict_xyz


    def _generate_synapses_xyz(self, score_threshold):

        syn_dict = self.query_synapses_by_score(score_threshold)

        dic_xyz = np.array(list(self.__build_location_dict(syn_dict).values())) 
        x = dic_xyz[:,0]
        y = dic_xyz[:,1]
        z = dic_xyz[:,2]

        print("Number of synapses : ", len(syn_dict))
        print("Normalized density of synapses: ", len(syn_dict)/np.sum(self.shape_array))
        
        return x, y, z

    def _generate_random_xy(self, score_threshold):

        n_synapses = len(self.query_synapses_by_score(score_threshold))
        i = 0
        x_r = []
        y_r = []
        
        while i < n_synapses:
            y = int(random.random()*self.shape_array.shape[0])
            x = int(random.random()*self.shape_array.shape[1])

            if self.shape_array[y, x] == 1:
                coord = daisy.Coordinate((0, y, x))
                coord = coord * self.sup_ds.voxel_size
                coord = self.roi.get_begin() + coord
                coord = np.array(coord).astype(float)
                coord = coord/self.sup_ds.voxel_size
                x_r.append(coord[2])
                y_r.append(coord[1])
                
                i += 1

        return x_r, y_r

    def _generate_shape_array(self):

        superfragments_list = []
        for sf in self.sfs_dict:
            superfragments_list.append(sf['id'])
        roi_ds = self.sup_ds[self.roi]
        # roi_ds.materialize()  # cache to memory
        print("Caching roi array ...")
        s = time.time()
        roi_array = roi_ds.to_ndarray()
        print("Cached roi_array in", (time.time()-s))

        shape_array = None
        for z in range(len(roi_array)):
            array = roi_array[z]
            #### ZEROSSS
            replace_vals = [1 for n in superfragments_list]
            array = replace_values(array, superfragments_list, replace_vals,
                                   inplace=True)
            np.place(array, array !=1, 0)

            if shape_array is None:
                shape_array = array
            else:
                shape_array = shape_array | array

        return shape_array

    def plot_shape(self):

        #zyx_min = self.roi.get_begin()/ self.voxel_size
        #zyx_max = self.roi.get_end()/self.voxel_size

        fig = plt.figure(figsize=(24,16))
        ax = fig.add_subplot(111)
        ax.imshow(self.shape_array,
                  #extent=[zyx_min[2],zyx_max[2],zyx_max[1],zyx_min[1]],
                  cmap='Greys' )
        print("# Info:: Shape plotted")
        fig.savefig("segway/synful_tasks/figures/%s_shape2d.png" % self.neuron_id)
        plt.close(fig)

    def _compute_color(self, mode):
        """mode is a dictionary with all the info needed to set the
           colors of the scatter plot.
           keys = {'type', 'x', 'y', 'syn_dict'}
           """
        # mode = ['density', 'area', 'dist']    
        if mode['type'] == 'density':
            xy = np.vstack([mode['x'],mode['y']])
            z = np.array(gaussian_kde(xy)(xy))
            z *= 1e+5
        elif mode['type'] == 'area':
            z = []
            for syn in mode['syn_dict']:
                pre = np.array([syn['pre_x'],syn['pre_y'],syn['pre_z']])
                post = np.array([syn['post_x'],syn['post_y'],syn['post_z']])
                z.append(np.linalg.norm(pre-post))

        elif mode['type'] == 'dist':
            z = []
            for syn in mode['syn_dict']:
                z.append(syn['area'])

        return z

    def __scatter(self, x, y, z, ax, **kwargs):

        sc = ax.scatter( x, y, c=z, **kwargs )
        cbar = plt.colorbar(sc, ax= ax)


    def scatter_plot_synsonshape_2d(self, 
                                    score_threshold, 
                                    colors='density',
                                    random=False):

        """colors = ['density', 'area', 'dist']"""

        zyx_min = self.roi.get_begin()/ self.sup_ds.voxel_size
        zyx_max = self.roi.get_end()/self.sup_ds.voxel_size
        fig = plt.figure(figsize=(24,16))
        ax = fig.add_subplot(111)
        ax.imshow(self.shape_array,
                  extent=[zyx_min[2],zyx_max[2],zyx_max[1],zyx_min[1]],
                  cmap='Greys' )
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

        if random:
            x,y = self._generate_random_xy(score_threshold)
            self.x_r = x
            self.y_r = y
            r = 'random'
        else:
            x,y,_ = self._generate_synapses_xyz(score_threshold)
            self.x = x
            self.y = y
            r = ''

        mode_dict = dict()
        mode_dict['type'] = colors

        if colors=='density':
            
            mode_dict['x'] = x
            mode_dict['y'] = y

            z = self._compute_color(mode_dict)
            self.__scatter(x,y,z,ax,cmap=cm.jet)

            if random:
                output_name = '%s_scatter_density_rand_%s.png' \
                               %(self.neuron_id, score_threshold)
                self.dens_r = z

            else:
                output_name = '%s_scatter_density_%s.png' % (self.neuron_id,
                                                             score_threshold)
                self.dens = z

        elif colors=='area' or colors=='dist':
            syn_dict = self.query_synapses_by_score(score_threshold)
            mode_dict['syn_dict'] = syn_dict
            z = self._compute_color(mode_dict)
            self.__scatter(x,y,z,ax,cmap=cm.jet)

            if colors=='area':
                output_name = '%s_scatter_area_%s.png' % (self.neuron_id,
                                                      score_threshold)
                self.areas = z 
            elif colors=='dist':
                output_name = '%s_scatter_prepost_dist_%s.png' \
                                %(self.neuron_id, score_threshold)
                self.dists = z

        else:
            print("Method of colors ('density', 'area', 'dist')\
             not declared! ")
            exit()

        print("# Info:: %s scatter %s plotted" % (r, colors))
        fig.savefig("segway/synful_tasks/figures/"+output_name)
        plt.close(fig)

        return z

    def scatter_plot_3d(self, score_threshold):

        x,y,z = self._generate_synapses_xyz(score_threshold)
        fig = plt.figure(figsize=(24,16))
        ax = Axes3D(fig)
        ax.scatter(x,y,z,
                   s=30,
                   c='b',
                   edgecolors='b',
                   alpha=0.3)

        ax.invert_yaxis()
        ax.invert_zaxis()

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        fig.savefig("segway/synful_tasks/figures/%s_scatter3d_%s.png" % 
                                (self.neuron_id, score_threshold))
        print("# Info:: scatter 3D plotted")
        plt.close(fig)


    def create_gif_scatter3d(self, score_threshold):
        print("# Info:: creation of GIF ...")
        start = time.time()
        fig = plt.figure(figsize=(24,16))
        ax = Axes3D(fig)
        x,y,z = self._generate_synapses_xyz(score_threshold)
        ax.scatter(x, y, z,
                    s=30, 
                    c='b', 
                    edgecolors='k',
                    alpha=0.3)

        ax.invert_yaxis()
        ax.invert_zaxis()

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        angles = np.linspace(0,360,21)[:-1] # Take 20 angles between 0 and 360
        output = "segway/synful_tasks/figures/%s_movie_%s.gif" %(neuron_id ,score_threshold)
        rotanimate(ax, angles, output,fps=10,bitrate=2000)  
        cost = time.time()-start
        print("# Info:: GIF 3D created in %f s" % cost)

    ######    
    # Histograms and statistical analysis with respect to random
    def plot_histograms(self, 
                        score_threshold,
                        mode='density'):
        """mode represents the color in the scatter: 
        mode = ['density', 'area_dist']
        - density will compare random vs synaptic density
        - area_dist will plot area vs distance btw partners
        """
        fig = plt.figure(figsize=(24,16))
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)

        if mode=='density':
            ax1.set_title("Synapses density")
            ax2.set_title("Random density")
            output_name = '%s_hist_density_%s.png' % (self.neuron_id,
                                                      score_threshold)

            y1,_,_ = ax1.hist(x=self.dens, bins='auto',
                          color='b',
                          alpha=0.7, 
                          rwidth=0.85)
       

            y2,_,_ = ax2.hist(x=self.dens_r, bins='auto',
                          color='r',
                          alpha=0.7, 
                          rwidth=0.85)


        elif mode=='area_dist':
            ax1.set_title("Area values")
            ax2.set_title("Distance partners values")
            output_name = '%s_hist_area_dist_%s.png' % (self.neuron_id,
                                                      score_threshold)

            y1,_,_ = ax1.hist(x=self.areas, bins='auto',
                          color='b',
                          alpha=0.7, 
                          rwidth=0.85)
       

            y2,_,_ = ax2.hist(x=self.dists, bins='auto',
                          color='r',
                          alpha=0.7, 
                          rwidth=0.85)
            
        ax1.set_xlabel("Values")
        ax1.set_ylabel("Frequency")

        ax2.set_xlabel("Values")
        ax2.set_ylabel("Frequency")

        print("# Info:: %s histograms created" % mode)
        fig.savefig("segway/synful_tasks/figures/"+output_name)
        plt.close(fig)

        return y1, y2

    def plot_cumulative_hist(self, z, score_threshold):

        fig = plt.figure(figsize=(24,16))
        ax = fig.add_subplot(111)

        ax.hist(z, bins='auto',
                 color='#0504aa',
                 alpha=0.7 , rwidth=0.85,
                 cumulative=True,density=True)
        
        ax.set_ylabel('Normalized Frequency')
        ax.set_xlabel("Values")
        ax.grid(axis='y', alpha=0.75)

        print("# Info:: cumulative histograms created")
        fig.savefig("segway/synful_tasks/figures/%s_cumulative_hist_%s.png" 
                                                    % (self.neuron_id,
                                                       score_threshold))
        plt.close(fig)
        
    
    def plot_hist_xy_synvsrand(self, score_threshold):
        fig = plt.figure(figsize=(24,16))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        ax1.hist(self.x,bins='auto',alpha=0.7,label='Synapses', color='b', rwidth=0.85)
        ax1.hist(self.x_r,bins='auto',alpha=0.6,label='Random', color='r', rwidth=0.85)

        ax2.hist(self.y,bins='auto',alpha=0.7,label='Synapses', color='b', rwidth=0.85)
        ax2.hist(self.y_r,bins='auto',alpha=0.6,label='Random', color='r', rwidth=0.85)

        ax1.set_title("X coordinate")
        ax2.set_title("Y coordinate")

        ax1.set_xlabel("Values")
        ax1.set_ylabel("Frequency")

        ax2.set_xlabel("Values")
        ax2.set_ylabel("Frequency")

        ax2.legend()
        print("# Info:: XY coordinates histograms (syns vs random) created")
        fig.savefig("segway/synful_tasks/figures/%s_hist_synvsrandom_%s.png" 
                                                    % (self.neuron_id,
                                                       score_threshold))
        plt.close(fig)

    def stat_diff_ttest(self,mode='density', y1=None,y2=None):
        """
        mode = ['density', 'x', 'y']
        y1 and y2 could be:
           - density colors of synapses vs random
             from the scatter plot_shape_and_synapses_2d
           - x coordinates synapses vs random
           - y coordinates synapses vs random           
        Also potentially:
           - histograms outputs synapses vs random but y1 and y2 must
             be given
        """
        print("#### T-TEST ####")
        if y1 is None and y2 is None:
            if mode == 'density':
                t, p = stats.ttest_ind(self.dens,self.dens_r)
            elif mode == 'x':
                t, p = stats.ttest_ind(self.x,self.x_r)
            elif mode == 'y':
                t, p = stats.ttest_ind(self.y,self.y_r)    
            else:
                print("ERROR: mode specified not valid, please use\
                    density, x or y")
                exit()

        else:
            t, p = stats.ttest_ind(y1,y2)


        print("Results: t stat = %f, p-value = %f" %(t,p))
        if p<0.05:
            print("The 2 distributions are different!")
        else:
            print("Nothing can be said on the difference between the 2 distributions!")


# if __name__ == '__main__':
#     if len(sys.argv) == 3:
#         print("Reading the neuron %s ROI" % sys.argv[2])

#         config_file = sys.argv[1]
#         neuron_id = sys.argv[2]

#         with open(config_file) as f:
#             neurons = json.load(f)
#             params = neurons[neuron_id]
#             for key, item in params.items():
#                 vars()[key] = tuple(item)

#     else:
#         print("ERROR: number of arguments must be 2: config_file and neuron_id")
#         exit(1)

#     # neuron/segment
#     db_host = "mongodb://10.117.28.250:27018/"
#     db_name = "production_cb2_v3_synapses_v3"
#     # Synapses and superfragments 
#     db_name_n = "neurondb_cb2_v3" 
#     db_host_n = "mongodb://10.117.28.250:27018/"

#     score_th_list = [0.6,0.7,0.65]

#     SP = SynapsesPlots(db_name,
#                        db_host,
#                        db_name_n,
#                        db_host_n,
#                        neuron_id,
#                        voxel_size_zyx,
#                        upper_left_xyz,
#                        lower_right_xyz,
#                        )
#     score_threshold = score_th_list[0]

#     SP.plot_shape()
#     # density scatter plot
#     z = SP.scatter_plot_synsonshape_2d(score_threshold)
#     # density scatter plot for random dist
#     z_r = SP.scatter_plot_synsonshape_2d(score_threshold, random=True)
#     # colors of scatter: area
#     z_a = SP.scatter_plot_synsonshape_2d(score_threshold,colors='area')
#     # colors of scatter: distance between pre and post partners
#     z_d = SP.scatter_plot_synsonshape_2d(score_threshold,colors='dist')
#     # scatter 3D
#     SP.scatter_plot_3d(score_threshold)
#     # GIF
#     SP.create_gif_scatter3d(score_threshold)
#     # Histogram density syn vs random
#     y1sr, y2sr = SP.plot_histograms(score_threshold)
#     # Histogram area and distance
#     y1ad, y2ad = SP.plot_histograms(score_threshold, mode='area_dist')
#     # cumulative histogtram area synapses
#     SP.plot_cumulative_hist(SP.areas, score_threshold)
#     # x and  y coordinates histograms random vs synapses
#     SP.plot_hist_xy_synvsrand(score_threshold)
#     # Statistical difference of density random vs not random
#     SP.stat_diff_ttest()
#     # Statistical difference of x coordinate random vs not random
#     SP.stat_diff_ttest(mode='x')
#     # Statistical difference of y coordinate random vs not random
#     SP.stat_diff_ttest(mode='y')
#     # compare 2 variables (e.g. histograms output)
#     SP.stat_diff_ttest(y1=y1sr,y2=y2sr)
