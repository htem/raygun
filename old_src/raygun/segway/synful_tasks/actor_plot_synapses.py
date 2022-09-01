import sys
from synapses_plots import *

if __name__ == "__main__":

    if len(sys.argv) == 3:
        print("Plotting neurons : ", sys.argv[2])

        config_file = sys.argv[1]
        neuron_id = sys.argv[2]
        # define DB to access
        db_host = "mongodb://10.117.28.250:27018/"
        db_name = "production_cb2_v3_synapses_v3"
        db_name_n = "neurondb_cb2_v3" 
        db_host_n = "mongodb://10.117.28.250:27018/"   

        score_th_list = [0.6,0.7,0.65]

        if neuron_id != 'all':
            with open(config_file) as f:
                neurons = json.load(f)
                params = neurons[neuron_id]
                for key, item in params.items():
                    vars()[key] = tuple(item)

            SP = SynapsesPlots(db_name,
                       db_host,
                       db_name_n,
                       db_host_n,
                       neuron_id,
                       voxel_size_zyx,
                       upper_left_xyz,
                       lower_right_xyz,
                       )

            SP.plot_shape()

            for score_threshold in score_th_list:
                print("## Info:: plotting for score score_threshold = ", score_threshold)
                # density scatter plot
                z = SP.scatter_plot_synsonshape_2d(score_threshold)
                # density scatter plot for random dist
                z_r = SP.scatter_plot_synsonshape_2d(score_threshold, random=True)
                # colors of scatter: area
                z_a = SP.scatter_plot_synsonshape_2d(score_threshold,colors='area')
                # colors of scatter: distance between pre and post partners
                z_d = SP.scatter_plot_synsonshape_2d(score_threshold,colors='dist')
                # # scatter 3D
                # SP.scatter_plot_3d(score_threshold)
                # # GIF
                # SP.create_gif_scatter3d(score_threshold)
                # Histogram density syn vs random
                y1sr, y2sr = SP.plot_histograms(score_threshold)
                # Histogram area and distance
                y1ad, y2ad = SP.plot_histograms(score_threshold, mode='area_dist')
                # cumulative histogtram area synapses
                SP.plot_cumulative_hist(SP.areas, score_threshold)
                # x and  y coordinates histograms random vs synapses
                SP.plot_hist_xy_synvsrand(score_threshold)
                # Statistical difference of density random vs not random
                print("Statistical difference of density : synapses vs random")
                SP.stat_diff_ttest()
                # Statistical difference of x coordinate random vs not random
                print("Statistical difference of x coordinate : synapses vs random")
                SP.stat_diff_ttest(mode='x')
                # Statistical difference of y coordinate random vs not random
                print("Statistical difference of y coordinate : synapses vs random")
                SP.stat_diff_ttest(mode='y')
                # compare 2 variables (e.g. histograms output)
                print("Statistical difference of histogram counts : synapses vs random")
                SP.stat_diff_ttest(y1=y1sr,y2=y2sr)

        elif neuron_id == 'all':
            print("WARNING: the plots will be done for all the morphologies declared in the code!")
            
            neuron_list = ['purkinje_13.den_0', 'purkinje_13.den_1',
                    'purkinje_1.den0', 'nonoverlap_purkinje_13.den_0']

            for nid in neuron_list:
                with open(config_file) as f:
                    neurons = json.load(f)
                    params = neurons[nid]
                    for key, item in params.items():
                        vars()[key] = tuple(item)

                print("## Info:: plotting neuron ", nid)

                SP = SynapsesPlots(db_name,
                           db_host,
                           db_name_n,
                           db_host_n,
                           nid,
                           voxel_size_zyx,
                           upper_left_xyz,
                           lower_right_xyz,
                           )

                SP.plot_shape()

                for score_threshold in score_th_list:
                    print("## Info:: plotting for score score_threshold = ", score_threshold)

                    # density scatter plot
                    z = SP.scatter_plot_synsonshape_2d(score_threshold)
                    # density scatter plot for random dist
                    z_r = SP.scatter_plot_synsonshape_2d(score_threshold, random=True)
                    # colors of scatter: area
                    z_a = SP.scatter_plot_synsonshape_2d(score_threshold,colors='area')
                    # colors of scatter: distance between pre and post partners
                    z_d = SP.scatter_plot_synsonshape_2d(score_threshold,colors='dist')
                    # # scatter 3D
                    # SP.scatter_plot_3d(score_threshold)
                    # # GIF
                    # SP.create_gif_scatter3d(score_threshold)
                    # Histogram density syn vs random
                    y1sr, y2sr = SP.plot_histograms(score_threshold)
                    # Histogram area and distance
                    y1ad, y2ad = SP.plot_histograms(score_threshold, mode='area_dist')
                    # cumulative histogtram area synapses
                    SP.plot_cumulative_hist(SP.areas, score_threshold)
                    # x and  y coordinates histograms random vs synapses
                    SP.plot_hist_xy_synvsrand(score_threshold)
                    # Statistical difference of density random vs not random
                    print("Statistical difference of density : synapses vs random")
                    SP.stat_diff_ttest()
                    # Statistical difference of x coordinate random vs not random
                    print("Statistical difference of x coordinate : synapses vs random")
                    SP.stat_diff_ttest(mode='x')
                    # Statistical difference of y coordinate random vs not random
                    print("Statistical difference of y coordinate : synapses vs random")
                    SP.stat_diff_ttest(mode='y')
                    # compare 2 variables (e.g. histograms output)
                    print("Statistical difference of histogram counts : synapses vs random")
                    SP.stat_diff_ttest(y1=y1sr,y2=y2sr)

                    
    else:
        print("ERROR: number of arguments must be 2: config_file and neuron_id")
        exit(1)

