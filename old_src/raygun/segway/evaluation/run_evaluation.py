import argparse
import json
import os
from task_05_graph_evaluation_print_error import compare_segmentation_to_ground_truth_skeleton, generate_error_plot, color_generator
from operator import add


# Consider altering task_defaults/configs to reflect actual method parameters
def parse_configs(path):
    global_configs = {}
    # first load default configs if avail
    try:
        default_filepath = os.path.dirname(os.path.realpath(__file__))
        config_file = default_filepath + '/' + 'task_defaults.json'
        with open(config_file, 'r') as f:
            global_configs = json.load(f)
    except Exception:
        print("Default task config not loaded")
        pass
    with open(path, 'r') as f:
        print("\nloading provided config %s" % path)
        new_configs = json.load(f)
        keys = set(list(global_configs.keys())).union(list(new_configs.keys()))
        for k in keys:
            if k in global_configs:
                if k in new_configs:
                    global_configs[k].update(new_configs[k])
            else:
                global_configs[k] = new_configs[k]
    print("\nconfig loaded")
    return global_configs


def construct_name_mapping(paths, names):
    d = {}
    for p, n in zip(paths, names):
        d[p] = n
    return d


# clean this method up and/or consider reformatting the config JSONs and task_defaults
def format_parameter_configs(config, volume, iteration):

    skeleton_configs = config['AdditionalFeatures']
    skeleton_configs['skeleton_path'] = volume['skeleton']
    error_count_configs = config['AdditionalFeatures']
    output_configs = config['Output']
    output_configs['config_JSON'] = config['file_name']
    output_configs['voxel_size'] = tuple(config['Input']['voxel_size'])
    volume_name = get_vol_name(volume, iteration)
    return {'skeleton': skeleton_configs, 'error_count': error_count_configs,
            'output': output_configs, 'name': volume_name}


def get_vol_name(vol, i):
    if "volume_name" not in vol:
        return "{}".format(i)
    else:
        return vol["volume_name"]


def get_weight(vol):
    if "weight" not in vol:
        return (1)
    else:
        return (vol["weight"])


def run_evaluation(
        config_path, num_processes, file_name):

    config = parse_configs(config_path)
    config['Output']['output_path'] = os.path.dirname(config_path)
    if 'skeleton_csv' in config['Input']:
        config['Input']['skeleton'] = config['Input']['skeleton_csv']
    elif 'skeleton_json' in config['Input']:
        config['Input']['skeleton'] = config['Input']['skeleton_json']
    elif 'skeleton' not in config['Input']:
        print("Please provide path to skeleton or check the keyword in json \
               file")

    model_name_mapping = {}
    if 'segment_names' in config['Input']:
        model_name_mapping = construct_name_mapping(
            config['Input']['segment_volumes'],
            config['Input']['segment_names'])
        print(model_name_mapping, "&&&&&")
    config['file_name'] = file_name

    if 'Inputs' in config:
        splits_and_merges = {}
        weights = []

        for num, volume in enumerate(config['Inputs']):
            
            model_name_mapping = {}
            if 'segment_names' in volume:
                model_name_mapping = construct_name_mapping(
                    volume['segment_volumes'],
                    volume['segment_names'])
                # print(model_name_mapping, "&&&&&")
            config['file_name'] = file_name

            weights.append(get_weight(volume))

            parameter_configs = format_parameter_configs(config, volume, num)

            splits_and_merges.update(
                compare_segmentation_to_ground_truth_skeleton(
                    config['Input']['segment_dataset'],
                    volume['segment_volumes'],
                    model_name_mapping,
                    num_processes,
                    parameter_configs))

        print("Splits and merges for each cutout:")
        print(splits_and_merges)

        splits_and_merges = add_weights(splits_and_merges, weights)

        print("Splits and merges for each cutout weighted:")
        print(splits_and_merges)

        try:
            splits_and_merges = [format_splits_and_merges(x)
                                 for x in zip(*splits_and_merges)]
            print("Summed splits and merges:")
            print(splits_and_merges)
        except:
            print("Could not sum, probably because only one input exists?")
            splits_and_merges = splits_and_merges[0]
            print(splits_and_merges)

        generate_error_plot(config['Input']['segment_dataset'],
                            config['file_name'], 'Combined',
                            config['Output']['output_path'],
                            config['Output']['markers'],
                            color_generator(len(volume['segment_volumes'])),
                            config['Output']['font_size'],
                            config['Output']['line_width'], 'number',
                            *splits_and_merges)
    else:
        parameter_configs = format_parameter_configs(config, config['Input'], 0)

        compare_segmentation_to_ground_truth_skeleton(
            config['Input']['segment_dataset'],
            config['Input']['segment_volumes'],
            model_name_mapping,
            num_processes,
            parameter_configs)


# Takes in dictionary, and returns a list of splits and merges ordered by volume
def add_weights(splits_and_merges, weights):

    weighted_list = []
    for i, vol in enumerate(splits_and_merges):
        weighted_volume = []

        for element in splits_and_merges[vol]:
            if isinstance(element, str):
                weighted_volume.append(element)
            else:
                new_list = [e * weights[i] for e in element]
                weighted_volume.append(new_list)

        weighted_list.append(weighted_volume)

    return(weighted_list)


def format_splits_and_merges(list_of_iterables):

    if isinstance(list_of_iterables[1], list):
        unweighted_list = (list(sum(x) for x in zip(*list_of_iterables)))
        return unweighted_list

    elif isinstance(list_of_iterables[1], str):
        if (all(x == list_of_iterables[0] for x in list_of_iterables)):
            return(list_of_iterables[0])
        else:
            return(list_of_iterables)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='provide the path to configs with input \
                                        information')
    parser.add_argument(
        '-p',
        '--processes',
        help='Number of processes to use, default to 8',
        type=int,
        default=16)
    args = parser.parse_args()
    file_name = args.config.split('/')[-1].split('.')[0]
    run_evaluation(args.config, args.processes, file_name)
