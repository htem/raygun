import os


def passing_locals(local_dict):
    kwargs = {}
    for k, v in local_dict.items():
        if k[0] != '_' and k != 'self':
            if k == 'kwargs':
                kwargs.update(v)
            else:
                kwargs[k] = v
    return kwargs

def get_config_name(config_path, base_folder):
        config_name = os.path.dirname(config_path)
        config_name = config_name.replace(base_folder, '')
        config_name = '_'.join(config_name.split('/'))[1:]

        return config_name
