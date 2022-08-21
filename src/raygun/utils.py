def passing_locals(local_dict):
    kwargs = {}
    for k, v in local_dict.items():
        if k[0] != '_' and k != 'self':
            if k == 'kwargs':
                kwargs.update(v)
            else:
                kwargs[k] = v
    return kwargs