from diffrax import LightField

def get_lf_prop(prop, data):
    for key, val in data.items():
        if isinstance(val, LightField):
            data[key] = getattr(val, prop)
        # otherwise assume already correct
    
    return data