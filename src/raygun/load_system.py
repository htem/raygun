from importlib import import_module
import logging
logger = logging.getLogger(__name__)

from .read_config import read_config

def load_system(config_path, checkpoint=None):
    config = read_config(config_path)
    System = getattr(import_module('.'.join(['raygun', config['framework'], 'systems', config['system']])), config['system'])
    system = System(config_path)

    # system.load_saved_model(checkpoint=checkpoint)

    return system
    