import argparse
from typing import Union, List


def set_env(config: Union[dict, argparse.Namespace], server: str = None) -> Union[dict, argparse.Namespace]:
    # set get and set for dict and argparse.Namespace
    def get(key, default=None):
        if isinstance(config, dict):
            return config.get(key, default)
        else:
            return getattr(config, key, default)

    def set(key, value):
        if isinstance(config, dict):
            config[key] = value
        else:
            setattr(config, key, value)

    # set server
    if server is not None:
        set('server', server)
    elif get('server') is None:
        raise ValueError("server must be provided either as a parameter or in config")    
    server = get('server')

    assert server in ['local', 'psc']

    # set data root
    if server == 'local':
        set('data_root', 'D:/data/tumor-controller')
        set('cache_dir', 'D:/data/tumor-controller/models')
    elif server == 'psc':
        set('data_root', '/ocean/projects/med230010p/mkwak/data/tumor-controller')
        set('cache_dir', '/ocean/projects/med230010p/mkwak/models')
    
    return config
