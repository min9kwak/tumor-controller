import argparse
import os
from typing import Union

from dotenv import load_dotenv


def set_env(config: Union[dict, argparse.Namespace], server: str = None) -> Union[dict, argparse.Namespace]:
    load_dotenv()

    data_root = os.getenv('DATA_ROOT')
    cache_dir = os.getenv('CACHE_DIR')

    if not data_root or not cache_dir:
        raise EnvironmentError(
            "DATA_ROOT and CACHE_DIR must be defined in .env file."
        )

    def _set(key, value):
        if isinstance(config, dict):
            config[key] = value
        else:
            setattr(config, key, value)

    _set('data_root', data_root)
    _set('cache_dir', cache_dir)

    return config
