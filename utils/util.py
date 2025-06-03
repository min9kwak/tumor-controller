def set_env(config: dict, server: str = None) -> dict:
    # set server
    if server is not None:
        config['server'] = server
    elif 'server' not in config:
        raise ValueError("server must be provided either as a parameter or in config")    
    server = config['server']
    
    assert server in ['local', 'psc']
    
    # set data root
    if server == 'local':
        config['data_root'] = 'D:/data/tumor-controller'
    elif server == 'psc':
        config['data_root'] = '/ocean/projects/med230010p/mkwak/data/tumor-controller'
    else:
        raise ValueError(f"Invalid server: {server}")
    
    return config
