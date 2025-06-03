def set_env(config: dict, server: str = None) -> dict:
    """
    server 파라미터나 config의 server 값을 기반으로 환경 설정을 합니다.
    
    Args:
        config (dict): 설정 정보를 담고 있는 딕셔너리
        server (str, optional): 서버 정보. Defaults to None.
    
    Returns:
        dict: 업데이트된 config 딕셔너리
    """
    # server 설정
    if server is not None:
        config['server'] = server
    elif 'server' not in config:
        raise ValueError("server must be provided either as a parameter or in config")
    
    # server에 따른 환경 설정
    server = config['server']
    
    # data_root 설정
    config['data_root'] = f"/path/to/{server}/data"  # 실제 경로로 수정 필요
    
    # 추가적인 환경 설정들을 여기에 추가할 수 있습니다
    # 예: config['cache_dir'] = f"/path/to/{server}/cache"
    #     config['output_dir'] = f"/path/to/{server}/output"
    
    return config 