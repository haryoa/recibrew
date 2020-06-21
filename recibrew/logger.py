

def get_logger(name='recibrew'):
    """
    Get Logger to print something eleganlty
    :param name: the logger name
    :return: Logger object 
    """
    import logging
    logger = logging.getLogger(name)
    c_handler = logging.StreamHandler()
    c_handler.setLevel(logging.INFO)
    c_format = logging.Formatter('[%(name)s] - [%(levelname)s] || %(message)s')
    c_handler.setFormatter(c_format)
    logger.addHandler(c_handler)
    logger.setLevel(logging.INFO)
    return logger
