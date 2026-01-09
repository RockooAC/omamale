import logging

def setup_logger(name="AgentLogger", debug=False):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    return logger
