import logging
from functools import partial, partialmethod


def get_custom_logger(file_location):
    logging.WARNING=800
    logging.ERROR=700
    logging.INFORM=600
    logging.POSE=500
    logging.FAILURE_LIFT=400
    logging.FAILURE_GRIPPER=300
    logging.FAILURE_DROPOFF=200

    logging.addLevelName(logging.WARNING,'WARNING')
    logging.addLevelName(logging.ERROR,'ERROR')
    logging.addLevelName(logging.INFORM,'INFORM')
    logging.addLevelName(logging.POSE,'POSE')
    logging.addLevelName(logging.FAILURE_LIFT, 'FAILURE_LIFT')
    logging.addLevelName(logging.FAILURE_GRIPPER, 'FAILURE_GRIPPER')
    logging.addLevelName(logging.FAILURE_DROPOFF, 'FAILURE_DROPOFF')

    logging.Logger.warning = partialmethod(logging.Logger.log, logging.WARNING)
    logging.warning = partial(logging.log, logging.WARNING)

    logging.Logger.error = partialmethod(logging.Logger.log, logging.ERROR)
    logging.error = partial(logging.log, logging.ERROR)

    logging.Logger.inform = partialmethod(logging.Logger.log, logging.INFORM)
    logging.inform = partial(logging.log, logging.INFORM)

    logging.Logger.pose = partialmethod(logging.Logger.log, logging.POSE)
    logging.pose = partial(logging.log, logging.POSE)

    logging.Logger.failure_lift = partialmethod(logging.Logger.log, logging.FAILURE_LIFT)
    logging.failure_lift = partial(logging.log, logging.FAILURE_LIFT)

    logging.Logger.failure_gripper = partialmethod(logging.Logger.log, logging.FAILURE_GRIPPER)
    logging.failure_gripper = partial(logging.log, logging.FAILURE_GRIPPER)

    logging.Logger.failure_dropoff = partialmethod(logging.Logger.log, logging.FAILURE_DROPOFF)
    logging.failure_dropoff = partial(logging.log, logging.FAILURE_DROPOFF)

    logger = logging.getLogger()
    # logger.setLevel(logging.POSE)
    # logger.setLevel(logging.FAILURE_LIFT)
    # logger.setLevel(logging.FAILURE_GRIPPER)
    logger.setLevel(logging.FAILURE_DROPOFF)

    log_format = "%(levelname)s: %(message)s"
    formatter = logging.Formatter(log_format)

    handler = logging.FileHandler(file_location)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger

if __name__=="__main__":
    logger=get_custom_logger("./testlog.log")
    for i in range(5):
        logger.failure_lift("Failure lift log")
        logger.failure_gripper("Failure gripper log")
        logger.failure_dropoff("Failure dropoff log")
        logger.pose("Pose log")
        logger.info("Information")
        logger.debug("Debug information")
        logger.warning("Warning information")
        logger.error("Error information")
