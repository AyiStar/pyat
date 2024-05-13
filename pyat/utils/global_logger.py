import logging
import pyat.utils.global_config as gconfig
from . import get_tags

global logger

EXP_LEVEL = 100
logging.addLevelName(EXP_LEVEL, "EXP")


class ColorfulFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s"

    FORMATS = {
        logging.DEBUG: yellow + format + reset,
        logging.INFO: grey + format + reset,
        EXP_LEVEL: red + format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def exp(self, message, *args, **kws):
    if self.isEnabledFor(EXP_LEVEL):
        # Yes, logger takes its '*args' as 'args'.
        self._log(EXP_LEVEL, message, args, **kws)


def init_logger(cfg):
    global logger
    _, exp_tags = get_tags(gconfig.config)
    logging.Logger.exp = exp
    logger = logging.getLogger("_".join(exp_tags))
    logger.setLevel(getattr(logging, cfg.level))
    if cfg.stream_handler and not len(logger.handlers):
        sh = logging.StreamHandler()
        sh.setFormatter(ColorfulFormatter())
        sh.setLevel(logging.DEBUG)
        logger.addHandler(sh)
    if cfg.file_handler and "file_path" in cfg:
        logger_format = logging.Formatter(
            "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s"
        )
        fh = logging.FileHandler(cfg.file_path, "a", encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(logger_format)
        logger.addHandler(fh)


if __name__ == "__main__":
    print("This is a test log")
    init_logger()
    logger.info("This is a test log")
    logger.debug("This is a test log")
