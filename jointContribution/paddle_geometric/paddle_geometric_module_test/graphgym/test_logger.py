from paddle_geometric.graphgym.config import set_run_dir
from paddle_geometric.graphgym.loader import create_loader
from paddle_geometric.graphgym.logger import Logger, LoggerCallback
from paddle_geometric.testing import withPackage


@withPackage('yacs', 'pytorch_lightning')
def test_logger_callback():
    loaders = create_loader()
    assert len(loaders) == 3

    set_run_dir('.')
    logger = LoggerCallback()
    assert isinstance(logger.train_logger, Logger)
    assert isinstance(logger.val_logger, Logger)
    assert isinstance(logger.test_logger, Logger)
