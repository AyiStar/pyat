from torch.utils.tensorboard import SummaryWriter

global writer


def init_writer(**kwargs):
    global writer
    writer = SummaryWriter(**kwargs)
