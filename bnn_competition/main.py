import argparse
import datetime
import os

from nip import load

import bnn_competition


def set_gpu(idx: str):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = idx


def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--config", dest="config", type=str, help="# path to config")
    parser.add_argument("--load", dest="load", type=str, default=None, help="# path to model")
    parser.add_argument("--logdir", dest="logdir", type=str, default=None)
    parser.add_argument(
        "--checkpoint_path", dest="checkpoint_path", type=str, default=None, help="# path to checkpoint"
    )
    parser.add_argument("--gpu", dest="gpu", default="0,1,2,3", help="# gpu")

    args = parser.parse_args()
    return args


def get_default_logdir():
    date_and_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = os.path.join(os.getcwd(), "logdir", date_and_time)
    return logdir


def create_task(args, logdir):
    configuration = load(args.config)

    task = configuration["task"]
    task.configure(
        logdir=logdir,
        num_gpus=len(args.gpu.split(",")),
        checkpoint_path=args.checkpoint_path,
    )

    return task


if __name__ == "__main__":
    bnn_competition.__init__(__name__)
    args = parse_args()
    logdir = args.logdir or get_default_logdir()
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    set_gpu(args.gpu)

    task = create_task(args, logdir)

    task.run()
