import argparse

from bnn_competition.testing import Tester
from load import load


def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--scalex2_model_path", dest="scalex2_model_path", type=str, default=None, help="# path to scalex2 model"
    )
    parser.add_argument(
        "--scalex4_model_path", dest="scalex4_model_path", type=str, default=None, help="# path to scalex4 model"
    )

    args = parser.parse_args()
    return args


def test(scalex2_model_path, scalex4_model_path):
    scalex2_model = load(scalex2_model_path, scale=2)
    scalex4_model = load(scalex4_model_path, scale=4)

    print(Tester().test(scalex2_model, scalex4_model))


if __name__ == "__main__":
    args = parse_args()
    test(args.scalex2_model_path, args.scalex4_model_path)
