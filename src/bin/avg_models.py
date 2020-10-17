import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import argparse

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('paths', nargs='+')
    parser.add_argument('--output', required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    ds = []
    for path in args.paths:
        d = torch.load(path)
        ds.append(d)

    out = {}
    for key in ds[0]:
        out[key] = sum([v[key] for v in ds]) / len(ds)

    torch.save(out, args.output)


if __name__ == '__main__':
    main()
