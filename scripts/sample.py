from beh.config_parser import parse_sample
from beh.adapter.sample import checkpoint_samples, save_samples
from beh.adapter.wiring import sample

def main():

    # Check that all registry paths and folder exists to catch errors before code runs

    args = parse_sample()

    x, y, size = sample(args.dim, args)

    checkpoint_samples(x, y, size)

    save_samples(args, x, y, size)

if __name__ == '__main__':
    main()