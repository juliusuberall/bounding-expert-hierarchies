import beh.registry

from beh.config_parser import parse_sample
from beh.adapter.sample import checkpoint_samples, save_samples
from beh.adapter.wiring import pre_process
from beh.shared import setup_dirs

# We want to sample 3D geometry ourselfs and not pass right away normalized 
# and training-ready data to be able to unstretch the model output again for
# result visualization. If sampled without this script ensure all info is 
# contained in the sample file.

# We only need to preprocess everything 3D and higher. 2D data can be passed 
# directly as RGBA images.

def main():

    setup_dirs(beh.registry)
    args = parse_sample()

    x, y, size, bounds = pre_process(args.dim, args)

    checkpoint_samples(x, y, size)

    save_samples(args, x, y, size, bounds)

if __name__ == '__main__':
    main()