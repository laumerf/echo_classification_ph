from heart_echo.Helpers import Helpers
import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

"""
This script pre-processes the videos (segmenting, scaling, etc.) and stores in a cache directory, for quicker 
loading during training. Only needs to be run 1x, or again once more data is available.
"""


parser = ArgumentParser(
    description='Preprocess videos and cache the resulting procesesd videos',
    formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--cache_dir', default='~/.heart_echo',
                    help='Path to the directory that should store the cached videos')
parser.add_argument('--scale_factor', default=0.25,
                    help='How much to scale (down) the videos, as a ratio of original size.'
                         'Note that if many scaling factors are desired, this script has to be run once for each.')
parser.add_argument('--videos_dir', default='~/.echo_videos',
                    help='Path to the directory storing the original videos')
parser.add_argument('--procs', default=3, help='Number of processes to use')


def main():
    args = parser.parse_args()
    cache_dir_per_scaling = os.path.join(os.path.expanduser(args.cache_dir), str(args.scale_factor))
    Helpers.rebuild_video_cache(cache_dir=cache_dir_per_scaling,
                                videos_dir=args.videos_dir,
                                scale_factor=args.scale_factor, procs=args.procs)


if __name__ == '__main__':
    main()
