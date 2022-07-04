#TODO: fix args
import argparse
import socket
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Digits 5 classification')
    parser.add_argument('--run_all',
                        help='specify to run all experiments or select parameters for single experiment',
                        type=bool,
                        default=True)

    parser.add_argument('--similarity',
                        help='cosine_similarity, MMD, projected',
                        type=str,
                        default='cosine_similarity')

    parser.add_argument('--TARGET_DOMAIN',
                        help='mnistm, mnist, syn, svhn, usps',
                        type=str,
                        default='mnistm')  # -1 so that we do not overwrite config if we do not pass anything

    parser.add_argument('--SOURCE_SAMPLE_SIZE',
                        type=int,
                        default=25000)

    parser.add_argument('--TARGET_SAMPLE_SIZE',
                        type=int,
                        default=9000)

    parser.add_argument('--lambda_sparse',
                        type=float,
                        default=1e-3)

    parser.add_argument('--lambda_OLS',
                        type=float,
                        default=1e-3)

    parser.add_argument('--lambda_orth',
                        type=float,
                        default=0)

    parser.add_argument('--early_stopping',
                        type=bool,
                        default=True)

    parser.add_argument('--fine_tune',
                        type=bool,
                        default=True)

    parser.add_argument('--run',
                        type=int,
                        default=0)

    parser.add_argument('--num_domains',
                        type=int,
                        default=5)

    args = parser.parse_args()
    if args.similarity == 'None':
        args.similarity = None
    return args

args = parse_args()

# define save dir
if socket.gethostname() == 'mtec-mis-502':
    args.data_dir = '/mnt/wave/odin/digitfive/'
    args.save_dir = '/local/home/evanweenen/gdu4dg-pytorch/results/digits5/'
elif socket.gethostname() == 'mtec-im-gpu01' and os.getusername() == 'evanweenen':
    args.data_dir = '/wave/odin/digitfive/'
    args.save_dir = '/local/home/evanweenen/gdu4dg-pytorch/results/digits5/'
else:
    print("You are running the code on an undefined machine. Please add your machine and data paths to config.py")

args.save_dir += f"{args.TARGET_DOMAIN}_{args.similarity}_{'ft' if args.fine_tune else 'e2e'}/"

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)