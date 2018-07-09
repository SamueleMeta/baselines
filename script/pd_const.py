'''
    Convergence test for the per-decision constant that appears in the variance bound
'''

import argparse
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--horizon', help='Horizon', type=int, default=500)
parser.add_argument('--gamma', type=float, default=0.99)
args = parser.parse_args()

s = 0
for i in range(args.horizon - 1):
    v = pow(args.gamma, 2 * i) * (1 + args.gamma - 2 * pow(args.gamma, args.horizon - i)) / (1 - args.gamma)
    print(v)
    #v = pow(args.gamma, 2 * i) + 2 * pow(args.gamma, i) * (pow(args.gamma, i+1) - pow(args.gamma, args.horizon)) / (1 - args.gamma)
    s += v

print("Sum:", s)
