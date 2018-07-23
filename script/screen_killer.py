from screenutils import list_screens
import argparse, re

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--re', help='RegExp to select dying screens.', default='*')
args = parser.parse_args()

rule = re.compile(args.re)

for s in list_screens():
    if rule.match(s):
        s.kill()
