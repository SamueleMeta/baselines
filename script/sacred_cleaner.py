'''
    Clean the sacred runs directory depending on many parameters.
'''

import json, re, argparse, glob, shutil, os

def load_runs(base_directory):
    if base_directory[-1] != '/':
        base_directory += '/'
    runs = {}
    runs_filenames = glob.glob(base_directory + '*/config.json')
    run_extractor = re.compile(base_directory + '([0-9]+)/config.json')
    for r in runs_filenames:
        try:
            run_number = int(run_extractor.match(r).group(1))
            runs[run_number] = {}
            runs[run_number]['config'] = json.load(open(base_directory + str(run_number) + '/config.json'))
            runs[run_number]['run'] = json.load(open(base_directory + str(run_number) + '/run.json'))
            runs[run_number]['metrics'] = json.load(open(base_directory + str(run_number) + '/metrics.json'))
        except:
            del runs[run_number]
    return runs

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dir', help='Directory of the sacred runs.', default='sacred_runs')
parser.add_argument('--uncompleted', default=False, action='store_true')
parser.add_argument('--cout', default=False, action='store_true')
args = parser.parse_args()

my_runs = load_runs(args.dir)

base_directory = args.dir
if base_directory[-1] != '/':
    base_directory += '/'

for key, value in my_runs.items():
    if value['run']['status'] != 'COMPLETED' and args.uncompleted:
        # Remove run with key
        shutil.rmtree(base_directory + str(key) + '/')
        print("Removed run:", key)
    elif args.cout:
        # Try to remove the cout file
        os.remove(base_directory + str(key) + '/cout.txt')
        print("Removed cout for run:", key)

print("Completed")
