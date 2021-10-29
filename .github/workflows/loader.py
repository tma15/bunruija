import sys

import yaml


with open(sys.argv[1]) as f:
    data = yaml.load(f, yaml.SafeLoader)