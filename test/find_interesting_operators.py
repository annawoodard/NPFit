import argparse
import pickle
import json
import logging
import os

import numpy as np
import tabulate
import yaml
# import scipy.optimize as so

from EffectiveTTV.EffectiveTTV import line
from EffectiveTTV.EffectiveTTV.signal_strength import load, load_mus
from EffectiveTTV.EffectiveTTV.nll import fit_nll
from EffectiveTTV.EffectiveTTV.plotting import label


parser = argparse.ArgumentParser(description='extended interpretation for ttV')
parser.add_argument('config', metavar='config', type=str,
                    help='a configuration file to use')

args = parser.parse_args()

def write(table, headers, name):
    with open(os.path.join(config['outdir'], '{}.txt'.format(name)), 'w') as f:
        f.write(tabulate.tabulate(table, headers=headers))
    with open(os.path.join(config['outdir'], '{}.tex'.format(name)), 'w') as f:
        text = tabulate.tabulate(table, headers=headers, tablefmt="latex")
        for key, value in label.items():
            text = text.replace(' {} '.format(key), ' {} '.format(value))
        f.write(text)

with open(args.config) as f:
    config = yaml.load(f)

nll, _ = fit_nll(config)
coefficients, cross_sections = load(config)
mus = load_mus(config)

extreme_mus = {}
processes = ['tt', 'H', 'DY', 'ZZ', 'WZ', 'WW', 'ttH', 'ttW', 'ttZ', 'WZZ', 'ZZZ', 'WWW', 'WWZ', 'tZq']
for operator, info in nll.items():
    print 'op is ', operator
    extreme_mus[operator] = {}
    for low, high in info['two sigma']:
        print 'found two sigma for ', operator
        extreme_mus[operator][(low, high)] = {}
        for process in processes:
            print mus.keys()
            print mus[operator].keys()
            y = mus[operator][process](np.linspace(low, high, 100))
            extreme_mus[operator][(low, high)][process] = y[np.abs(y - 1).argmax()]  # we want farthest from 1., not max

excluded = {r'|c_j| < 10': []}
for process in ['tt', 'H', 'DY', 'ZZ', 'WZ', 'WW']:
    excluded[r'\mu_{{{}}}(c_j) < 1.3'.format(process)] = []

for operator in nll:
    print(extreme_mus[operator])
    if len(extreme_mus[operator]) == 0:
        excluded[r'|c_j| < 10'].append(operator)
    for low, high in extreme_mus[operator]:
        for process in ['tt', 'H', 'DY', 'ZZ', 'WZ', 'WW']:
            if (extreme_mus[operator][(low, high)][process] > 1.3) and (operator not in excluded['\mu_{{{}}}(c_j) < 1.3'.format(process)]):
                excluded['\mu_{{{}}}(c_j) < 1.3'.format(process)].append(operator)

table = []
slim_table = []
for operator in nll:
    if len(extreme_mus[operator]) == 0:
        continue
        table.append([operator, 'out of bounds'] + ['-' for i in processes])
    for low, high in extreme_mus[operator]:
        row = [' {} '.format(operator), '({:.1f}, {:.1f})'.format(low, high)]
        for process in processes:
            row.append('{:0.2f}'.format(extreme_mus[operator][(low, high)][process]))
        table.append(row)
        if operator not in sum(excluded.values(), []):
            slim_table.append(row)
for operator in nll:
    if len(extreme_mus[operator]) == 0:
        table.append([operator, 'out of bounds'] + ['-' for i in processes])
slim_table.append(['-', '-'] + ['-' for i in processes])

write(slim_table, ['operator', '$2\sigma$ range'] + processes, 'slim_mus_at_95_CL')
write(table, ['operator', '$2\sigma$ range'] + processes, 'mus_at_95_CL')

print r'\begin{align*}'
for key, values in excluded.items():
    print r'{} \quad\rightarrow\quad& {}\\'.format(key, ',\, '.join([label[x] for x in values]))
print r'\end{align*}'

with open(os.path.join(config['outdir'], 'extreme_mus.pkl'), 'w') as f:
    pickle.dump(dict((k, v) for k, v in extreme_mus.items() if k not in sum(excluded.values(), [])), f)


# table = []
# for operator in nll:
#     for process in processes:
        # points = line.crossings(np.linspace(-100, 100, 10000), mus[operator][process](np.linspace(-100, 100, 10000)), sensitive_mus[operator])
        # print 'operator, points ', operator, points
        # one_sigma = so.brentq(find_confidence_interval, 0., 1., args=(pdf, 0.68))
