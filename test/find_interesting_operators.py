# first run source ~/setup_plotting
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
from EffectiveTTV.EffectiveTTV.nll import fit_nll
from EffectiveTTV.EffectiveTTV.plotting import label
from EffectiveTTV.EffectiveTTV.signal_strength import load


parser = argparse.ArgumentParser(description='extended interpretation for ttV')
parser.add_argument('config', metavar='config', type=str,
                    help='a configuration file to use')
parser.add_argument('--censored', action="store_true",
                    help='remove sensitive info from tables for sharing publicly')

args = parser.parse_args()

def write(table, headers, name, **kwargs):
    with open(os.path.join(config['outdir'], '{}.txt'.format(name)), 'w') as f:
        f.write(tabulate.tabulate(table, headers=headers, **kwargs))
    with open(os.path.join(config['outdir'], '{}.tex'.format(name)), 'w') as f:
        text = tabulate.tabulate(table, headers=headers, tablefmt="latex_raw", **kwargs)
        for key, value in label.items():
            text = text.replace(' {} '.format(key), ' {} '.format(value)) + '\n'
        f.write(text)

with open(args.config) as f:
    config = yaml.load(f)

nll = fit_nll(config)
mus = np.load(os.path.join(config['outdir'], 'mus.npy'))[()]
coefficients, cross_sections = load(config)

excluded = {
    r'\text{{no effect on}} {}'.format(', '.join(config['processes'])): [],
    r'simulation problem': ['c2W', 'cT', 'cA']
}

for coefficient in mus:
    has_effect = False
    for p in config['processes']:
        s0, s1, s2 = mus[coefficient][p].coef
        if (s1 > 1e-5) or (s2 > 1e-5):
            has_effect = True
    if not has_effect and coefficient not in excluded['simulation problem']:
        excluded[r'\text{{no effect on}} {}'.format(', '.join(config['processes']))] += [coefficient]

for process in ['tt', 'H', 'DY', 'ZZ', 'WZ', 'WW']:
    excluded[r'|\mu_{{{}}} - 1| > 0.7'.format(process)] = []

extreme_mus = {}
processes = set(sum([mus[coefficient].keys() for coefficient in mus], []))
for coefficient, info in nll.items():
    print 'op is ', coefficient
    extreme_mus[coefficient] = {}
    if len(nll[coefficient]['two sigma']) > 0:
        left = min(np.array(nll[coefficient]['two sigma'])[:, 0])
        right = max(np.array(nll[coefficient]['two sigma'])[:, 1])
        extreme_mus[coefficient][(left, right)] = {}
        for process in processes:
            y = mus[coefficient][process](np.linspace(left, right, 100))
            extreme_mus[coefficient][(left, right)][process] = y[np.abs(y - 1).argmax()]  # we want farthest from 1., not max
            if process in ['tt', 'H', 'DY', 'ZZ', 'WZ', 'WW']:
                if np.abs(extreme_mus[coefficient][(left, right)][process] - 1) > 0.7:
                    excluded[r'|\mu_{{{}}} - 1| > 0.7'.format(process)] += [coefficient]

table = []
slim_table = []
excluded_table = []
for coefficient in extreme_mus:
    for left, right in extreme_mus[coefficient]:
        row = [' {} '.format(coefficient)]
        if not args.censored:
            row += ['({:.1e}, {:.1e})'.format(left, right)]
        for process in ['ttZ', 'ttW', 'ttH', 'tt', 'H', 'DY', 'ZZ', 'WZ', 'WW']:
            row.append(extreme_mus[coefficient][(left, right)][process])
        table.append(row)
        if coefficient not in sum(excluded.values(), []):
            slim_table.append(row)
        else:
            excluded_table.append(row)
for coefficient in extreme_mus:
    if len(extreme_mus[coefficient]) == 0:
        table.append([coefficient, 'out of bounds'] + ['-' for i in processes])

headers = ['coefficient']
if not args.censored:
    headers += ['$2\sigma$ range']
headers += ['$\mu_\mathrm{ext,t\overline{t}Z}$', '$\mu_\mathrm{ext,t\overline{t}W}$', '$\mu_\mathrm{ext,t\overline{t}H}$', '$\mu_\mathrm{ext,t\overline{t}}$', '$\mu_\mathrm{ext,H}$', '$\mu_\mathrm{ext,DY}$', '$\mu_\mathrm{ext,ZZ}$', '$\mu_\mathrm{ext,WZ}$', '$\mu_\mathrm{ext,WW}$']
write(slim_table, headers, 'slim_mus_at_95_CL', floatfmt='.1f')
write(table, headers, 'mus_at_95_CL', floatfmt='.1f')
write(excluded_table, headers, 'excluded_mus_at_95_CL', floatfmt='.1f')

table = []
for coefficient in extreme_mus:
    for left, right in extreme_mus[coefficient]:
        row = [' {} '.format(coefficient)]
        if not args.censored:
            row += ['({:.1e}, {:.1e})'.format(left, right)]
        for process in ['ttZ', 'ttW', 'ttH', 'WWW', 'WZZ', 'ZZZ', 'WWZ', 'VH', 'tZq', 'tHq', 'tHW', 'tttt', 'tWZ', 'tG']:
            row.append(extreme_mus[coefficient][(left, right)][process])
        if coefficient not in sum(excluded.values(), []):
            table.append(row)

headers = ['coefficient']
if not args.censored:
    headers += ['$2\sigma$ range']
headers += ['$\mu_\mathrm{ext,t\overline{t}Z}$', '$\mu_\mathrm{ext,t\overline{t}W}$', '$\mu_\mathrm{ext,t\overline{t}H}$'] + ['$\mu_\mathrm{{ext,{}}}$'.format(p) for p in ['WWW', 'WZZ', 'ZZZ', 'WWZ', 'VH', 'tZq', 'tHq', 'tHW', 'tttt', 'tWZ', 'tG']]
write(table, headers, 'background_mus', floatfmt='.1f')

table = []
print excluded
print r'\begin{align*}'
for key, values in excluded.items():
    print r'{} \quad\rightarrow\quad& {}\\'.format(key, ',\, '.join([label[x].replace('$', '') for x in values]))
    table.append([key, ',\, '.join([label[x].replace('$', '') for x in values])])
print r'\end{align*}'
write(table, ['requirement', 'eliminated coefficients'], 'eliminated')

surviving = set(mus.keys()) - set(sum(excluded.values(), []))
print 'surviving ', surviving

print excluded
with open(os.path.join(config['outdir'], 'extreme_mus.pkl'), 'w') as f:
    # pickle.dump(extreme_mus, f)
    pickle.dump(dict((k, v) for k, v in extreme_mus.items() if k not in sum(excluded.values(), [])), f)


# table = []
# for coefficient in nll:
#     for process in processes:
        # points = line.crossings(np.linspace(-100, 100, 10000), mus[coefficient][process](np.linspace(-100, 100,
        # 10000)), sensitive_mus[coefficient])
        # print 'coefficient, points ', coefficient, points
        # one_sigma = so.brentq(find_confidence_interval, 0., 1., args=(pdf, 0.68))
