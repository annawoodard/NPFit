import argparse
import imp
import pickle
import os

import numpy as np
import tabulate

from EffectiveTTV.EffectiveTTV.nll import fit_nll
from EffectiveTTV.EffectiveTTV.plotting import label
from EffectiveTTV.EffectiveTTV.scaling import load


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


config = imp.load_source('', args.config).config

nll = fit_nll(config)
scales = np.load(os.path.join(config['outdir'], 'scales.npy'))[()]
coefficients, cross_sections = load(config)

excluded = {
    r'\text{{no effect on}} {}'.format(', '.join(config['processes'])): [],
    r'simulation problem': ['c2W', 'cT', 'cA']
}

for coefficient in scales:
    has_effect = False
    for p in config['processes']:
        s0, s1, s2 = scales[coefficient][p].coef
        if (s1 > 1e-5) or (s2 > 1e-5):
            has_effect = True
    if not has_effect and coefficient not in excluded['simulation problem']:
        excluded[r'\text{{no effect on}} {}'.format(', '.join(config['processes']))] += [coefficient]

for process in ['tt', 'H', 'DY', 'ZZ', 'WZ', 'WW']:
    excluded[r'|\mu_{{{}}} - 1| > 0.7'.format(process)] = []

extreme_scales = {}
processes = set(sum([scales[coefficient].keys() for coefficient in scales], []))
for coefficient, info in nll.items():
    print 'op is ', coefficient
    extreme_scales[coefficient] = {}
    if len(nll[coefficient]['two sigma']) > 0:
        left = min(np.array(nll[coefficient]['two sigma'])[:, 0])
        right = max(np.array(nll[coefficient]['two sigma'])[:, 1])
        extreme_scales[coefficient][(left, right)] = {}
        for process in processes:
            y = scales[coefficient][process](np.linspace(left, right, 100))
            extreme_scales[coefficient][(left, right)][process] = y[np.abs(y - 1).argmax()]  # we want farthest from 1., not max
            if process in ['tt', 'H', 'DY', 'ZZ', 'WZ', 'WW']:
                if np.abs(extreme_scales[coefficient][(left, right)][process] - 1) > 0.7:
                    excluded[r'|\mu_{{{}}} - 1| > 0.7'.format(process)] += [coefficient]

table = []
slim_table = []
excluded_table = []
for coefficient in extreme_scales:
    for left, right in extreme_scales[coefficient]:
        row = [' {} '.format(coefficient)]
        if not args.censored:
            row += ['({:.1e}, {:.1e})'.format(left, right)]
        for process in ['ttZ', 'ttW', 'ttH', 'tt', 'H', 'DY', 'ZZ', 'WZ', 'WW']:
            row.append(extreme_scales[coefficient][(left, right)][process])
        table.append(row)
        if coefficient not in sum(excluded.values(), []):
            slim_table.append(row)
        else:
            excluded_table.append(row)
for coefficient in extreme_scales:
    if len(extreme_scales[coefficient]) == 0:
        table.append([coefficient, 'out of bounds'] + ['-' for i in processes])

headers = ['coefficient']
if not args.censored:
    headers += ['$2\sigma$ range']
headers += ['$\mu_\mathrm{ext,t\overline{t}Z}$', '$\mu_\mathrm{ext,t\overline{t}W}$', '$\mu_\mathrm{ext,t\overline{t}H}$', '$\mu_\mathrm{ext,t\overline{t}}$', '$\mu_\mathrm{ext,H}$', '$\mu_\mathrm{ext,DY}$', '$\mu_\mathrm{ext,ZZ}$', '$\mu_\mathrm{ext,WZ}$', '$\mu_\mathrm{ext,WW}$']
write(slim_table, headers, 'slim_scales', floatfmt='.1f')
write(table, headers, 'scales', floatfmt='.1f')
write(excluded_table, headers, 'excluded_scales', floatfmt='.1f')

table = []
for coefficient in extreme_scales:
    for left, right in extreme_scales[coefficient]:
        row = [' {} '.format(coefficient)]
        if not args.censored:
            row += ['({:.1e}, {:.1e})'.format(left, right)]
        for process in ['ttZ', 'ttW', 'ttH', 'WWW', 'WZZ', 'ZZZ', 'WWZ', 'VH', 'tZq', 'tHq', 'tHW', 'tttt', 'tWZ', 'tG']:
            row.append(extreme_scales[coefficient][(left, right)][process])
        if coefficient not in sum(excluded.values(), []):
            table.append(row)

headers = ['coefficient']
if not args.censored:
    headers += ['$2\sigma$ range']
headers += ['$\mu_\mathrm{ext,t\overline{t}Z}$', '$\mu_\mathrm{ext,t\overline{t}W}$', '$\mu_\mathrm{ext,t\overline{t}H}$'] + ['$\mu_\mathrm{{ext,{}}}$'.format(p) for p in ['WWW', 'WZZ', 'ZZZ', 'WWZ', 'VH', 'tZq', 'tHq', 'tHW', 'tttt', 'tWZ', 'tG']]
write(table, headers, 'background_scales', floatfmt='.1f')

table = []
print excluded
print r'\begin{align*}'
for key, values in excluded.items():
    print r'{} \quad\rightarrow\quad& {}\\'.format(key, ',\, '.join([label[x].replace('$', '') for x in values]))
    table.append([key, ',\, '.join([label[x].replace('$', '') for x in values])])
print r'\end{align*}'
write(table, ['requirement', 'eliminated coefficients'], 'eliminated')

surviving = set(scales.keys()) - set(sum(excluded.values(), []))
print 'surviving ', surviving

print excluded
with open(os.path.join(config['outdir'], 'extreme_scales.pkl'), 'w') as f:
    pickle.dump(dict((k, v) for k, v in extreme_scales.items() if k not in sum(excluded.values(), [])), f)
