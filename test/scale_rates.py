import argparse
import logging
import os
import pickle

import numpy as np
import tabulate
import yaml

import CombineHarvester.CombineTools.ch as ch
from EffectiveTTV.EffectiveTTV import line
from EffectiveTTV.EffectiveTTV.signal_strength import load, load_mus
from EffectiveTTV.EffectiveTTV.nll import fit_nll
from EffectiveTTV.EffectiveTTV.plotting import label
from EffectiveTTV.EffectiveTTV.parameters import names, process_groups


parser = argparse.ArgumentParser(description='extended interpretation for ttV')
parser.add_argument('config', metavar='config', type=str,
                    help='a configuration file to use')

args = parser.parse_args()

with open(args.config) as f:
    config = yaml.load(f)

with open(os.path.join(config['outdir'], 'extreme_mus.pkl'), 'rb') as f:
    extreme_mus = pickle.load(f)

coefficients, cross_sections = load(config)
mus = load_mus(config)

cb = ch.CombineHarvester()
cb.ParseDatacard('/afs/crc.nd.edu/user/a/awoodard/releases/effective-ttV/CMSSW_7_4_7/src/EffectiveTTV/EffectiveTTV/data/cards/20Jan_2017/2L.txt', '', '', '2lss')
cb.ParseDatacard('/afs/crc.nd.edu/user/a/awoodard/releases/effective-ttV/CMSSW_7_4_7/src/EffectiveTTV/EffectiveTTV/data/cards/0212_ttHseparated/3L.txt', '', '', '3l')
cb.ParseDatacard('/afs/crc.nd.edu/user/a/awoodard/releases/effective-ttV/CMSSW_7_4_7/src/EffectiveTTV/EffectiveTTV/data/cards/0212_ttHseparated/4L.txt', '', '', '4l')


processes = ['ttZ', 'ttW', 'ttH', 'ZZ', 'WZ', 'ttX', 'rare', 'charge', 'fake']

for operator, info in extreme_mus.items():
    print 'operator ', operator
    table = []
    for bin in cb.cp().bin_set():
        for (low, high), mus in info.items():
            row = [cb.cp().bin([bin]).channel_set()[0], bin]
            rates = {}
            for process in cb.cp().bin([bin]).process_set():
                rates[names[process]] = cb.cp().bin([bin]).process([process]).GetRate()
            scale = {}
            for process in processes:
                scale[process] = 1.
                if process in process_groups:
                    for p in process_groups[process]:
                        scale[process] *= mus[p]
                else:
                    scale[process] = mus[process]
            for process in processes:
                if process in rates:
                    if (round(rates[process], 2) == round(rates[process] * scale[process], 2)):
                        row.append('{:-10.2f}'.format(rates[process]))
                    else:
                        row.append('{:-10.2f} ({:.2f})'.format(rates[process], rates[process] * scale[process]))
                else:
                    row.append('-')
            table.append(row)
    table.append(['-'] * (len(processes) * 2 + 1))
    with open(os.path.join(config['outdir'], '{}.txt'.format(operator)), 'w') as f:
        f.write(tabulate.tabulate(table, headers=['ch', 'bin'] + ['{}'.format(p) for p in processes]))

for operator, info in extreme_mus.items():
    print 'operator ', operator
    rates = {}
    scale = {}
    sums = {}
    for channel in cb.cp().channel_set():
        rates[channel] = {}
        scale[channel] = {}
        sums[channel] = {'ttZ/W/H': 0, 'backgrounds': 0, 'scaled ttZ/W/H': 0, 'scaled backgrounds': 0}
        for (low, high), mus in info.items():  #FIXME assuming one 2 sigma band for now
            for process in cb.cp().channel([channel]).process_set():
                rates[channel][names[process]] = cb.cp().channel([channel]).process([process]).GetRate()
            for process in processes:
                scale[channel][process] = 1.
                if process in process_groups:
                    for p in process_groups[process]:
                        scale[channel][process] *= mus[p]
                        print '{} {}: scaling by {:.1f}'.format(operator, p, mus[p])
                else:
                    scale[channel][process] = mus[process]
                if process in rates[channel]:
                    if process in ['ttZ', 'ttW', 'ttH']:
                        sums[channel]['ttZ/W/H'] += rates[channel][process]
                        sums[channel]['scaled ttZ/W/H'] += rates[channel][process] * scale[channel][process]
                    else:
                        sums[channel]['backgrounds'] += rates[channel][process]
                        sums[channel]['scaled backgrounds'] += rates[channel][process] * scale[channel][process]
    table = []
    for process in processes:
        row = [process]
        for channel in cb.cp().channel_set():
            if process in rates[channel]:
                row.append('{:-10.1f}'.format(rates[channel][process]))
                row.append('{:-10.1f}'.format(rates[channel][process] * scale[channel][process]))
            else:
                row.extend(['-', '-'])
        table.append(row)
    row = ['sum(ttZ/W/H)']
    for channel in cb.cp().channel_set():
        row.append('{:.1f}'.format(sums[channel]['ttZ/W/H']))
        row.append('{:.1f}'.format(sums[channel]['scaled ttZ/W/H']))
    table.append(row)
    row = ['sum(others)']
    for channel in cb.cp().channel_set():
        row.append('{:.1f}'.format(sums[channel]['backgrounds']))
        row.append('{:.1f}'.format(sums[channel]['scaled backgrounds']))
    table.append(row)
    table.append(['-'] * 7)
    with open(os.path.join(config['outdir'], '{}.integrated.txt'.format(operator)), 'w') as f:
        text = r'\begin{centering}' + operator + r'\\\vspace{.5cm}' + r'\end{centering}'
        text += tabulate.tabulate(
                table,
                headers=['process', '2lss', '2lss scaled', '3l', '3l scaled', '4l', '4l scaled'],
                tablefmt='latex'
        )
        for key in label:
            text = text.replace(key, label[key])

        f.write(text)

print '\, '.join([label[operator] for operator in extreme_mus])
