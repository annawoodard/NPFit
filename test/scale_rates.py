import argparse
import imp
import os
import pickle
from collections import defaultdict

import numpy as np
import tabulate

import CombineHarvester.CombineTools.ch as ch
from EffectiveTTV.EffectiveTTV.plotting import label
from EffectiveTTV.EffectiveTTV.parameters import names
from EffectiveTTV.EffectiveTTV.signal_strength import load


parser = argparse.ArgumentParser(description='extended interpretation for ttV')
parser.add_argument('config', metavar='config', type=str,
                    help='a configuration file to use')

args = parser.parse_args()

prefixes = {
    '2l': 'ch2_',
    '3l': 'ch1_ch1_',
    '4l': 'ch1_ch2_'
}

process_groups = {
    'ttX': ['tZq', 'tttt', 'tHq', 'tHW', 'tWZ'],
    'ttother': ['tZq', 'tttt', 'tHq', 'tHW', 'tWZ'],
    'rare': ['WZZ', 'ZZZ', 'WWW', 'WWZ'],
    'charge': ['tt', 'DY'],
    'fake': ['tt', 'DY'],
    'Fake': ['tt', 'DY'],
    'WZ': ['WZ'],
    'ZZ': ['ZZ'],
    'ttH': ['ttH'],
    'ttZ': ['ttZ'],
    'ttW': ['ttW'],
}
config = imp.load_source('', args.config).config

with open(os.path.join(config['outdir'], 'extreme_mus.pkl'), 'rb') as f:
    extreme_mus = pickle.load(f)

mus = np.load(os.path.join(config['outdir'], 'mus.npy'))[()]
coefficients, cross_sections = load(config)

cb = ch.CombineHarvester()
cb.ParseDatacard(os.path.join(config['outdir'], '2l.txt'), '', '', '2lss')
cb.ParseDatacard(os.path.join(config['outdir'], '3l.txt'), '', '', '3l')
cb.ParseDatacard(os.path.join(config['outdir'], '4l.txt'), '', '', '4l')

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
                numerator = sum([mus[p] * cross_sections[p]['sm'] for p in process_groups[process]])
                denominator = sum([cross_sections[p]['sm'] for p in process_groups[process]])
                scale[process] = numerator / denominator
            for process in processes:
                if process in rates:
                    if (round(rates[process], 2) == round(rates[process] * scale[process], 2)):
                        row.append('{:-10.2f}'.format(rates[process]))
                    else:
                        row.append('{:-10.2f} ({:.2f})'.format(rates[process], rates[process] * scale[process]))
                else:
                    row.append('-')
            table.append(row)
    with open(os.path.join(config['outdir'], '{}.txt'.format(operator)), 'w') as f:
        f.write(tabulate.tabulate(table, headers=['ch', 'bin'] + ['{}'.format(p) for p in processes]))

if os.path.isfile(os.path.join(config['outdir'], 'all.integrated.tex')):
    os.remove(os.path.join(config['outdir'], 'all.integrated.tex'))
diff = {}
for operator, info in extreme_mus.items():
    print 'operator ', operator
    rates = {}
    scale = {}
    sums = {}
    diff[operator] = defaultdict(int)
    for channel in cb.cp().channel_set():
        rates[channel] = {}
        scale = {}
        sums[channel] = {'\ttZ/W/H': 0, 'backgrounds': 0, 'scaled \ttZ/W/H': 0, 'scaled backgrounds': 0}
        for (low, high), mus in info.items():
            for process in cb.cp().channel([channel]).process_set():
                rates[channel][names[process]] = cb.cp().channel([channel]).process([process]).GetRate()
            for process in processes:
                print '{} {} {} {} {} {}'.format('channel', 'operator', 'process', 'sm xsec', 'xsec * mu', 'additional yield')
                for p in process_groups[process]:
                    if process in rates[channel]:
                        new = rates[channel][process] * mus[p] * cross_sections[p]['sm'] / sum([cross_sections[i]['sm'] for i in process_groups[process]])
                        old = rates[channel][process] * cross_sections[p]['sm'] / sum([cross_sections[i]['sm'] for i in process_groups[process]])
                        diff[operator][p] += new - old
                        y = '{:.2f}'.format(new - old)
                    else:
                        y = '-'
                    print '{} {} {} {:.2f} {:.2f} {}'.format(
                        channel,
                        operator,
                        p,
                        cross_sections[p]['sm'],
                        mus[p] * cross_sections[p]['sm'],
                        y)

                numerator = sum([mus[p] * cross_sections[p]['sm'] for p in process_groups[process]])
                denominator = sum([cross_sections[p]['sm'] for p in process_groups[process]])
                scale[process] = numerator / denominator
                if process in rates[channel]:
                    if process in ['ttZ', 'ttW', 'ttH']:
                        sums[channel]['\ttZ/W/H'] += rates[channel][process]
                        sums[channel]['scaled \ttZ/W/H'] += rates[channel][process] * scale[process]
                    else:
                        sums[channel]['backgrounds'] += rates[channel][process]
                        sums[channel]['scaled backgrounds'] += rates[channel][process] * scale[process]
    table = []
    for process in processes:
        row = [process]
        for channel in cb.cp().channel_set():
            if process in rates[channel]:
                row.append('{:-10.1f}'.format(rates[channel][process]))
                row.append('{:-10.1f}'.format(rates[channel][process] * scale[process]))
            else:
                row.extend(['-', '-'])
        table.append(row)
    row = ['sum(ttZ/W/H)']
    for channel in cb.cp().channel_set():
        row.append('{:.1f}'.format(sums[channel]['\ttZ/W/H']))
        row.append('{:.1f}'.format(sums[channel]['scaled \ttZ/W/H']))
    table.append(row)
    row = ['sum(others)']
    for channel in cb.cp().channel_set():
        row.append('{:.1f}'.format(sums[channel]['backgrounds']))
        row.append('{:.1f}'.format(sums[channel]['scaled backgrounds']))
    table.append(row)
    with open(os.path.join(config['outdir'], '{}.integrated.tex'.format(operator)), 'w') as f:
        text = r"""
        \begin{{frame}}{{{} yields}}
        \resizebox{{\linewidth}}{{!}}{{
        """.format(label[operator])
        text += tabulate.tabulate(
            table,
            headers=['process', '2lss', '2lss scaled', '3l', '3l scaled', '4l', '4l scaled'],
            tablefmt='latex_raw'
        ) + """
        }
        \end{frame}
        """

        for key in label:
            text = text.replace(key, label[key])

        f.write(text)
    with open(os.path.join(config['outdir'], 'all.integrated.tex'), 'a') as f:
        f.write(text + '\n')

table = []
headers = ['coefficient', '$\mathrm{t\overline{t}Z}$', '$\mathrm{t\overline{t}W}$', '$\mathrm{t\overline{t}H}$'] + ['$\mathrm{{{}}}$'.format(p) for p in ['WWW', 'WZZ', 'ZZZ', 'WWZ', 'VH', 'tZq', 'tHq', 'tHW', 'tttt', 'tWZ', 'tG', 'WZ', 'ZZ']] + ['sum(rare + $\mathrm{t\overline{t}X + diboson}$']
for operator in diff:
    row = [label[operator]]
    for process in ['ttZ', 'ttW', 'ttH', 'WWW', 'WZZ', 'ZZZ', 'WWZ', 'VH', 'tZq', 'tHq', 'tHW', 'tttt', 'tWZ', 'tG', 'WZ', 'ZZ']:
        row += ['{:.1f}'.format(diff[operator][process])]
    row += ['{:.1f}'.format(sum([diff[operator][p] for p in ['WWW', 'WZZ', 'ZZZ', 'WWZ', 'VH', 'tZq', 'tHq', 'tHW', 'tttt', 'tWZ', 'tG', 'WZ', 'ZZ']]))]
    table.append(row)
with open(os.path.join(config['outdir'], 'yield_diffs.tex'), 'w') as f:
    f.write(tabulate.tabulate(table, headers=headers, tablefmt='latex_raw', floatfmt='.1f'))

print '\, '.join([label[operator] for operator in extreme_mus])
