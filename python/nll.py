import glob
import os

import numpy as np
import scipy.signal
import tabulate

from EffectiveTTV.EffectiveTTV import line
from EffectiveTTV.EffectiveTTV.parameters import label, conversion
from EffectiveTTV.EffectiveTTV.signal_strength import load, load_mus

def fit_nll(config, transform=False, dimensionless=True):
    """Note that the best fit is not straightforward with multiple minima, see:
    https://hypernews.cern.ch/HyperNews/CMS/get/statistics/551/1.html
    """
    from root_numpy import root2array
    # TODO implement this myself to get rid of root_numpy dependency (and possibly run everything in combine cmssw)
    # TODO only run this once after running combine instead of every time I plot
    # TODO change to returning a dict with dimensionless and transformed

    # mus = load_mus(config)
    mus = np.load(os.path.join(config['outdir'], 'mus.npy'))[()]
    res = {}
    # for operator in config['operators']:
    base = os.path.join(config['outdir'], 'scans')
    for operator in [x.replace(base + '/', '').replace('.total.root', '') for x in glob.glob(base +
        '/c*total.root') + glob.glob(base + '/tc*total.root')]:
        res[operator] = {
            'label': '',
            'best fit': [],
            'one sigma': [],
            'two sigma': [],
            'transformed': False
        }

        data = root2array(os.path.join(config['outdir'], 'scans', '{}.total.root'.format(operator)))
        # make sure min point is at 0 (combine might have chosen wrong best
        # fit for offset)
        data['deltaNLL'] -= data['deltaNLL'].min()
        max_nll = (14. / 2)
        # max_nll = (30. / 2)
        # data = data[data['deltaNLL'] < max_nll]
        # _, unique = np.unique(data[operator], return_index=True)
        if data['deltaNLL'].max() > max_nll:
            while len(data[data['deltaNLL'] < max_nll][operator]) < 10:
                max_nll += 1
        data = data[data['deltaNLL'] < max_nll]
        _, unique = np.unique(data[operator], return_index=True)

        conversion_factor = 1.
        if not dimensionless:
            if operator not in conversion:
                print 'no conversion available for {}; reporting dimensionless value'.format(operator)
            else:
                conversion_factor = conversion[operator]

        x = data[unique][operator] * conversion_factor
        y = 2 * data[unique]['deltaNLL']

        if len(y) == 0:
            print 'skipping {}: no nll points found'.format(operator)
            continue

        minima = scipy.signal.argrelmin(y, order=5)
        threshold = (y[minima] - min(y)) < 0.1
        res[operator]['conversion'] = conversion_factor
        res[operator]['units'] = '' if conversion_factor == 1. else '$\ [\mathrm{TeV}^{-2}]$'
        # FIXME: adjust threshold so to get rid of 'transform' argument and transform automatically if 2 bfs
        if transform and (len(x[minima][threshold]) == 2 or config['asimov data']):
            print 'nll difference is: ', y[minima] - min(y), operator
            xi = np.linspace(x.min(), x.max(), 10000)
            total = mus[operator]['ttH'](xi) + mus[operator]['ttZ'](xi) + mus[operator]['ttW'](xi)
            offset = xi[total.argmin()] * conversion_factor
            print 'operator, new offset ', operator, offset
            def transform(i):
                return np.abs(i - offset)

            best_fit = transform(x[minima][threshold][0])
            res[operator]['best fit'] = [(best_fit, y[transform(x).argsort()][minima][threshold][0])]

            y = y[transform(x)[x < offset].argsort()]
            x = np.array(sorted(transform(x)[x < offset]))

            res[operator]['one sigma'] = [line.interval(x, y, 1.0, best_fit)] if line.interval(x, y, 1.0, best_fit) else []
            res[operator]['two sigma'] = [line.interval(x, y, 3.84, best_fit)] if line.interval(x, y, 3.84, best_fit) else []

            sign = '+' if offset < 0 else '-'
            template = r'$|{}{}|$' if conversion_factor == 1. else r'$|{}/\Lambda^2{}|$'
            if round(offset, 1) != 0:
                res[operator]['label'] = template.format(
                        label[operator].replace('$',''),
                        r' {} {:03.1f}{}'.format(sign, np.abs(offset), '\ \mathrm{TeV}^{-2}' if conversion_factor != 1. else ''))
                # res[operator]['units'] = ''
            else:
                res[operator]['label'] = template.format(label[operator].replace('$',''), '')
            res[operator]['transformed'] = True
        else:
            for xbf, ybf in zip(x[minima][threshold], y[minima][threshold]):
                res[operator]['best fit'].append((xbf, ybf))
            res[operator]['one sigma'] = []
            res[operator]['two sigma'] = []
            for xbf, ybf in zip(x[minima], y[minima]):
                if line.interval(x, y, 1.0, xbf) and line.interval(x, y, 1.0, xbf) not in res[operator]['one sigma']:
                    res[operator]['one sigma'].append(line.interval(x, y, 1.0, xbf))
                if line.interval(x, y, 3.84, xbf) and line.interval(x, y, 3.84, xbf) not in res[operator]['two sigma']:
                    res[operator]['two sigma'].append(line.interval(x, y, 3.84, xbf))
            template = r'${}$' if conversion_factor == 1. else r'${}/\Lambda^2$'
            res[operator]['label'] = template.format(label[operator].replace('$',''))

        res[operator]['x'] = x
        res[operator]['y'] = y

    table = []
    for operator, info in res.items():
        table.append([
            info['label'],
            ', '.join(['{:.1f}'.format(round(x[0], 2) + 0) for x in info['best fit']]),
            ' and '.join(['[{:.1f}, {:.1f}]'.format(i, j) for i, j in info['one sigma']]),
            ' and '.join(['[{:.1f}, {:.1f}]'.format(i, j) for i, j in info['two sigma']])
        ])
    headers = ['Wilson coefficient', 'best fit', '$1\sigma$ CL', '$2\sigma$  CL']
    tag = '{}{}'.format(('_transformed' if transform else ''), ('_dimensionless' if info['conversion'] == 1. else ''))
    with open(os.path.join(config['outdir'], 'best_fit{}.txt'.format(tag)), 'w') as f:
        f.write(tabulate.tabulate(table, headers=headers))
    with open(os.path.join(config['outdir'], 'best_fit{}.tex'.format(tag)), 'w') as f:
        f.write(tabulate.tabulate(table, headers=headers, tablefmt='latex_raw'))

    np.save('nll{}{}.npy'.format('_transformed' if transform else '', '_dimensionless' if dimensionless else ''), res)

    return res

