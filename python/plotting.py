import atexit
import contextlib
from datetime import datetime
import glob
import json
import logging
import os
import shutil
import tarfile
import tabulate

import jinja2
import matplotlib
from matplotlib.mlab import griddata
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'

import numpy as np
import ROOT
import scipy.optimize as so
from scipy.stats import kde
from scipy.stats import gaussian_kde

from EffectiveTTV.EffectiveTTV.parameters import nlo, kappa, label, conversion
from EffectiveTTV.EffectiveTTV import kde
from EffectiveTTV.EffectiveTTV.signal_strength import load, load_mus
from EffectiveTTV.EffectiveTTV.nll import fit_nll
from EffectiveTTV.EffectiveTTV.fluctuate import par_names


extent = (0, 1500, 0, 1600)
x_min, x_max, y_min, y_max = extent


class Plotter(object):

    def __init__(self, config):
        atexit.register(self.webify)
        self.config = config

    def webify(self):
        path = os.path.join(os.environ['LOCALRT'], 'src', 'EffectiveTTV', 'EffectiveTTV', 'data')
        env = jinja2.Environment(loader=jinja2.FileSystemLoader(path))
        env.filters["datetime"] = lambda d: datetime.fromtimestamp(d).strftime('%a, %d %b %Y, %H:%M')
        env.filters["basename"] = lambda d: os.path.basename(d)
        env.tests["sum"] = lambda s: s == "Total"
        template = env.get_template('template.html')

        tfilename = os.path.join(self.config['outdir'], 'plots', 'plots.tar.gz')
        tfile = tarfile.open(tfilename, 'w:gz')
        for root, dirs, files in os.walk(os.path.abspath(self.config['outdir'])):
            for path in [os.path.join(root, d) for d in dirs] + [root]:
                pngs = glob.glob(os.path.join(path, '*.png'))
                pdfs = glob.glob(os.path.join(path, '*.pdf'))
                plots = [os.path.splitext(os.path.basename(name))[0] for name in pngs]
                if plots:
                    plots.sort()
                    [tfile.add(f, arcname=os.path.basename(name)) for f in [os.path.join(path, name) for name in pdfs]]

                dirs = ['..']
                files = []
                names = [x for x in os.listdir(path) if not x.startswith('.') and '#' not in x and x != 'index.html']
                for name in names:
                    if '.png' not in name and '.pdf' not in name:
                        fullname = os.path.join(path, name)
                        if os.path.isdir(fullname):
                            if name not in ['javascripts', 'stylesheets']:
                                dirs.append(name)
                        else:
                            files.append({'name': name, 'timestamp': os.path.getmtime(fullname)})

                with open(os.path.join(path, 'index.html'), 'w') as f:
                    logging.info('updating index {}'.format(os.path.join(path, 'index.html')))
                    f.write(template.render(
                        dirs=sorted(dirs),
                        files=sorted(files),
                        plots=plots,
                    ).encode('utf-8'))

        tfile.close()

    @contextlib.contextmanager
    def saved_figure(self, x_label, y_label, name, header=False):
        fig, ax = plt.subplots(figsize=(11, 11))
        lumi = str(self.config['luminosity']) + ' fb$^{-1}$ (13 TeV)'
        if header:
            plt.title(lumi, loc='right', fontweight='normal')
            plt.title(r'CMS', loc='left', fontweight='bold')
            if header == 'preliminary':
                plt.text(0.155, 1.009, r'Preliminary', style='italic', transform=ax.transAxes)

        try:
            yield ax

        finally:
            logging.info('saving {}'.format(name))
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.savefig(os.path.join(self.config['outdir'], 'plots', '{}.pdf'.format(name)), bbox_inches='tight')
            plt.savefig(os.path.join(self.config['outdir'], 'plots', '{}.png'.format(name)), bbox_inches='tight')
            plt.close()

def mu_new(config, plotter, overlay_results=False, dimensionless=False):
    coefficients, cross_sections = load(config)
    mus = load_mus(config)

    for operator in config['operators']:
        scale = 1 if dimensionless else conversion[operator]
    # for operator, xmin, xmax in [('cuW', -5, 5), ('cuB', -5, 5), ('cu', -30, 30), ('cHu', -7, 7)]:
        if operator == 'sm':
            continue

        data = []
        with plotter.saved_figure(
                label[operator] + ('' if dimensionless else r' $/\Lambda^2$'),
                '$\sigma_{NP+SM} / \sigma_{SM}$',
                os.path.join('mu', operator + ('_overlay' if overlay_results else ''))) as ax:

            c_max = (4 * np.pi) ** 2
            print 'processes ', mus[operator].keys()
            try:
                xmin = max([(mus[operator][p] - 5).roots().min() for p in ['ttW', 'ttZ', 'ttH'] if (mus[operator][p](c_max) > 5)] + [-1 * c_max])
                xmax = min([(mus[operator][p] - 5).roots().max() for p in ['ttW', 'ttZ', 'ttH'] if (mus[operator][p](c_max) > 5)] + [c_max])
            except KeyError:
                continue
            # xmin = -160
            # xmax = 160
            xi = np.linspace(xmin, xmax, 10000)
            for process, marker in [('ttW', 'x'), ('ttZ', '+'), ('ttH', 'o')]:
                print operator
                print process, (mus[operator][process] - 5).roots()
                x = coefficients[process][operator]
                y = cross_sections[process][operator] / cross_sections[process]['sm']

                print operator, xmin, xmax
                ax.plot(xi * scale, mus[operator][process](xi), color='#C6C6C6')
                ax.plot(x * scale, y, marker, mfc='none', markeredgewidth=2, markersize=15, label=label[process])

            if overlay_results:
                nll, units = fit_nll(config, transform=False, dimensionless=True)
                colors = ['black', 'gray']
                for (x, _), color in zip(nll[operator]['best fit'], colors):
                    plt.axvline(
                        x=x,
                        ymax=0.5,
                        linestyle='-',
                        color=color,
                        label='best fit\n {}$={:.2f}$'.format(label[operator], x)
                    )
                for (low, high), color in zip(nll[operator]['one sigma'], colors):
                    plt.axvline(
                        x=low,
                        ymax=0.5,
                        linestyle='--',
                        color=color,
                        label='$1 \sigma [{:03.2f}, {:03.2f}]$'.format(low, high)
                    )
                    plt.axvline(
                        x=high,
                        ymax=0.5,
                        linestyle='--',
                        color=color
                    )
                for (low, high), color in zip(nll[operator]['two sigma'], colors):
                    plt.axvline(
                        x=low,
                        ymax=0.5,
                        linestyle=':',
                        color=color,
                        label='$2 \sigma [{:03.2f}, {:03.2f}]$'.format(low, high)
                    )
                    plt.axvline(
                        x=high,
                        ymax=0.5,
                        linestyle=':',
                        color=color
                    )

            plt.xlim(xmin=xmin * scale, xmax=xmax * scale)
            plt.ylim(ymin=0, ymax=5)
            plt.title(r'CMS simulation', loc='left', fontweight='bold')
            plt.title(r'mg5_aMC LO', loc='right', size='small')
            ax.legend(loc='upper center')

def mu(config, plotter, overlay_results=False, dimensionless=False):

    nll, units = fit_nll(config, transform=False, dimensionless=True)
    coefficients, cross_sections = load(config)
    mus = load_mus(config)

    # for operator in config['operators']:
    for operator, xmin, xmax in [('cuW', -5, 5), ('cuB', -14, 14), ('cu', -30, 30), ('cHu', -10, 10)]:
        scale = 1 if dimensionless else conversion[operator]
        print 'dimensionless, scale ', dimensionless, scale
        if operator == 'sm':
            continue

        data = []
        with plotter.saved_figure(
                label[operator] + ('' if dimensionless else r' $/\Lambda^2$'),
                '$\sigma_{NP+SM} / \sigma_{SM}$',
                os.path.join('mu', operator + ('_overlay' if overlay_results else ''))) as ax:

            # xmin = min(np.array(nll[operator]['two sigma'])[:, 0])
            # xmax = max(np.array(nll[operator]['two sigma'])[:, 1])

            # xmin = xmin - (np.abs(xmin) * 0.1)
            # xmax = xmax + (np.abs(xmax) * 0.1)
            # xmin = min(coefficients['ttZ'][operator]) #FIXME remove
            # xmax = max(coefficients['ttZ'][operator])
            for process, marker, c in [('ttW', 'x', 'blue'), ('ttZ', '+', '#2fd164'), ('ttH', 'o', '#ff321a')]:
                x = coefficients[process][operator]
                y = cross_sections[process][operator] / cross_sections[process]['sm']
                # above = lambda low: x >= low
                # below = lambda high: x <= high
                # while len(x[above(xmin) & below(xmax)]) < 3:
                #     xmin -= np.abs(xmin) * 0.1
                #     xmax += np.abs(xmax) * 0.1
                # if operator == 'cHu':
                #     xmin = -12. / scale
                #     xmax = 3. / scale
                xi = np.linspace(xmin, xmax, 10000)

                ax.plot(xi * scale, mus[operator][process](xi), color='#C6C6C6')
                ax.plot(x * scale, y, marker, mfc='none', markeredgewidth=2, markersize=15, label=label[process],
                        color=c)

            if overlay_results:
                colors = ['black', 'gray']
                for (x, _), color in zip(nll[operator]['best fit'], colors):
                    plt.axvline(
                        x=x,
                        ymax=0.5,
                        linestyle='-',
                        color=color,
                        label='best fit\n {}$={:.2f}$'.format(label[operator], x)
                    )
                for (low, high), color in zip(nll[operator]['one sigma'], colors):
                    plt.axvline(
                        x=low,
                        ymax=0.5,
                        linestyle='--',
                        color=color,
                        label='$1 \sigma [{:03.2f}, {:03.2f}]$'.format(low, high)
                    )
                    plt.axvline(
                        x=high,
                        ymax=0.5,
                        linestyle='--',
                        color=color
                    )
                for (low, high), color in zip(nll[operator]['two sigma'], colors):
                    plt.axvline(
                        x=low,
                        ymax=0.5,
                        linestyle=':',
                        color=color,
                        label='$2 \sigma [{:03.2f}, {:03.2f}]$'.format(low, high)
                    )
                    plt.axvline(
                        x=high,
                        ymax=0.5,
                        linestyle=':',
                        color=color
                    )

            print operator, 'scale ', scale, xmin, xmax
            plt.xlim(xmin=xmin, xmax=xmax)
            plt.ylim(ymin=0, ymax=5)
            plt.title(r'CMS simulation', loc='left', fontweight='bold')
            # plt.title(r'aMC@NLO_Madgraph5 LO', loc='right', fontweight='bold')
            plt.title(r'MG5_aMC@NLO LO', loc='right', size='medium')
            ax.legend(loc='upper center')


def nll(args, config, plotter, transform=False, dimensionless=True):
    data, units = fit_nll(config, transform, dimensionless)
    units = '' if dimensionless else '\ [{}]'.format(units)

    for operator, info in data.items():
        if transform:
            template = r'$|{}{}|${}' if dimensionless else r'$|{}/\Lambda^2{}|{}$'
        else:
            template = r'${}{}${}' if dimensionless else r'${}/\Lambda^2{}{}$'

        x_label = template.format(label[operator].replace('$',''), info['offset label'], units)

        with plotter.saved_figure(
                x_label,
                '$-2\ \Delta\ \mathrm{ln}\ \mathrm{L}$' + (' (asimov data)' if config['asimov data'] else ''),
                os.path.join('nll', ('transformed' if transform else ''), operator + ('_dimensionless' if
                    dimensionless else '')),
                header=args.header) as ax:
            ax.plot(info['x'], info['y'], 'o', color='black')

            for x, y in info['best fit']:
                plt.axvline(
                    x=x,
                    ymax=0.5,
                    linestyle='-',
                    color='black',
                    label='best fit',
                )
            for i, (low, high) in enumerate(info['one sigma']):
                ax.plot([low, high], [1.0, 1.0], '--', label=r'$1\sigma$ CL' if (i==0) else '', color='blue')
            for i, (low, high) in enumerate(info['two sigma']):
                ax.plot([low, high], [3.84, 3.84], ':', label=r'$2\sigma$ CL' if (i==0) else '', color='#ff321a')

            ax.legend(loc='upper center')
            plt.ylim(ymin=0, ymax=12)
            if transform:
                plt.xlim(xmin=0)

def ttZ_ttW_2D(config, ax):
    from root_numpy import root2array
    # limits = root2array(os.path.join('scans', 'ttZ_ttW_2D.total.root'))
    # FIXME make contours optional
    limits = root2array(os.path.join(config['outdir'], 'best-fit-2d.root'))

    x = limits['r_ttW'] * nlo['ttW']
    y = limits['r_ttZ'] * nlo['ttZ']
    z = 2 * limits['deltaNLL']

    # levels = {
    #     2.30: '  1 $\sigma$',
    #     5.99: '  2 $\sigma$',
    #     # 11.83: ' 3 $\sigma$',
    #     # 19.33: ' 4 $\sigma$',
    #     # 28.74: ' 5 $\sigma$'
    # }

    # xi = np.linspace(x_min, x_max, 1000)
    # yi = np.linspace(y_min, y_max, 1000)
    # zi = griddata(x, y, z, xi, yi, interp='linear')

    # cs = plt.contour(xi, yi, zi, sorted(levels.keys()), colors='black', linewidths=2)
    # plt.clabel(cs, fmt=levels)

    handles = []
    labels = []

    bf, = plt.plot(
        x[z.argmin()],
        y[z.argmin()],
        color='black',
        mew=3,
        markersize=17,
        marker="+",
        linestyle='None'
    )
    handles.append(bf)
    labels.append('2D best fit')

    ttW_theory_xsec = plt.axvline(x=nlo['ttW'], linestyle='--', color='black')
    ttW_theory_error = ax.axvspan(
        nlo['ttW'] - nlo['ttW'] * 0.1173,
        nlo['ttW'] + nlo['ttW'] * 0.1316,
        edgecolor='#555555',
        fill=False,
        linewidth=0.0,
        zorder=2,
        hatch='//'
    )
    handles.append((ttW_theory_xsec, ttW_theory_error))
    labels.append('{} theory'.format(label['ttW']))

    ttZ_theory_xsec = plt.axhline(y=nlo['ttZ'], linestyle='--', color='black')
    ttZ_theory_error = ax.axhspan(
        nlo['ttZ'] - nlo['ttZ'] * 0.1164,
        nlo['ttZ'] + nlo['ttZ'] * 0.10,
        edgecolor='#555555',
        fill=False,
        linewidth=0.0,
        zorder=2,
        hatch='\\'
    )
    handles.append((ttZ_theory_xsec, ttZ_theory_error))
    labels.append('{} theory'.format(label['ttZ']))

    ax.set_xlim([x_min, x_max])
    ax.set_autoscalex_on(False)

    return handles, labels


def ttZ_ttW_2D_1D_ttZ_1D_ttW(args, config, plotter):
    from root_numpy import root2array
    with plotter.saved_figure(
            label['sigma ttW'],
            label['sigma ttZ'],
            'ttZ_ttW_2D_1D_ttZ_1D_ttW',
            header=args.header) as ax:
        handles, labels = ttZ_ttW_2D(config, ax)

        data = root2array('ttW.root')

        ttW_1D_xsec = plt.axvline(x=data['limit'][0] * nlo['ttW'], color='black')
        ttW_1D_error = ax.axvspan(
            data['limit'][1] * nlo['ttW'],
            data['limit'][2] * nlo['ttW'],
            alpha=0.5,
            color='#FA6900',
            linewidth=0.0
        )
        handles.append((ttW_1D_xsec, ttW_1D_error))
        labels.append('{} 1D $\pm$ $1\sigma$'.format(label['ttW']))

        data = root2array('ttZ.root')

        ttZ_1D_xsec = plt.axhline(y=data['limit'][0] * nlo['ttZ'], color='black')
        ttZ_1D_error = ax.axhspan(
            data['limit'][1] * nlo['ttZ'],
            data['limit'][2] * nlo['ttZ'],
            color='#69D2E7',
            alpha=0.5,
            linewidth=0.0
        )
        handles.append((ttZ_1D_xsec, ttZ_1D_error))
        labels.append('{} 1D $\pm$ $1\sigma$'.format(label['ttZ']))

        plt.legend(handles, labels)

def ttZ_ttW_2D_1D_eff_op(args, config, plotter, transform=False, dimensionless=True):
    if config['asimov data']:
        transform=False
    nll, units = fit_nll(config, transform, dimensionless)

    table = []
    for operator in config['operators']:
        data = np.load(os.path.join(config['outdir'], 'fluctuations-{}.npy'.format(operator)))

        with plotter.saved_figure(
                label['sigma ttW'],
                label['sigma ttZ'],
                os.path.join('transformed' if transform else '', 'ttZ_ttW_2D_1D_{}'.format(operator)),
                header=args.header) as ax:
            handles, labels = ttZ_ttW_2D(config, ax)

            x = data['x_sec_ttW']
            y = data['x_sec_ttZ']

            kdehist = kde.kdehist2(x[:10000], y[:10000], [70, 70])
            clevels = sorted(kde.confmap(kdehist[0], [.6827,.9545]))
            contour = ax.contour(kdehist[1], kdehist[2], kdehist[0], clevels, colors=['#ff321a', 'blue'],
                    linestyles=['dotted', 'dashed'])
            for index, handle in enumerate(contour.collections[::-1]):
                handles.append(handle)
                labels.append('{}$\sigma$ CL'.format(index + 1))

            table.append([operator, '{:.2f}'.format(data[0][operator])] + ['{:.2f}'.format(data[0][x]) for x in ['r_ttZ', 'r_ttW', 'r_ttH']])
            colors = ['black', 'gray']
            for (bf, _), color in zip(nll[operator]['best fit'], colors):
                # data[0] contains the best fit parameters
                # for low, high in nll[operator]['one sigma']:
                #     if (bf > low) and (bf < high):
                #         break
                point, = plt.plot(
                    data[0]['x_sec_ttW'],
                    data[0]['x_sec_ttZ'],
                    color=color,
                    markeredgecolor=color,
                    mew=3,
                    markersize=17,
                    marker="*",
                    linestyle='None'
                )
                handles.append(point)

                # if transform:
                #     template = r'$|{}{}{}|={:03.1f}\,{}$' if dimensionless else r'$|{}/\Lambda^2{}\,{}|={:03.1f}\,{}$'
                # else:
                #     template = r'${}{}{}={:03.1f}\,{}$' if dimensionless else r'${}/\Lambda^2{}{}={:03.1f}\,{}$'

                # labels.append("best fit:\n" + template.format(label[operator].replace('$', ''),
                #     nll[operator]['offset label'],
                #     (units if (transform or (nll[operator]['offset label'] != '')) else ''),
                #     round(bf, 2) + 0,
                #     units)
                # )
                if transform:
                    template = r'$|{}{}{}|$' if dimensionless else r'$|{}/\Lambda^2{}\,{}|$'
                else:
                    template = r'${}{}{}$' if dimensionless else r'${}/\Lambda^2{}{}$'

                labels.append("best fit\n" + template.format(label[operator].replace('$', ''),
                    nll[operator]['offset label'],
                    (units if (transform or (nll[operator]['offset label'] != '')) else ''))
                )
                print 'dimensionless, labels ', dimensionless, labels
            plt.legend(handles, labels, loc='lower right', fontsize=22)
            # left = plt.legend(handles[:3], labels[:3], loc='upper left', fontsize=21.)
            # plt.legend(handles[3:], labels[3:], loc='upper right', fontsize=21.)
            # plt.gca().add_artist(left)
            plt.ylim(ymin=0, ymax=y_max)
            plt.xlim(xmin=0, xmax=1400)
            # plt.xlim(xmin=0, xmax=2400)

        # with plotter.saved_figure(label['sigma ttW'], '', 'x_sec_ttW_{}'.format(operator)) as ax:
        #     avg = np.average(data['x_sec_ttW'])
        #     var = np.std(data['x_sec_ttW'], ddof=1)
        #     info = u"$\mu$ = {:.3f}, $\sigma$ = {:.3f}".format(avg, var)

        #     n, bins, _ = ax.hist(data['x_sec_ttW'], bins=100)
        #     ax.text(0.75, 0.8, info, ha="center", transform=ax.transAxes, fontsize=20)
        #     plt.axvline(x=avg - var, linestyle='--', color='black', label='$\mu - \sigma$')
        #     plt.axvline(x=avg + var, linestyle='--', color='red', label='$\mu + \sigma$')
        #     plt.legend()

        # for par in par_names + [operator]:
        #     with plotter.saved_figure(par, '', 'pulls/{}_{}'.format(operator, par)) as ax:
        #         plt.title(operator)
        #         ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        #         avg = np.average(data[par])
        #         var = np.std(data[par])
        #         info = u"$\mu$ = {:.5f}, $\sigma$ = {:.5f}".format(avg, var)

        #         n, _, _ = ax.hist(data[par], bins=100)
        #         ax.text(0.75, 0.8, info, ha="center", transform=ax.transAxes, fontsize=20)

    print tabulate.tabulate(table, headers=['coefficient', 'coefficient value', 'ttZ', 'ttW', 'ttH'])

def plot(args, config):

    plotter = Plotter(config)

    if args.plot != 'all':
        config['operators'] = [args.plot]

    # nll(args, config, plotter, dimensionless=False)
    nll(args, config, plotter, transform=True, dimensionless=False)
    nll(args, config, plotter, transform=False, dimensionless=False)
    mu(config, plotter)
    # mu(config, plotter, overlay_results=False, dimensionless=True)
    # mu_new(config, plotter, overlay_results=False, dimensionless=True)

    # ttZ_ttW_2D_1D_eff_op(args, config, plotter, transform=True, dimensionless=False)
    # ttZ_ttW_2D_1D_eff_op(args, config, plotter, transform=False, dimensionless=False)

    # ttZ_ttW_2D_1D_ttZ_1D_ttW(args, config, plotter)
