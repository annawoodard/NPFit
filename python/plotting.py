# FIXME fix back for multidimensional
import atexit
import contextlib
from datetime import datetime
import glob
import itertools
import json
import logging
import os
import pickle
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

from EffectiveTTV.EffectiveTTV.parameters import nlo, label, conversion
from EffectiveTTV.EffectiveTTV import kde
from EffectiveTTV.EffectiveTTV.signal_strength import load, load_mus
from EffectiveTTV.EffectiveTTV.nll import fit_nll

from EffectiveTTVProduction.EffectiveTTVProduction.cross_sections import CrossSectionScan

scale = 1 / 1000.
x_min, x_max, y_min, y_max = np.array([200, 1200, 550, 2250]) * scale
for process in nlo:
    nlo[process] *= scale


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
            plt.title(lumi, loc='right', fontweight='normal', fontsize=27)
            plt.title(r'CMS', loc='left', fontweight='bold')
            if header == 'preliminary':
                plt.text(0.155, 1.009, r'Preliminary', style='italic', transform=ax.transAxes)

        try:
            yield ax

        finally:
            logging.info('saving {}'.format(name))
            plt.xlabel(x_label, horizontalalignment='right', x=1.0)
            plt.ylabel(y_label, horizontalalignment='right', y=1.0)
            plt.savefig(os.path.join(self.config['outdir'], 'plots', '{}.pdf'.format(name)), bbox_inches='tight')
            plt.savefig(os.path.join(self.config['outdir'], 'plots', '{}.png'.format(name)), bbox_inches='tight')
            plt.close()

def mu_per_process(config, plotter):
    coefficients, cross_sections = load(config)
    mus = load_mus(config)
    nll, _ = fit_nll(config, transform=False, dimensionless=True)
    with open(os.path.join(config['outdir'], 'extreme_mus.pkl'), 'rb') as f:
        extreme_mus = pickle.load(f)

    # for coefficient in mus:
    for coefficient in ['cHQ']:
        if coefficient == 'sm':
            continue

        # for process in mus[coefficient]:
        for process in ['ZZ']:
            if len(nll[coefficient]['two sigma']) == 0:
                continue
            s0, s1, s2 = mus[coefficient][process].coef
            if not ((s1 > 1e-5) or (s2 > 1e-5)):
                continue
            with plotter.saved_figure(
                    label[coefficient],
                    '$\sigma_{NP+SM} / \sigma_{SM}$',
                    os.path.join('mu', 'processes', '{}_{}'.format(coefficient, process))) as ax:

                try:
                    xmin = min(np.array(nll[coefficient]['two sigma'])[:, 0])
                    xmax = max(np.array(nll[coefficient]['two sigma'])[:, 1])
                except KeyError:
                    continue
                # xmin = -160
                # xmax = 160
                xi = np.linspace(xmin, xmax, 10000)
                x = coefficients[process][coefficient]
                y = cross_sections[process][coefficient] / cross_sections[process]['sm']

                ax.plot(xi, mus[coefficient][process](xi), color='#C6C6C6')
                l = label[process] if process in label else process
                ax.plot(x, y, '+', mfc='none', markeredgewidth=2, markersize=15, label=l)
                exmu = extreme_mus[coefficient][(xmin, xmax)][process]

                plt.axvline(
                    x=xmin,
                    ymax=0.48,
                    linestyle=':',
                    color='black',
                    # label='$2 \sigma [{:03.2f}, {:03.2f}]$'.format(low, high)
                )
                plt.axvline(
                    x=xmax,
                    ymax=0.48,
                    linestyle=':',
                    color='black'
                )
                plt.axhline(
                        y=exmu,
                        linestyle=':',
                        color='red',
                        label='$\mu_\mathrm{ext}$'
                )
                # plt.xlim(xmin=xmin, xmax=xmax)
                # plt.ylim(ymax=2 * max(y))
                plt.title(r'CMS Simulation', loc='left', fontweight='bold')
                plt.title(r'mg5_aMC LO', loc='right', size='small')
                ax.legend(loc='upper center')

def mu_new(config, plotter, overlay_results=False, dimensionless=False):
    #FIXME figure out if I still need this, fix x axis range to match nll if it is there
    coefficients, cross_sections = load(config)
    mus = load_mus(config)
    nll = fit_nll(config, transform=False, dimensionless=True)

    for coefficient in config['coefficients']:
    # for coefficients, xmin, xmax in [('cuW', -5, 5), ('cuB', -5, 5), ('cu', -30, 30), ('cHu', -7, 7)]:
        if coefficients == 'sm':
            continue

        with plotter.saved_figure(
                label[coefficient] + ('' if dimensionless else r' $/\Lambda^2$'),
                '$\sigma_{NP+SM} / \sigma_{SM}$',
                os.path.join('mu', coefficient + ('_overlay' if overlay_results else ''))) as ax:

            c_max = (4 * np.pi) ** 2
            print 'processes ', mus[coefficient].keys()
            try:
                xmin = max([(mus[coefficient][p] - 5).roots().min() for p in ['ttW', 'ttZ', 'ttH'] if (mus[coefficient][p](c_max) > 5)] + [-1 * c_max])
                xmax = min([(mus[coefficient][p] - 5).roots().max() for p in ['ttW', 'ttZ', 'ttH'] if (mus[coefficient][p](c_max) > 5)] + [c_max])
            except KeyError:
                continue
            # xmin = -160
            # xmax = 160
            xi = np.linspace(xmin, xmax, 10000)
            for process, marker in [('ttW', 'x'), ('ttZ', '+'), ('ttH', 'o')]:
                print coefficient
                print process, (mus[coefficient][process] - 5).roots()
                x = coefficients[process][coefficient]
                y = cross_sections[process][coefficient] / cross_sections[process]['sm']

                print coefficient, xmin, xmax
                ax.plot(xi * nll[coefficient]['conversion'], mus[coefficient][process](xi), color='#C6C6C6')
                ax.plot(x * nll[coefficient]['conversion'], y, marker, mfc='none', markeredgewidth=2, markersize=15, label=label[process])

            if overlay_results:
                colors = ['black', 'gray']
                for (x, _), color in zip(nll[coefficient]['best fit'], colors):
                    plt.axvline(
                        x=x,
                        ymax=0.5,
                        linestyle='-',
                        color=color,
                        label='Best fit\n {}$={:.2f}$'.format(label[coefficient], x)
                    )
                for (low, high), color in zip(nll[coefficient]['one sigma'], colors):
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
                for (low, high), color in zip(nll[coefficient]['two sigma'], colors):
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

            plt.xlim(xmin=xmin * nll[coefficient]['conversion'], xmax=xmax * nll[coefficient]['conversion'])
            plt.ylim(ymin=0, ymax=5)
            plt.title(r'CMS Simulation', loc='left', fontweight='bold')
            plt.title(r'mg5_aMC LO', loc='right', size='small')
            ax.legend(loc='upper center')

def mu(config, plotter, overlay_results=False, dimensionless=False):

    fn = os.path.join(config['outdir'], 'cross_sections.npz')
    scan = CrossSectionScan([fn])
    mus = load_mus(config)
    nll = fit_nll(config, transform=False, dimensionless=dimensionless)

    headers = []
    combinations = [tuple(sorted(x)) for x in itertools.combinations(config['coefficients'], config['dimension'])]
    for coefficients in combinations:
        tag = '-'.join(coefficients)
        with plotter.saved_figure(
                label[tag] + ('' if dimensionless else r' $/\Lambda^2$'),
                '$\sigma_{NP+SM} / \sigma_{SM}$',
                os.path.join('mu', tag + ('_overlay' if overlay_results else '') + ('_dimensionless' if
                    dimensionless else ''))) as ax:
            xmin = nll[tag]['x'][nll[tag]['y'] < 13].min()
            xmax = nll[tag]['x'][nll[tag]['y'] < 13].max()

            for process, marker, c in [('ttW', 'x', 'blue'), ('ttZ', '+', '#2fd164'), ('ttH', 'o', '#ff321a')]:
                x = scan.points[coefficients][process]
                y = scan.signal_strengths[coefficients][process]
                xi = np.linspace(xmin, xmax, 10000)

                ax.plot(xi * nll[tag]['conversion'], mus[tag][process](xi), color='#C6C6C6')
                ax.plot(x * nll[tag]['conversion'], y, marker, mfc='none', markeredgewidth=2, markersize=15, label=label[process],
                        color=c)

            if overlay_results:
                colors = ['black', 'gray']
                for (x, _), color in zip(nll[tag]['best fit'], colors):
                    plt.axvline(
                        x=x,
                        ymax=0.5,
                        linestyle='-',
                        color=color,
                        label='Best fit\n {}$={:.2f}$'.format(label[tag], x)
                    )
                for (low, high), color in zip(nll[tag]['one sigma'], colors):
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
                for (low, high), color in zip(nll[tag]['two sigma'], colors):
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

            plt.xlim(xmin=xmin, xmax=xmax)
            if tag == 'cuB':
                plt.xlim(xmin=-3.5, xmax=3.5)
            plt.ylim(ymin=0, ymax=3.2)
            plt.title(r'CMS Simulation', loc='left', fontweight='bold')
            plt.title(r'MG5_aMC@NLO LO', loc='right', size=27)
            ax.legend(loc='upper center')
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))


def nll(args, config, plotter, transform=False, dimensionless=True):
    data = fit_nll(config, transform, dimensionless)
    mus = np.load(os.path.join(config['outdir'], 'mus.npy'))[()]

    for coefficient, info in data.items():
        if coefficient not in config['coefficients']:
            continue
        for p in config['processes']:
            s0, s1, s2 = mus[coefficient][p].coef
            if not ((s1 > 1e-5) or (s2 > 1e-5)):
                continue # coefficient has no effect on any of the scaled processes
        x_label = '{} {}'.format(info['label'].replace('\ \mathrm{TeV}^{-2}', ''), info['units'])

        with plotter.saved_figure(
                x_label,
                '$-2\ \Delta\ \mathrm{ln}\ \mathrm{L}$' + (' (asimov data)' if config['asimov data'] else ''),
                os.path.join('nll', ('transformed' if transform else ''), coefficient + ('_dimensionless' if
                    dimensionless else '')),
                header=args.header) as ax:
            ax.plot(info['x'], info['y'], color='black')

            for i, (x, y) in enumerate(info['best fit']):
                if i == 0:
                    plt.axvline(
                        x=x,
                        ymax=0.5,
                        linestyle=':',
                        color='black',
                        label='Best fit',
                    )
                else:
                    plt.axvline(
                        x=x,
                        ymax=0.5,
                        linestyle=':',
                        color='black',
                    )
            for i, (low, high) in enumerate(info['one sigma']):
                ax.plot([low, high], [1.0, 1.0], '--', label=r'68% CL' if (i==0) else '', color='blue')
            for i, (low, high) in enumerate(info['two sigma']):
                ax.plot([low, high], [3.84, 3.84], '-.', label=r'95% CL' if (i==0) else '', color='#ff321a')

            ax.legend(loc='upper center')
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            plt.ylim(ymin=0, ymax=12)
            plt.xlim(xmin=info['x'][info['y'] < 13].min(), xmax=info['x'][info['y'] < 13].max())
            if info['transformed']:
                plt.xlim(xmin=0)

def ttZ_ttW_2D(config, ax):
    from root_numpy import root2array
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
        marker="*",
        linestyle='None'
    )
    handles.append(bf)
    labels.append('2D best fit')

    theory = plt.errorbar(
            nlo['ttW'], nlo['ttZ'],
            yerr=[[nlo['ttZ'] * 0.1164], [nlo['ttZ'] * 0.10]],
            xerr=[[nlo['ttW'] * 0.1173], [nlo['ttW'] * 0.1316]],
            capsize=5,
            mew=2,
            color='black',
            ls='',
            marker='o',
            markersize=10,
            linewidth=3
            )
    handles.append(theory)
    labels.append('{} theory\n[1610.07922]'.format(label['ttV']))

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
    nll = fit_nll(config, transform, dimensionless)

    table = []
    for coefficient in config['coefficients']:
        data = np.load(os.path.join(config['outdir'], 'fluctuations-{}.npy'.format(coefficient)))
        if np.isnan(data['x_sec_ttZ']).any() or np.isnan(data['x_sec_ttW']).any():
            print 'skipping coefficient {} with nan fluctuations'.format(coefficient)
            continue

        with plotter.saved_figure(
                label['sigma ttW'],
                label['sigma ttZ'],
                os.path.join('transformed' if transform else '', 'ttZ_ttW_2D_1D_{}'.format(coefficient)),
                header=args.header) as ax:
            handles, labels = ttZ_ttW_2D(config, ax)

            x = data['x_sec_ttW'] * scale
            y = data['x_sec_ttZ'] * scale

            kdehist = kde.kdehist2(x[:10000], y[:10000], [70, 70])
            clevels = sorted(kde.confmap(kdehist[0], [.6827,.9545]))
            contour = ax.contour(kdehist[1], kdehist[2], kdehist[0], clevels, colors=['#ff321a', 'blue'], linestyles=['-.', '--'])
            for handle, l in zip(contour.collections[::-1], ['68% CL', '95% CL']):
                handles.append(handle)
                labels.append(l)

            colors = ['black', 'gray']
            for (bf, _), color in zip(nll[coefficient]['best fit'], colors):
                table.append([coefficient, '{:.2f}'.format(bf), '{:.2f}'.format(data[0][coefficient])] + ['{:.2f}'.format(data[0][x]) for x in ['r_ttZ', 'r_ttW', 'r_ttH']])
                point, = plt.plot(
                    data[0]['x_sec_ttW'] * scale,
                    data[0]['x_sec_ttZ'] * scale,
                    color=color,
                    markeredgecolor=color,
                    mew=3,
                    markersize=17,
                    marker="x",
                    linestyle='None'
                )
                handles.append(point)

                labels.append("Best fit\n{}".format(nll[coefficient]['label']))
                print 'dimensionless, labels ', dimensionless, labels
            plt.legend(handles, labels, loc='upper right', fontsize=27)
            # left = plt.legend(handles[:3], labels[:3], loc='upper left', fontsize=21.)
            # plt.legend(handles[3:], labels[3:], loc='upper right', fontsize=21.)
            # plt.gca().add_artist(left)
            plt.ylim(ymin=y_min, ymax=y_max)
            plt.xlim(xmin=x_min, xmax=x_max)
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        # with plotter.saved_figure(label['sigma ttW'], '', 'x_sec_ttW_{}'.format(coefficient)) as ax:
        #     avg = np.average(data['x_sec_ttW'])
        #     var = np.std(data['x_sec_ttW'], ddof=1)
        #     info = u"$\mu$ = {:.3f}, $\sigma$ = {:.3f}".format(avg, var)

        #     n, bins, _ = ax.hist(data['x_sec_ttW'], bins=100)
        #     ax.text(0.75, 0.8, info, ha="center", transform=ax.transAxes, fontsize=20)
        #     plt.axvline(x=avg - var, linestyle='--', color='black', label='$\mu - \sigma$')
        #     plt.axvline(x=avg + var, linestyle='--', color='red', label='$\mu + \sigma$')
        #     plt.legend()

        # for par in config['systematics'] + [coefficient]:
        #     with plotter.saved_figure(par, '', 'pulls/{}_{}'.format(coefficient, par)) as ax:
        #         plt.title(coefficient)
        #         ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        #         avg = np.average(data[par])
        #         var = np.std(data[par])
        #         info = u"$\mu$ = {:.5f}, $\sigma$ = {:.5f}".format(avg, var)

        #         n, _, _ = ax.hist(data[par], bins=100)
        #         ax.text(0.75, 0.8, info, ha="center", transform=ax.transAxes, fontsize=20)

    print tabulate.tabulate(table, headers=['coefficient', 'bf', 'coefficient value', 'ttZ', 'ttW', 'ttH'])

def plot(args, config):

    plotter = Plotter(config)

    # FIXME allow for any available, any specified at the CL, and all in the config file
    if args.coefficient != 'all':
        config['coefficient'] = [args.coefficient]

    # nll(args, config, plotter, transform=False, dimensionless=False)
    # nll(args, config, plotter, transform=True, dimensionless=True)
    # nll(args, config, plotter, transform=False, dimensionless=True)

    # mu(config, plotter, overlay_results=False, dimensionless=True)
    # mu_new(config, plotter, overlay_results=False, dimensionless=True)
    # mu_per_process(config, plotter)

    # ttZ_ttW_2D_1D_eff_op(args, config, plotter, transform=True, dimensionless=True)
    # ttZ_ttW_2D_1D_eff_op(args, config, plotter, transform=False, dimensionless=True)
    # ttZ_ttW_2D_1D_eff_op(args, config, plotter, transform=False, dimensionless=False)

    nll(args, config, plotter, transform=True, dimensionless=False)
    mu(config, plotter)
    ttZ_ttW_2D_1D_eff_op(args, config, plotter, transform=True, dimensionless=False)

    # ttZ_ttW_2D_1D_ttZ_1D_ttW(args, config, plotter)
