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
from matplotlib.backends.backend_pgf import FigureCanvasPgf
matplotlib.backend_bases.register_backend('pdf', FigureCanvasPgf)

pgf_with_latex = {
    "pgf.texsystem": "pdflatex",     # Use xetex for processing
    # "text.usetex": True,            # use LaTeX to write all text
    # "font.family": "serif",         # use serif rather than sans-serif
    # "font.serif": "Ubuntu",         # use Ubuntu as the font
    # "font.sans-serif": [],          # unset sans-serif
    # "font.monospace": "Ubuntu Mono",# use Ubuntu for monospace
    # "axes.labelsize": 10,
    # "font.size": 10,
    # "legend.fontsize": 8,
    # "axes.titlesize": 14,           # Title size when one figure
    # "xtick.labelsize": 8,
    # "ytick.labelsize": 8,
    # "figure.titlesize": 12,         # Overall figure title
    "pgf.rcfonts": False,           # Ignore Matplotlibrc
    # "text.latex.unicode": True,     # Unicode in LaTeX
    # "pgf.preamble": [               # Set-up LaTeX
    #     r'\usepackage{fontspec}',
    #     r'\setmainfont{Ubuntu}',
    #     r'\setmonofont{Ubuntu Mono}',
    #     r'\usepackage{unicode-math}',
    #     r'\setmathfont{Ubuntu}'
    # ]
}

matplotlib.rcParams.update(pgf_with_latex)

import numpy as np
import ROOT
import scipy.optimize as so
from scipy.stats import kde
from scipy.stats import gaussian_kde

from EffectiveTTV.EffectiveTTV.parameters import nlo, kappa
from EffectiveTTV.EffectiveTTV import kde
from EffectiveTTV.EffectiveTTV.signal_strength import load, load_mus
from EffectiveTTV.EffectiveTTV.nll import fit_nll
from EffectiveTTV.EffectiveTTV.fluctuate import par_names

label = {
    'sigma ttW': r'$\sigma_{\mathrm{t\bar{t}W}}$ $\mathrm{[fb]}$',
    'sigma ttZ': r'$\sigma_{\mathrm{t\bar{t}Z}}$ $\mathrm{[fb]}$',
    'ttW': r'$\mathrm{t\bar{t}W}$',
    'ttZ': r'$\mathrm{t\bar{t}Z}$',
    'ttH': r'$\mathrm{t\bar{t}H}$',
    'tt': r'$\mathrm{t\bar{t}}$',
    'c2B': r'$\bar{c}_{2B}$',
    'c2G': r'$\bar{c}_{2G}$',
    'c2W': r'$\bar{c}_{2W}$',
    'c3G': r'$\bar{c}_{3G}$',
    'cuB': r'$\bar{c}_{uB}$',
    'c3W': r'$\bar{c}_{3W}$',
    'c6': r'$\bar{c}_{6}$',
    'cA': r'$\bar{c}_{A}$',
    'cB': r'$\bar{c}_{B}$',
    'cG': r'$\bar{c}_{G}$',
    'cH': r'$\bar{c}_{H}$',
    'cHB': r'$\bar{c}_{HB}$',
    'cHL': r'$\bar{c}_{HL}$',
    'cHQ': r'$\bar{c}_{HQ}$',
    'cHW': r'$\bar{c}_{HW}$',
    'cHd': r'$\bar{c}_{Hd}$',
    'cHe': r'$\bar{c}_{He}$',
    'cHu': r'$\bar{c}_{Hu}$',
    'cHud': r'$\bar{c}_{Hud}$',
    'cT': r'$\bar{c}_{T}$',
    'cWW': r'$\bar{c}_{WW}$',
    'cd': r'$\bar{c}_{d}$',
    'cdB': r'$\bar{c}_{dB}$',
    'cdG': r'$\bar{c}_{dG}$',
    'cdW': r'$\bar{c}_{dW}$',
    'cl': r'$\bar{c}_{l}$',
    'clB': r'$\bar{c}_{lB}$',
    'clW': r'$\bar{c}_{lW}$',
    'cpHL': r"$\bar{c}'_{HL}$",
    'cpHQ': r"$\bar{c}'_{HQ}$",
    'cu': r'$\bar{c}_{u}$',
    'cuB': r'$\bar{c}_{uB}$',
    'cuG': r'$\bar{c}_{uG}$',
    'cuW': r'$\bar{c}_{uW}$',
    'tc3G': r'$\widetilde{c}_{3G}$',
    'tc3W': r'$\widetilde{c}_{3W}$',
    'tcA': r'$\widetilde{c}_{A}$',
    'tcG': r'$\widetilde{c}_{G}$',
    'tcHB': r'$\widetilde{c}_{HB}$',
    'tcHW': r'$\widetilde{c}_{HW}$',
}

extent = (0, 1500, 0, 2000)
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
            plt.title(lumi, loc='right', fontweight='bold')
            if header in ['preliminary only', 'preliminary']:
                plt.title(r'CMS preliminary', loc='left', fontweight='bold')
            elif header == 'final':
                plt.title(r'CMS', loc='left', fontweight='bold')

        try:
            yield ax

        finally:
            logging.info('saving {}'.format(name))
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.savefig(os.path.join(self.config['outdir'], 'plots', '{}.pdf'.format(name)), bbox_inches='tight', transparent=True)
            plt.savefig(os.path.join(self.config['outdir'], 'plots', '{}.png'.format(name)), bbox_inches='tight')
            plt.close()


def ratio_fits(config, plotter):
    coefficients, cross_sections = load(config)
    mus = load_mus(config)

    for operator in coefficients.values()[0]:
        if operator == 'sm':
            continue

        with plotter.saved_figure(
                label[operator],
                '$\sigma_{NP+SM} / \sigma_{SM}$',
                os.path.join('cross_sections', 'ratio_fits', operator)) as ax:
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            for process in ['tt', 'H', 'ttZ', 'ttH', 'ttW']:
            # for process in ['ttZ', 'ttH', 'ttW']:
                x = coefficients[process][operator]
                y = cross_sections[process][operator] / cross_sections[process]['sm']
                fit = mus[operator][process]
                if process in ['tt', 'H', 'ttZ', 'ttH', 'ttW']:
                    ax.plot(np.linspace(-1, 1, 100), fit(np.linspace(-1, 1, 100)), color='#C6C6C6')
                    ax.plot(x, y, 'o', label=process)

            ax.legend(loc='upper center')

        for process in mus[operator]:
            with plotter.saved_figure(
                    label[operator],
                    '$\sigma_{NP+SM} / \sigma_{SM}$',
                    os.path.join('cross_sections', 'ratio_fits', '{}_{}'.format(operator, process))) as ax:
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                x = coefficients[process][operator]
                y = cross_sections[process][operator] / cross_sections[process]['sm']
                fit = mus[operator][process]
                ax.plot(np.linspace(-1, 1, 100), fit(np.linspace(-1, 1, 100)), color='#C6C6C6')
                ax.plot(x, y, 'o', label=process)

                ax.legend(loc='upper center')

def mu(config, plotter, overlay_results=False):
    nll = fit_nll(config)
    coefficients, cross_sections = load(config)
    mus = load_mus(config)

    # for operator in coefficients.values()[0]:
    for operator in config['operators']:
        if operator == 'sm':
            continue

        data = []
        with plotter.saved_figure(
                label[operator],
                '$\sigma_{NP+SM} / \sigma_{SM}$',
                os.path.join('mu', operator + ('_overlay' if overlay_results else ''))) as ax:

            xmin = min(np.array(nll[operator]['two sigma'])[:, 0])
            xmax = max(np.array(nll[operator]['two sigma'])[:, 1])

            xmin = xmin - (np.abs(xmin) * 0.1)
            xmax = xmax + (np.abs(xmax) * 0.1)
            for process in ['ttW', 'ttZ', 'ttH']:
                x = coefficients[process][operator]
                y = cross_sections[process][operator] / cross_sections[process]['sm']
                above = lambda low: x >= low
                below = lambda high: x <= high
                while len(x[above(xmin) & below(xmax)]) < 3:
                    xmin -= np.abs(xmin) * 0.1
                    xmax += np.abs(xmax) * 0.1
                xi = np.linspace(xmin, xmax, 10000)
                ax.plot(xi, mus[operator][process](xi), color='#C6C6C6')
                ax.plot(x[above(xmin) & below(xmax)], y[above(xmin) & below(xmax)], 'o', label=process)

            if operator in config['operators'] and overlay_results:
                colors = ['black', 'gray']
                for (x, _), color in zip(nll[operator]['best fit'], colors):
                    plt.axvline(
                        x=x,
                        ymax=0.5,
                        linestyle='-',
                        color=color,
                        label='best fit {}$={:.2f}$'.format(label[operator], x)
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

            plt.xlim(xmin=xmin, xmax=xmax)
            ax.legend(loc='upper center')


def nll(config, plotter):
    data = fit_nll(config)

    # for operator, info in data.items():
    #     with plotter.saved_figure(
    #             label[operator],
    #             '-2 $\Delta$ ln L',
    #             os.path.join('nll', operator)) as ax:
    #         ax.plot(info['x'], info['y'], 'o')

    #         for x, y in info['best fit']:
    #             ax.plot(x, y, 'o', c='#fc4f30', label='best fit: {:03.2f}'.format(x))

    #         for low, high in info['one sigma']:
    #             ax.plot([low, high], [1.0, 1.0], '-', label='$1\sigma$ CL [{:03.2f}, {:03.2f}]'.format(low, high))

    #         for low, high in info['two sigma']:
    #             ax.plot([low, high], [3.84, 3.84], '-', label='$2\sigma$ CL [{:03.2f}, {:03.2f}]'.format(low, high))

    #         ax.legend(loc='upper center')
    #         plt.ylim(ymin=0)

    for operator, info in data.items():
        low, high = np.array(info['best fit'])[:,0]
        offset = (low + high) / 2.
        def transform(x):
            return np.abs(x - offset)

        labelinfo = ''
        if round(offset, 2) != 0:
            labelinfo = ' {} {:03.2f}'.format(('+' if offset < 0 else '-'), np.abs(offset))
        with plotter.saved_figure(
                '$|{}{}|$'.format(label[operator].replace('$',''), labelinfo),
                '-2 $\Delta$ ln L',
                os.path.join('nll', operator + '_abs')) as ax:
            x = transform(info['x'])
            y = info['y']
            ax.plot(x[x.argsort()], y[x.argsort()], 'o')
            print '|{} - {:+03.2f}|'.format(label[operator], offset)

            bf = transform(low)
            plt.axvline(
                x=bf,
                ymax=0.5,
                linestyle='-',
                color='black',
                label='best fit: {:03.2f}'.format(bf)
            )

            print 'one sigma ', operator, info['one sigma']
            print 'one sigma 0 ', operator, info['one sigma'][0]
            print 'two sigma ', operator, info['two sigma']
            low, high = info['one sigma'][0]
            low = transform(low)
            high = transform(high)
            print 'low, high ', operator, low, high
            if (low > bf) and (high > bf):
                low = 0
            ax.plot([low, high], [1.0, 1.0], '--', label='$1\sigma$ CL [{:03.2f}, {:03.2f}]'.format(*sorted([low, high])),
                    color='black')

            low, high = info['two sigma'][0]
            low = transform(low)
            high = transform(high)
            if (low > bf) and (high > bf):
                low = 0
            ax.plot([low, high], [3.84, 3.84], ':', label='$2\sigma$ CL [{:03.2f}, {:03.2f}]'.format(*sorted([low, high])),
                    color='black')

            ax.legend(loc='upper center')
            plt.ylim(ymin=0)

def ttZ_ttW_2D(config, ax):
    from root_numpy import root2array
    # FIXME: use config outdir here
    limits = root2array(os.path.join('scans', 'ttZ_ttW_2D.total.root'))

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
    print 'best fit ttW, ttZ', x[z.argmin()], y[z.argmin()]

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

        print 'ttW 1D: ', data['limit'][0] * nlo['ttW']
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

        print 'ttZ 1D: ', data['limit'][0] * nlo['ttZ']
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

        plt.legend(handles, labels, fontsize=15)

def find_confidence_interval(x, pdf, confidence_level):
    return pdf[pdf > x].sum() - confidence_level

def ttZ_ttW_2D_1D_eff_op(args, config, plotter):
    from root_numpy import root2array
    mus = load_mus(config)
    nll = fit_nll(config)

    table = []
    for operator in config['operators']:
        data = np.load(os.path.join(config['outdir'], 'fluctuations-{}.npy'.format(operator)))

        with plotter.saved_figure(
                label['sigma ttW'],
                label['sigma ttZ'],
                'ttZ_ttW_2D_1D_{}'.format(operator),
                header=args.header) as ax:
            handles, labels = ttZ_ttW_2D(config, ax)

            # info = root2array('best-fit-{}.root'.format(operator))
            mu = mus[operator]
            # bf, low, high = info[operator]

            x = data['x_sec_ttW']
            y = data['x_sec_ttZ']
            # plt.plot(x, y, linestyle='none', marker='o', color='red')

            kdehist = kde.kdehist2(x[:10000], y[:10000], [70, 70])
            clevels = sorted(kde.confmap(kdehist[0], [.6827,.9545]))
            contour = ax.contour(kdehist[1], kdehist[2], kdehist[0], clevels, colors=['#30a2da', '#fc4f30'])
            for index, handle in enumerate(contour.collections[::-1]):
                handles.append(handle)
                labels.append('{} {}$\sigma$ CL'.format(label[operator], index + 1))

            print 'best fit x sec ttZ', data[0]['x_sec_ttZ']
            print 'best fit x sec ttW', data[0]['x_sec_ttW']
            print 'best fit x sec ttH', data[0]['x_sec_ttH']
            table.append([operator, '{:.1f}'.format(data[0][operator])] + ['{:.1f}'.format(data[0][x]) for x in ['r_ttZ', 'r_ttW', 'r_ttH']])
            colors = ['black', 'gray']
            for (bf, _), color in zip(nll[operator]['best fit'], colors):
                for low, high in nll[operator]['one sigma']:
                    if (bf > low) and (bf < high):
                        break
                # data[0] contains the best fit parameters
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
                labels.append(
                    'best fit {}\n{:03.2f} [{:03.2f}, {:03.2f}]'.format(
                        label[operator],
                        round(bf, 2) + 0,
                        low,
                        high
                    )
                )
            plt.legend(handles, labels, fontsize=15)
            plt.ylim(ymin=0, ymax=y_max)

        with plotter.saved_figure(label['sigma ttW'], '', 'x_sec_ttW_{}'.format(operator)) as ax:
            avg = np.average(data['x_sec_ttW'])
            var = np.std(data['x_sec_ttW'], ddof=1)
            info = u"$\mu$ = {:.3f}, $\sigma$ = {:.3f}".format(avg, var)

            n, bins, _ = ax.hist(data['x_sec_ttW'], bins=100)
            ax.text(0.75, 0.8, info, ha="center", transform=ax.transAxes, fontsize=20)
            plt.axvline(x=avg - var, linestyle='--', color='black', label='$\mu - \sigma$')
            plt.axvline(x=avg + var, linestyle='--', color='red', label='$\mu + \sigma$')
            plt.legend()

        for par in par_names + [operator]:
            with plotter.saved_figure(par, '', 'pulls/{}_{}'.format(operator, par)) as ax:
                plt.title(operator)
                ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
                avg = np.average(data[par])
                var = np.std(data[par])
                info = u"$\mu$ = {:.5f}, $\sigma$ = {:.5f}".format(avg, var)

                n, _, _ = ax.hist(data[par], bins=100)
                ax.text(0.75, 0.8, info, ha="center", transform=ax.transAxes, fontsize=20)

    print tabulate.tabulate(table, headers=['coefficient', 'coefficient value', 'ttZ', 'ttW', 'ttH'])

def wilson_coefficients_in_window(args, config, plotter):

    mus = load_mus(config)

    for operator in config['operators']:
        with plotter.saved_figure(
                label['sigma ttW'],
                label['sigma ttZ'],
                'ttZ_ttW_2D_cross_section_with_sampled_{}'.format(operator),
                header=args.header) as ax:
            ttZ_ttW_2D(config, ax)

            ax.set_color_cycle([plt.cm.cool(i) for i in np.linspace(0, 1, 28)])

            high = min(max((mus[operator]['ttW'] - 1000).roots().real), max((mus[operator]['ttZ'] - 1000).roots().real))
            low = max(min((mus[operator]['ttW'] - 1000).roots().real), min((mus[operator]['ttZ'] - 1000).roots().real))
            coefficients = np.linspace(low, high, 28)
            # avoid overlapping due to symmetric points
            for coefficient in np.hstack([coefficients[:14:2], coefficients[14::2]]):
                plt.plot(
                    mus[operator]['ttW'](coefficient) * nlo['ttW'],
                    mus[operator]['ttZ'](coefficient) * nlo['ttZ'],
                    marker='x',
                    markersize=17,
                    mew=3,
                    label='{}={:03.2f}'.format(operator, coefficient),
                    linestyle="None"
                )

            ax.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0., frameon=False)


def plot(args, config):

    from root_numpy import root2array
    plotter = Plotter(config)

    if args.plot != 'all':
        config['operators'] = [args.plot]

    if args.publication:
        plt.rcParams['axes.facecolor'] = '#ffffff'
        plt.rcParams['savefig.facecolor'] = '#ffffff'

    nll(config, plotter)
    mu(config, plotter)
    mu(config, plotter, overlay_results=True)

    ttZ_ttW_2D_1D_eff_op(args, config, plotter)

    # ratio_fits(config, plotter)
    # ttZ_ttW_2D_1D_ttZ_1D_ttW(args, config, plotter)
    # wilson_coefficients_in_window(args, config, plotter)
