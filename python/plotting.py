import atexit
import contextlib
from datetime import datetime
import glob
import logging
import os
import tarfile
import tabulate

import jinja2
import matplotlib
from matplotlib.mlab import griddata
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

import numpy as np
from root_numpy import root2array

from EffectiveTTV.EffectiveTTV.parameters import nlo, label
from EffectiveTTV.EffectiveTTV import kde
from EffectiveTTV.EffectiveTTV.nll import fit_nll
from EffectiveTTV.EffectiveTTV.makeflow import multidim_np, multi_signal, max_likelihood_fit, multidim_grid, fluctuate

from EffectiveTTVProduction.EffectiveTTVProduction.cross_sections import CrossSectionScan

x_min, x_max, y_min, y_max = np.array([0.200, 1.200, 0.550, 2.250])


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


class Plot(object):
    def __init__(self, subdir):
        self.subdir = subdir

    def make(self):
        """Make all inputs

        This method should add all of the commands producing input files which
        are needed by the Plot to the MakeflowSpecification. This command will
        be run for each plot in the analysis config file. By specifying this
        per-plot, commenting out plots in the config will remove their inputs
        from the Makefile, so that only the needed inputs are produced.

        """
        pass

    def write(self, config):
        """Write the plots

        This command should actually produce and save the plots.
        """
        try:
            os.makedirs(os.path.join(config['outdir'], 'plots', self.subdir))
        except OSError:
            pass  # the directory has already been made


class NewPhysicsScaling(Plot):
    def __init__(self, processes=[('ttZ', '+', '#2fd164')], subdir='scaling', overlay_result=False, dimensionless=False):
        self.subdir = subdir
        self.processes = processes
        self.overlay_result = overlay_result
        self.dimensionless = dimensionless

    def make(self, config, spec, index):
        spec.add(['config.py', 'cross_sections.npz'], 'scales.npy', ['run', 'scale', 'config.py'])
        inputs = ['matplotlibrc', 'config.py', 'scales.npy']
        inputs += multidim_np(config, spec, np.ceil(config['np points'] / config['np chunksize']))

        for coefficient in config['coefficients']:
            spec.add(inputs, [], ['run', 'plot', '--coefficient', coefficient, '--index', index, 'config.py'])

    def write(self, config, plotter, args):
        super(NewPhysicsScaling, self).write(config)
        fn = os.path.join(config['outdir'], 'cross_sections.npz')
        scan = CrossSectionScan([fn])
        mus = np.load(os.path.join(config['outdir'], 'scales.npy'))[()]
        nll = fit_nll(config, transform=False, dimensionless=self.dimensionless)

        for coefficient in config['coefficients']:
            with plotter.saved_figure(
                    label[coefficient] + ('' if self.dimensionless else r' $/\Lambda^2$'),
                    '$\sigma_{NP+SM} / \sigma_{SM}$',
                    os.path.join(self.subdir, coefficient)) as ax:
                xmin = nll[coefficient]['x'][nll[coefficient]['y'] < 13].min()
                xmax = nll[coefficient]['x'][nll[coefficient]['y'] < 13].max()

                for process, marker, c in self.processes:
                    x = scan.points[tuple([coefficient])][process]
                    y = scan.signal_strengths[tuple([coefficient])][process]
                    xi = np.linspace(xmin, xmax, 10000)

                    ax.plot(xi * nll[coefficient]['conversion'], mus[coefficient][process](xi), color='#C6C6C6')
                    ax.plot(x * nll[coefficient]['conversion'], y, marker, mfc='none', markeredgewidth=2, markersize=15, label=label[process],
                            color=c)

                if self.overlay_result:
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

                plt.xlim(xmin=xmin, xmax=xmax)
                plt.ylim(ymin=0, ymax=3.2)
                plt.title(r'CMS Simulation', loc='left', fontweight='bold')
                plt.title(r'MG5_aMC@NLO LO', loc='right', size=27)
                ax.legend(loc='upper center')
                ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))


class NLL(Plot):
    def __init__(self, subdir='nll', transform=True, dimensionless=False):
        self.subdir = subdir
        self.transform = transform
        self.dimensionless = dimensionless

    def make(self, config, spec, index):
        inputs = ['matplotlibrc', 'config.py', 'scales.npy']
        inputs += multidim_np(config, spec, np.ceil(config['np points'] / config['np chunksize']))
        spec.add(['config.py', 'cross_sections.npz'], 'scales.npy', ['run', 'scale', 'config.py'])

        for coefficient in config['coefficients']:
            spec.add(inputs, [], ['run', 'plot', '--coefficient', coefficient, '--index', index, 'config.py'])

    def write(self, config, plotter, args):
        super(NLL, self).write(config)
        data = fit_nll(config, self.transform, self.dimensionless)
        mus = np.load(os.path.join(config['outdir'], 'scales.npy'))[()]

        for coefficient in config['coefficients']:
            info = data[coefficient]
            for p in config['processes']:
                s0, s1, s2 = mus[coefficient][p].coef
                if not ((s1 > 1e-5) or (s2 > 1e-5)):
                    continue  # coefficient has no effect on any of the scaled processes
            x_label = '{} {}'.format(info['label'].replace('\ \mathrm{TeV}^{-2}', ''), info['units'])

            with plotter.saved_figure(
                    x_label,
                    '$-2\ \Delta\ \mathrm{ln}\ \mathrm{L}$' + (' (asimov data)' if config['asimov data'] else ''),
                    os.path.join(self.subdir, coefficient),
                    header=config['header']) as ax:
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
                    ax.plot([low, high], [1.0, 1.0], '--', label=r'68% CL' if (i == 0) else '', color='blue')
                for i, (low, high) in enumerate(info['two sigma']):
                    ax.plot([low, high], [3.84, 3.84], '-.', label=r'95% CL' if (i == 0) else '', color='#ff321a')

                ax.legend(loc='upper center')
                ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                plt.ylim(ymin=0, ymax=12)
                plt.xlim(xmin=info['x'][info['y'] < 13].min(), xmax=info['x'][info['y'] < 13].max())
                if info['transformed']:
                    plt.xlim(xmin=0)


def two_signal_best_fit(config, ax, signals, theory_errors, tag, contours):
    limits = root2array(os.path.join(config['outdir'], 'best-fit-{}.root'.format(tag)))

    x = limits['r_{}'.format(signals[0])] * nlo[signals[0]]
    y = limits['r_{}'.format(signals[1])] * nlo[signals[1]]
    z = 2 * limits['deltaNLL']

    if contours:
        levels = {
            2.30: '  1 $\sigma$',
            5.99: '  2 $\sigma$',
            # 11.83: ' 3 $\sigma$',
            # 19.33: ' 4 $\sigma$',
            # 28.74: ' 5 $\sigma$'
        }

        xi = np.linspace(x_min, x_max, 1000)
        yi = np.linspace(y_min, y_max, 1000)
        zi = griddata(x, y, z, xi, yi, interp='linear')

        cs = plt.contour(xi, yi, zi, sorted(levels.keys()), colors='black', linewidths=2)
        plt.clabel(cs, fmt=levels)

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

    if theory_errors:
        x_process = signals[0]
        y_process = signals[1]
        xerr_low, xerr_high = np.array(theory_errors[y_process]) * nlo[y_process]
        yerr_low, yerr_high = np.array(theory_errors[x_process]) * nlo[x_process]
        theory = plt.errorbar(
            nlo[x_process], nlo[y_process],
            yerr=[[xerr_low], [xerr_high]],
            xerr=[[yerr_low], [yerr_high]],
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


class TwoProcessCrossSectionSM(Plot):
    def __init__(self, subdir='.', signals=['ttW', 'ttZ'], theory_errors=None, tag=None, numpoints=500, chunksize=500, contours=True):
        self.subdir = subdir
        self.signals = signals
        self.theory_errors = theory_errors
        if tag:
            self.tag = tag
        else:
            self.tag = '-'.join(signals)
        self.numpoints = numpoints
        self.chunksize = chunksize
        self.contours = contours

    def make(self, config, spec, index):
        inputs = ['matplotlibrc', 'config.py']
        inputs += multi_signal(self.signals, self.tag, spec, config)
        for signal in self.signals:
            inputs += max_likelihood_fit(signal, spec, config)
        if self.contours:
            inputs += multidim_grid(config, self.tag, self.numpoints, self.chunksize, spec)

        spec.add(inputs, [],  ['run', 'plot', '--index', index, 'config.py'])

    def write(self, config, plotter, args):
        x = self.signals[0]
        y = self.signals[1]
        with plotter.saved_figure(
                label['sigma {}'.format(x)],
                label['sigma {}'.format(y)],
                self.tag,
                header=config['header']) as ax:
            handles, labels = two_signal_best_fit(config, ax, self.signals, self.theory_errors, self.tag, self.contours)

            data = root2array(os.path.join(config['outdir'], 'best-fit-{}.root'.format(x)))

            x_cross_section = plt.axvline(x=data['limit'][0] * nlo[x], color='black')
            x_error = ax.axvspan(
                data['limit'][1] * nlo[x],
                data['limit'][2] * nlo[x],
                alpha=0.5,
                color='#FA6900',
                linewidth=0.0
            )
            handles.append((x_cross_section, x_error))
            labels.append('{} 1D $\pm$ $1\sigma$'.format(label[x]))

            data = root2array(os.path.join(config['outdir'], 'best-fit-{}.root'.format(y)))

            y_cross_section = plt.axhline(y=data['limit'][0] * nlo[y], color='black')
            y_error = ax.axhspan(
                data['limit'][1] * nlo[y],
                data['limit'][2] * nlo[y],
                color='#69D2E7',
                alpha=0.5,
                linewidth=0.0
            )
            handles.append((y_cross_section, y_error))
            labels.append('{} 1D $\pm$ $1\sigma$'.format(label[y]))

            plt.legend(handles, labels)


class TwoProcessCrossSectionSMAndNP(Plot):
    def __init__(self, subdir='.', signals=['ttW', 'ttZ'], theory_errors=None, tag=None, transform=True, dimensionless=False):
        self.subdir = subdir
        self.signals = signals
        self.theory_errors = theory_errors
        if tag:
            self.tag = tag
        else:
            self.tag = '-'.join(signals)
        self.transform = transform
        self.dimensionless = dimensionless

    def make(self, config, spec, index):
        inputs = ['matplotlibrc', 'config.py', 'scales.npy']
        inputs += multi_signal(self.signals, self.tag, spec, config)
        inputs += multidim_np(config, spec, np.ceil(config['np points'] / config['np chunksize']))
        inputs += fluctuate(config, spec)

        spec.add(inputs, [],  ['run', 'plot', '--index', index, 'config.py'])

    def write(self, config, plotter, args):
        x_proc = self.signals[0]
        y_proc = self.signals[1]
        nll = fit_nll(config, self.transform, self.dimensionless)

        table = []
        scales = ['r_{}'.format(x) for x in config['processes']]
        for coefficient in config['coefficients']:
            data = np.load(os.path.join(config['outdir'], 'fluctuations-{}.npy'.format(coefficient)))[()]
            if np.isnan(data['x_sec_{}'.format(x_proc)]).any() or np.isnan(data['x_sec_{}'.format(y_proc)]).any():
                print 'skipping coefficient {} with nan fluctuations'.format(coefficient)
                continue

            with plotter.saved_figure(
                    label['sigma {}'.format(x_proc)],
                    label['sigma {}'.format(y_proc)],
                    os.path.join(self.subdir, '{}_{}'.format(self.tag, coefficient)),
                    header=config['header']) as ax:
                handles, labels = two_signal_best_fit(config, ax, self.signals, self.theory_errors, self.tag, False)

                x = data['x_sec_{}'.format(x_proc)]
                y = data['x_sec_{}'.format(y_proc)]

                try:
                    kdehist = kde.kdehist2(x, y, [70, 70])
                    clevels = sorted(kde.confmap(kdehist[0], [.6827, .9545]))
                    contour = ax.contour(kdehist[1], kdehist[2], kdehist[0], clevels, colors=['#ff321a', 'blue'], linestyles=['-.', '--'])
                    for handle, l in zip(contour.collections[::-1], ['68% CL', '95% CL']):
                        handles.append(handle)
                        labels.append(l)
                except Exception as e:
                    print 'problem making contour for {}: {}'.format(coefficient, e)

                colors = ['black', 'gray']
                for (bf, _), color in zip(nll[coefficient]['best fit'], colors):
                    table.append([coefficient, '{:.2f}'.format(bf), '{:.2f}'.format(data[0][coefficient])] + ['{:.2f}'.format(data[0][i]) for i in scales])
                    point, = plt.plot(
                        data[0]['x_sec_{}'.format(x_proc)],
                        data[0]['x_sec_{}'.format(y_proc)],
                        color=color,
                        markeredgecolor=color,
                        mew=3,
                        markersize=17,
                        marker="x",
                        linestyle='None'
                    )
                    handles.append(point)

                    labels.append("Best fit\n{}".format(nll[coefficient]['label']))
                plt.legend(handles, labels, loc='upper right', fontsize=27)
                plt.ylim(ymin=y_min, ymax=y_max)
                plt.xlim(xmin=x_min, xmax=x_max)
                ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        print tabulate.tabulate(table, headers=['coefficient', 'bf', 'coefficient value'] + config['processes'])


def plot(args, config):

    plotter = Plotter(config)

    if args.coefficient:
        config['coefficients'] = args.coefficient

    if args.header:
        config['header'] = args.header

    if args.index is not None:
        config['plots'][args.index].write(config, plotter, args)
    else:
        for p in config['plots']:
            p.write(config, plotter, args)
