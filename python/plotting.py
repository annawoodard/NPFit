import atexit
import contextlib
import glob
import logging
import os
import sys
import tarfile
from collections import defaultdict
from datetime import datetime

import jinja2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import scipy.ndimage
import seaborn as sns
import tabulate
from matplotlib.mlab import griddata
from matplotlib.ticker import FormatStrFormatter, LogLocator
from mpl_toolkits.axes_grid1 import ImageGrid
from root_numpy import root2array
from scipy.stats import chi2

from NPFit.NPFit import kde
from NPFit.NPFit.makeflow import (fluctuate, max_likelihood_fit, multi_signal,
                                  multidim_grid, multidim_np)
from NPFit.NPFit.nll import fit_nll
from NPFit.NPFit.parameters import conversion, label, nlo
from NPFit.NPFit.scaling import load_fitted_scan
from NPFitProduction.NPFitProduction.cross_sections import CrossSectionScan
from NPFitProduction.NPFitProduction.utils import (cartesian_product,
                                                   sorted_combos)

tweaks = {
    "lines.markeredgewidth": 0.0,
    "lines.linewidth": 7,
    "lines.markersize": 23,
    "patch.edgecolor": "black",
    "legend.facecolor": "white",
    "legend.frameon": True,
    "legend.edgecolor": "white",
    "legend.fontsize": "medium",
    "legend.handletextpad": 0.5,
    "mathtext.fontset": "custom",
    "mathtext.rm": "Bitstream Vera Sans",
    "mathtext.it": "Bitstream Vera Sans:italic",
    "mathtext.bf": "Bitstream Vera Sans:bold",
    "axes.labelsize": "large",
    "axes.titlesize": "medium",
    "xtick.labelsize": "medium",
    "xtick.major.size": 5,
    "ytick.major.size": 5,
    "xtick.direction": "in",
    "ytick.direction": "in",
}
sns.set(context="poster", style="white", font_scale=1.5, rc=tweaks)

x_min, x_max, y_min, y_max = np.array([0.200, 1.200, 0.550, 2.250])


def get_masked_colormap(bottom_map, top_map, norm, width, masked_value):
    low = masked_value - width / 2.
    high = masked_value + width / 2.
    if low > norm.vmin:
        colors = zip(np.linspace(0., norm(low), 100), bottom_map(np.linspace(0.1, 1., 100)))
        colors += [(norm(low), 'gray')]
    else:
        colors = [(0., 'gray')]
    if high < norm.vmax:
        colors += [(norm(high), 'gray')]
        colors += zip(np.linspace(norm(high), 1., 100), top_map(np.linspace(0.1, 1., 100)))
    else:
        colors += [(1., 'gray')]

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('masked_map', colors)

    return cmap

def get_stacked_colormaps(cmaps, interfaces, norm):
    colors = []
    low = 0.
    for cmap, interface in zip(cmaps, interfaces):
        colors += zip(np.linspace(low, norm(interface), 100), cmap(np.linspace(0, 1., 100)))
        low = norm(interface)

    colors += zip(np.linspace(low, 1., 100), cmaps[-1](np.linspace(0, 1., 100)))

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('masked_map', colors)

    return cmap

class Plotter(object):

    def __init__(self, config):
        atexit.register(self.webify)
        self.config = config

    def webify(self):
        path = os.path.join(os.environ['LOCALRT'], 'src', 'NPFit', 'NPFit', 'data')
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
    def saved_figure(self, x_label, y_label, name, header=False, figsize=(11, 11)):
        fig, ax = plt.subplots(figsize=figsize)
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
            plt.xlabel(x_label, horizontalalignment='right', x=1.0)
            plt.ylabel(y_label, horizontalalignment='right', y=1.0)
            plt.savefig(os.path.join(self.config['outdir'], 'plots', '{}.pdf'.format(name)), bbox_inches='tight')
            plt.savefig(os.path.join(self.config['outdir'], 'plots', '{}.png'.format(name)), bbox_inches='tight')
            plt.close()


class Plot(object):

    def __init__(self, subdir):
        self.subdir = subdir

    def specify(self):
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


def get_errs(scan, dimension, processes, config):
    res = None
    for coefficients in sorted_combos(config['coefficients'], dimension):
        for process in processes:
            if process in scan.points[coefficients]:
                predicted = scan.evaluate(coefficients, scan.points[coefficients][process], process)
                scales = scan.scales(coefficients, process)
                print('scales shape ', scales.shape, scan.points[coefficients][process].shape, coefficients, process)
                errs = (scales - predicted) / scales * 100

                bad = scan.points[coefficients][process][np.abs(errs) > 5]
                good = scan.points[coefficients][process][np.abs(errs) < 5]
                try:
                    print coefficients, process, 'len bad ', len(bad), 'len good ', len(good), len(bad) / float(len(good)) * 100
                    # for row in bad:
                    #     print row
                    # for row in good:
                    #     print row
                except ZeroDivisionError:
                    pass
                if res is None:
                    res = errs
                else:
                    res = np.concatenate([res, errs])

    return res


class FitFailures(Plot):

    def __init__(self, dimensions=[1], processes=['ttZ', 'ttH', 'ttW'], points=None, subdir='fit'):
        self.dimensions = dimensions
        self.processes = processes
        self.subdir = subdir
        if points is None:
            self.fitpoints = np.array(range(10, 1000, 100))
        else:
            self.fitpoints = np.array(points)

    def specify(self, config, spec, index):
        spec.add(['cross_sections.npz'], [], ['run', 'plot', '--index', index, config['fn']])

    def write(self, config, plotter, args):
        super(FitFailures, self).write(config)

        name = os.path.join(self.subdir, 'errors')
        x_label = r'$(\mathrm{r}_{\mathrm{MG}} - \mathrm{r}_{\mathrm{fit}}) / \mathrm{r}_{\mathrm{MG}} * 100$'
        maxtestpoints = None
        for dimension in self.dimensions:
            scan = load_fitted_scan(config, 'cross_sections.npz', maxpoints=max(self.fitpoints))
            errs = get_errs(scan, dimension, self.processes, config)
            if maxtestpoints is None:
                maxtestpoints = len(errs)
            else:
                maxtestpoints = min(len(errs), maxtestpoints)
        with plotter.saved_figure(x_label, 'counts', name) as ax:
            labels = []
            table = []
            scan = load_fitted_scan(config, 'cross_sections.npz')
            failure_ratio = np.zeros(len(self.fitpoints))

            for index, points in enumerate(self.fitpoints):
                scan = load_fitted_scan(config, 'cross_sections.npz', maxpoints=points)
                for dimension in self.dimensions:
                    errs = get_errs(scan, dimension, self.processes, config)
                    np.random.shuffle(errs)
                    errs = errs[:maxtestpoints]
                    bad = errs[np.abs(errs) > 5]
                    table.append([points, len(bad), len(errs), '{:.1f} %'.format(100. * len(bad) / len(errs))])
                    failure_ratio[index] = float(len(bad)) / len(errs)
                    # residuals

        name = os.path.join(self.subdir, 'failures')
        with plotter.saved_figure('fit points', 'percent failure (|percent error| > 5%)', name) as ax:
            ax.plot(self.fitpoints, failure_ratio * 100., marker='o', markersize=10, linewidth=1)

        headers = ['fit points', 'test points with |err| > 5%', 'total test points', 'percent failure']
        with open(os.path.join(config['outdir'], 'failures.txt'), 'w') as f:
            f.write(tabulate.tabulate(table, headers=headers) + '\n')

class FitErrors(Plot):

    def __init__(self, fit_dimensions=[1], eval_dimensions=[1], processes=['ttZ', 'ttH', 'ttW'], fit_to_test_ratio=1,
            subdir='fit', xmin=-100, xmax=50):
        self.fit_dimensions = fit_dimensions
        self.eval_dimensions = eval_dimensions
        self.processes = processes
        self.fit_to_test_ratio = float(fit_to_test_ratio)
        self.subdir = subdir
        self.xmin = xmin
        self.xmax = xmax

    def specify(self, config, spec, index):
        spec.add(['cross_sections.npz'], [], ['run', 'plot', '--index', index, config['fn']])

    def write(self, config, plotter, args):
        super(FitErrors, self).write(config)

        name = os.path.join(self.subdir, 'errors')
        x_label = r'$(\mathrm{r}_{\mathrm{MG}} - \mathrm{r}_{\mathrm{fit}}) / \mathrm{r}_{\mathrm{MG}} * 100$'
        maxpoints = None
        scan = CrossSectionScan(os.path.join(config['outdir'], 'cross_sections.npz'))
        for coefficients in scan.points:
            if len(coefficients) in sum(self.fit_dimensions, []):
                for process in scan.points[coefficients]:
                    numpoints = len(scan.points[coefficients][process])
                    if maxpoints is None:
                        maxpoints = int(np.ceil(self.fit_to_test_ratio * numpoints / (self.fit_to_test_ratio + 1)))
                    else:
                        maxpoints = int(min(maxpoints, np.ceil(self.fit_to_test_ratio * numpoints / (self.fit_to_test_ratio + 1))))

        with plotter.saved_figure(x_label, 'counts', name) as ax:
            labels = []
            table = []

            for fit_dims in self.fit_dimensions:
                scan.fit(dimensions=fit_dims)
                for eval_dim in self.eval_dimensions:
                    errs = get_errs(scan, eval_dim, self.processes, config)
                    errs[errs < self.xmin] = self.xmin
                    errs[errs > self.xmax] = self.xmax
                    bad = errs[np.abs(errs) > 5]
                    label = '{} fit, {}d evaluation\n{:,} test points'.format(
                        ' and '.join(('{}d'.format(dim) for dim in fit_dims)),
                        eval_dim,
                        len(errs),
                    )
                    ax.hist(errs, 70, histtype='step', fill=False, label=label)
                ax.set_yscale('log', subsy=range(10))
                ax.yaxis.set_tick_params('minor', size=5)
                plt.ylim(ymin=0, ymax=10e5)
                # box = ax.get_position()
                # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                # ax.legend(loc='center left', fontsize='small', bbox_to_anchor=(1, 0.5))
                plt.legend(fontsize='x-small', loc='upper left')


class NewPhysicsScaling2D(Plot):

    def __init__(
            self,
            processes=['ttZ', 'ttH', 'ttW'],
            subdir='scaling2d',
            dimensionless=False,
            dimension=2,
            maxnll=None,
            match_zwindows=False,
            madgraph=False,
            numvalues=100,
            numbins=80,
            points=40000):
        if dimension < 2:
            raise NotImplementedError('must have at least two dimensions')
        if maxnll is True and dimension is not 2:
            raise NotImplementedError('maxnll only works with dimension == 2')
        if madgraph is True and dimension is not 2:
            raise NotImplementedError('madgraph=True only works with dimension == 2')
        self.subdir = subdir
        self.processes = processes
        self.dimensionless = dimensionless
        self.dimension = dimension
        self.maxnll = maxnll
        self.match_zwindows = match_zwindows
        self.madgraph = madgraph
        self.numvalues = numvalues
        self.numbins = numbins
        self.points = points

    def specify(self, config, spec, index):
        inputs = []
        if self.maxnll is not None:
            inputs += multidim_np(config, spec, self.dimension, points=self.points)

        for coefficients in sorted_combos(config['coefficients'], 2):
            cmd = 'run plot --coefficients {coefficients} --index {index} {fn}'
            base = os.path.join(config['outdir'], 'plots', self.subdir, '_'.join(coefficients))
            outputs = [base + ext for ext in ['.pdf', '.png']]
            spec.add(inputs + ['cross_sections.npz'], outputs, cmd.format(coefficients=' '.join(coefficients), index=index, fn=config['fn']))

    def write(self, config, plotter, args):
        super(NewPhysicsScaling2D, self).write(config)
        scan = CrossSectionScan(os.path.join(config['outdir'], 'cross_sections.npz'))
        scan.fit(dimensions=[self.dimension])

        if self.match_zwindows:
            zmin = None
            zmax = None
            for coefficients in sorted_combos(config['coefficients'], 2):
                madgraph = scan.dataframe(coefficients)
                if zmin is None:
                    zmin = min(madgraph[self.processes].min())
                    zmax = max(madgraph[self.processes].max())
                else:
                    zmin = min(min(madgraph[self.processes].min()), zmin)
                    zmax = max(max(madgraph[self.processes].max()), zmax)

        for coefficients in sorted_combos(config['coefficients'], 2):
            tag = '_'.join(coefficients)
            name = os.path.join(self.subdir, tag)
            x = coefficients[0]
            y = coefficients[1]
            x_label = label[x] + ('' if self.dimensionless else r' $/\Lambda^2\ [\mathrm{TeV}^{-2}]$')
            y_label = label[y] + ('' if self.dimensionless else r' $/\Lambda^2\ [\mathrm{TeV}^{-2}]$')
            x_conv = 1. if self.dimensionless else conversion[x]
            y_conv = 1. if self.dimensionless else conversion[y]

            madgraph = scan.dataframe(coefficients)

            if self.maxnll:
                try:
                    data = root2array(os.path.join(config['outdir'], 'scans', '{}.total.root'.format(tag)))
                except IOError as e:
                    print 'input data missing, will not match nll for {}'.format(tag)
                    continue

                zi = 2 * data['deltaNLL']
                xi = data[x]
                yi = data[y]
                xmin = xi[zi < self.maxnll].min()
                ymin = yi[zi < self.maxnll].min()
                xmax = xi[zi < self.maxnll].max()
                ymax = yi[zi < self.maxnll].max()
                window = (madgraph[x] > xmin) & (madgraph[x] < xmax) & (madgraph[y] > ymin) & (madgraph[y] < ymax)
                madgraph = madgraph[window]

            if not self.match_zwindows:
                zmin = min(madgraph[self.processes].min())
                zmax = max(madgraph[self.processes].max())

            norm = matplotlib.colors.LogNorm(vmin=min(zmin, 0.9), vmax=zmax)
            masked_map = get_masked_colormap(
                    sns.light_palette("navy", as_cmap=True),
                    sns.light_palette((210, 90, 60), input="husl", as_cmap=True),
                    norm,
                    0.1,
                    1.0
            )

            fig = plt.figure(figsize=(30, 9))
            grid = ImageGrid(
                fig, 111,
                nrows_ncols=(1, len(self.processes)),
                axes_pad=0.45,
                share_all=True,
                cbar_location="right",
                cbar_mode="single",
                cbar_size="7%",
                cbar_pad=0.15,
                aspect=False
            )

            if self.madgraph:
                df = madgraph
            else:
                values = []
                columns = coefficients if self.dimension is 2 else sorted(config['coefficients'])
                for column in columns:
                    if column in coefficients:
                        values += [np.linspace(madgraph[column].min(), madgraph[column].max(), self.numvalues)]
                    else:
                        values += [np.zeros(1)]
                df = scan.dataframe(columns, evaluate_points=cartesian_product(*values))

            for ax, process in zip(grid, self.processes):
                columns = list(coefficients) + [process]
                data = df[columns]
                data[x] *= x_conv
                data[y] *= y_conv

                msize = 200 if self.madgraph else 25
                marker = 'o' if self.madgraph else 's'
                scatter = ax.scatter(
                        data[x].tolist(),
                        data[y].tolist(),
                        c=data[process],
                        norm=norm,
                        s=msize,
                        marker=marker,
                        cmap=masked_map,
                        edgecolors='face'
                )

                ax.scatter(
                        [0.0],
                        [0.0],
                        c='gray',
                        s=300,
                        marker='o',
                        label='SM'
                )
                ax.set_ylabel(y_label, horizontalalignment='right', y=1.0)
                ax.set_xlim([data[x].min(), data[x].max()])
                ax.set_ylim([data[y].min(), data[y].max()])
                ax.annotate(
                    label[process],
                    xy=(0.5, 0.9),
                    xycoords='axes fraction',
                    horizontalalignment='center',
                    bbox=dict(boxstyle="round,pad=.5", fc="white", ec="none")
                )

            bar = fig.colorbar(
                    scatter,
                    cax=ax.cax,
                    label='$\sigma_{NP+SM} / \sigma_{SM}$ ' + '({}d fit)'.format(self.dimension) if not self.madgraph else '',
                    ticks=LogLocator(subs=range(10)),
                )
            # there is bug in this version of matplotlib ignores zorder, so redraw ticklines
            for t in ax.cax.yaxis.get_ticklines():
                ax.cax.add_artist(t)

            ax.legend(fancybox=True)

            logging.info('saving {}'.format(name))
            ax.set_xlabel(x_label, horizontalalignment='right', x=1.0)
            plt.savefig(os.path.join(config['outdir'], 'plots', '{}.pdf'.format(name)), bbox_inches='tight')
            plt.savefig(os.path.join(config['outdir'], 'plots', '{}.png'.format(name)), bbox_inches='tight')
            plt.close()


class NewPhysicsScaling(Plot):

    def __init__(
            self,
            processes=[('ttZ', '+', '#2fd164')],
            subdir='scaling',
            overlay_result=False,
            dimensionless=False,
            match_nll_window=True,
            points=300):
        self.subdir = subdir
        self.processes = processes
        self.overlay_result = overlay_result
        self.dimensionless = dimensionless
        self.match_nll_window = match_nll_window
        self.points = points

    def specify(self, config, spec, index):
        inputs = ['cross_sections.npz']
        if self.match_nll_window:
            inputs = multidim_np(config, spec, 1, points=self.points)

        for coefficient in config['coefficients']:
            spec.add(inputs, [], ['run', 'plot', '--coefficients', coefficient, '--index', index, config['fn']])

    def write(self, config, plotter, args):
        super(NewPhysicsScaling, self).write(config)
        # FIXME does this work letting WQ transfer?
        scan = CrossSectionScan(os.path.join(config['outdir'], 'cross_sections.npz'))
        if self.match_nll_window:
            nll = fit_nll(config, transform=False, dimensionless=self.dimensionless)

        for coefficient in config['coefficients']:
            conv = 1. if self.dimensionless else conversion[coefficient]
            if not np.any([p in scan.points[coefficient] for p, _, _ in self.processes]):
                continue
            with plotter.saved_figure(
                    label[coefficient] + ('' if self.dimensionless else r' $/\Lambda^2$'),
                    '$\sigma_{NP+SM} / \sigma_{SM}$',
                    os.path.join(self.subdir, coefficient)) as ax:

                for process, marker, c in self.processes:
                    x = scan.points[coefficient][process]
                    y = scan.scales(coefficient, process)
                    if self.match_nll_window:
                        xmin = nll[coefficient]['x'][nll[coefficient]['y'] < 13].min()
                        xmax = nll[coefficient]['x'][nll[coefficient]['y'] < 13].max()
                    else:
                        xmin = min(x * conv)
                        xmax = max(x * conv)

                    xi = np.linspace(xmin, xmax, 10000).reshape(10000, 1)
                    ax.plot(xi * conv, scan.evaluate(coefficient, xi, process), color='#C6C6C6')
                    ax.plot(x * conv, y, marker, mfc='none', markeredgewidth=2, markersize=15, label=label[process],
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
                plt.title(r'CMS Simulation', loc='left', fontweight='bold')
                plt.title(r'MG5_aMC@NLO LO', loc='right', size=27)
                ax.legend(loc='upper center')
                if self.match_nll_window:
                    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))


class NLL2D(Plot):

    def __init__(self, subdir='nll2d', dimensionless=False, scatter=False, maxnll=12, vmin=0.02, points=40000):
        self.subdir = subdir
        self.dimensionless = dimensionless
        self.scatter = scatter
        self.maxnll = maxnll
        self.vmin = vmin
        self.points = points

    def specify(self, config, spec, index):
        inputs = multidim_np(config, spec, 2, points=self.points)

        for coefficients in sorted_combos(config['coefficients'], 2):
            cmd = 'run plot --coefficients {coefficients} --index {index} {fn}'
            spec.add(inputs, [], cmd.format(coefficients=' '.join(coefficients), index=index, fn=config['fn']))

    def write(self, config, plotter, args):
        super(NLL2D, self).write(config)

        levels = sorted(chi2.isf([0.05, 0.32], 2))
        labels = ['68% CL', '95% CL']
        for coefficients in sorted_combos(config['coefficients'], 2):
            tag = '_'.join(coefficients)
            try:
                data = root2array(os.path.join(config['outdir'], 'scans', '{}.total.root'.format(tag)))
            except IOError as e:
                print 'input data missing, skipping {}'.format(tag)
                continue

            x = coefficients[0]
            y = coefficients[1]
            zi = 2 * data['deltaNLL']
            xi = data[x]
            yi = data[y]
            x_label = label[x] + ('' if self.dimensionless else r' $/\Lambda^2$')
            y_label = label[y] + ('' if self.dimensionless else r' $/\Lambda^2$')
            if not self.dimensionless:
                xi *= conversion[x]
                yi *= conversion[y]
                x_label += '$\ [\mathrm{TeV}^{-2}]$'
                y_label += '$\ [\mathrm{TeV}^{-2}]$'
            with plotter.saved_figure(
                    x_label,
                    y_label,
                    os.path.join(self.subdir, tag),
                    header=config['header'],
                    figsize=(15, 11)) as ax:

                contour = plt.tricontour(
                    xi[zi != 0],
                    yi[zi != 0],
                    zi[zi != 0],
                    levels,
                    colors=['black', 'black'],
                    linestyles=['--', '-']
                )
                for i, l in enumerate(labels):
                    contour.collections[i].set_label(l)

                plt.plot(
                    xi[zi.argmin()],
                    yi[zi.argmin()],
                    mew=3,
                    marker="x",
                    linestyle='None',
                    color='black',
                    label='best fit',
                    zorder=10
                )

                if self.scatter:
                    xmin = xi[zi < self.maxnll].min()
                    xmax = xi[zi < self.maxnll].max()
                    ymin = yi[zi < self.maxnll].min()
                    ymax = yi[zi < self.maxnll].max()
                    window = (xi > xmin) & (xi < xmax) & (yi > ymin) & (yi < ymax)

                    np.clip(zi, self.vmin, zi.max(), zi)
                    scatter = ax.scatter(
                        xi[window],
                        yi[window],
                        c=zi[window],
                        norm=matplotlib.colors.LogNorm(vmin=self.vmin, vmax=zi[window].max()),
                        s=600,
                        marker='s',
                        linewidths=0,
                        # cmap=sns.diverging_palette(240, 10, s=99, l=55, sep=1, as_cmap=True)
                        cmap=get_stacked_colormaps(
                            [sns.light_palette("red", reverse=True, as_cmap=True), sns.light_palette("blue", as_cmap=True)],
                            interfaces=levels[:-1],
                            norm=matplotlib.colors.LogNorm(vmin=self.vmin, vmax=zi[window].max())
                        )
                    )
                    bar = plt.colorbar(
                        scatter,
                        label='$-2\ \Delta\ \mathrm{ln}\ \mathrm{L}$' + (' (asimov data)' if config['asimov data'] else ''),
                        ticks=LogLocator(subs=range(10))
                    )
                    for t in bar.ax.get_yticklines():
                        bar.ax.add_artist(t)
                ax.legend(fancybox=True, ncol=3)
                plt.ylim(ymin=yi[zi < self.maxnll].min(), ymax=yi[zi < self.maxnll].max())
                plt.xlim(xmin=xi[zi < self.maxnll].min(), xmax=xi[zi < self.maxnll].max())
                # ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))


class NLL(Plot):

    def __init__(self, subdir='nll', transform=True, dimensionless=False, points=300):
        self.subdir = subdir
        self.transform = transform
        self.dimensionless = dimensionless
        self.points = points

    def specify(self, config, spec, index):
        inputs = multidim_np(config, spec, 1, points=self.points)

        for coefficient in config['coefficients']:
            spec.add(inputs, [], ['run', 'plot', '--coefficient', coefficient, '--index', index, config['fn']])

    def write(self, config, plotter, args):
        super(NLL, self).write(config)
        data = fit_nll(config, self.transform, self.dimensionless)
        scan = CrossSectionScan(os.path.join(config['outdir'], 'cross_sections.npz'))

        for coefficient in config['coefficients']:
            info = data[coefficient]
            for p in config['processes']:
                s0, s1, s2 = scan.construct(p, [coefficient])
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

    def specify(self, config, spec, index):
        inputs = multi_signal(self.signals, self.tag, spec, config)
        for signal in self.signals:
            inputs += max_likelihood_fit(signal, spec, config)
        if self.contours:
            inputs += multidim_grid(config, self.tag, self.numpoints, self.chunksize, spec)

        spec.add(inputs, [],  ['run', 'plot', '--index', index, config['fn']])

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

    def __init__(self, subdir='.', signals=['ttW', 'ttZ'], theory_errors=None, tag=None, transform=True, dimensionless=False, points=300):
        self.subdir = subdir
        self.signals = signals
        self.theory_errors = theory_errors
        if tag:
            self.tag = tag
        else:
            self.tag = '-'.join(signals)
        self.transform = transform
        self.dimensionless = dimensionless
        self.points = points

    def specify(self, config, spec, index):
        inputs = multi_signal(self.signals, self.tag, spec, config)
        inputs += multidim_np(config, spec, 1, points=self.points)
        inputs += fluctuate(config, spec)

        spec.add(inputs, [],  ['run', 'plot', '--index', index, config['fn']])

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

    if args.coefficients:
        config['coefficients'] = args.coefficients

    if args.header:
        config['header'] = args.header

    if args.index is not None:
        config['plots'][args.index].write(config, plotter, args)
    else:
        for p in config['plots']:
            p.write(config, plotter, args)
