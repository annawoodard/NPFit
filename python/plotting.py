import atexit
import contextlib
from datetime import datetime
import glob
import logging
import os
import shutil
import tarfile

import jinja2
import matplotlib
from matplotlib.mlab import griddata
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Polynomial

label = {
    'sigma ttW': r'$\sigma_{\mathrm{t\bar{t}W}}$ $\mathrm{[fb]}$',
    'sigma ttZ': r'$\sigma_{\mathrm{t\bar{t}Z}}$ $\mathrm{[fb]}$',
    'ttW': r'$\mathrm{t\bar{t}W}}$',
    'ttZ': r'$\mathrm{t\bar{t}Z}}$',
    'cuB': r'$\bar{c}_{uB}$',
    'cHQ': r'$\bar{c}_{HQ}$',
    'cHu': r'$\bar{c}_{Hu}$',
    'c3W': r'$\bar{c}_{3W}$',
    'cpHQ': r"$\bar{c}'_{HQ}$",
    'c2W': r'$\bar{c}_{2W}$',
    'c3G': r'$\bar{c}_{3G}$',
    'cA': r'$\bar{c}_{A}$',
    'cB': r'$\bar{c}_{B}$',
    'cG': r'$\bar{c}_{G}$',
    'cHB': r'$\bar{c}_{HB}$',
    'cHW': r'$\bar{c}_{HW}$',
    'cHd': r'$\bar{c}_{Hd}$',
    'cHud': r'$\bar{c}_{Hud}$',
    'cT': r'$\bar{c}_{T}$',
    'cWW': r'$\bar{c}_{WW}$',
    'cu': r'$\bar{c}_{u}$',
    'cuG': r'$\bar{c}_{uG}$',
    'cuW': r'$\bar{c}_{uW}$',
    'tc3G': r'$\widetilde{c}_{3G}$',
    'tc3W': r'$\widetilde{c}_{3W}$',
    'tcG': r'$\widetilde{c}_{G}$',
    'tcHW': r'$\widetilde{c}_{HW}$',
}

cross_sections = {
    'ttW': 601.,
    'ttZ': 839.,
    'ttH': 496.
}


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

        for root, dirs, files in os.walk(os.path.abspath(self.config['outdir'])):
            for path in [os.path.join(root, d) for d in dirs] + [root]:
                pngs = glob.glob(os.path.join(path, '*.png'))
                pdfs = glob.glob(os.path.join(path, '*.pdf'))
                plots = [os.path.splitext(os.path.basename(name))[0] for name in pngs]
                if plots:
                    plots.sort()

                    tfilename = os.path.join(path, 'plots.tar.gz')
                    tfile = tarfile.open(tfilename, 'w:gz')
                    [tfile.add(f) for f in [os.path.join(path, name) for name in pngs + pdfs]]
                    tfile.close()

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

    @contextlib.contextmanager
    def saved_figure(self, x_label, y_label, name, header=False):
        fig, ax = plt.subplots(figsize=(11, 11))
        if header:
            if header is 'preliminary':
                plt.title(r'CMS preliminary', loc='left', fontweight='bold')
            else:
                plt.title(r'CMS preliminary', loc='left', fontweight='bold')

            plt.title('{}'.format(self.config['luminosity']) + ' fb$^{-1}$ (13 TeV)', loc='right', fontweight='bold')

        try:
            yield ax

        finally:
            logging.info('saving {}'.format(name))
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.savefig(os.path.join(self.config['outdir'], '{}.pdf'.format(name)), bbox_inches='tight')
            plt.savefig(os.path.join(self.config['outdir'], '{}.png'.format(name)), bbox_inches='tight')
            plt.close()


class NumPyPlotter(Plotter):
    def hist(self, data, num_bins, xlim, xlabel, title, name):
        data[data > xlim] = xlim
        data[data < (xlim * -1)] = xlim * -1

        info = u"$\mu$ = {0:.3g}\n$\sigma$ = {1:.3g}\nmedian = {2:.3g}"

        with self.saved_figure(xlabel, 'counts', title, name) as ax:
            ax.hist(data, bins=num_bins)
            ax.text(
                0.95, 0.95,
                info.format(np.average(data), np.std(data), np.median(data)),
                horizontalalignment='right',
                verticalalignment='top',
                transform=ax.transAxes,
                backgroundcolor='white'
            )

    def plot(self, data, xlabel, ylabel, name, series_labels=None, title=None):
        with self.saved_figure(xlabel, ylabel, name, title) as ax:
            if series_labels:
                for (x, y), l in zip(data, series_labels):
                    ax.plot(x, y, 'o', label=l)
                ax.legend(numpoints=1)
            else:
                for (x, y) in data:
                    ax.plot(x, y, 'o')


def fit_nll(config):
    from root_numpy import root2array
    import scipy.signal

    def slopes(x, y):
        rise = y[1:] - y[:-1]
        run = x[1:] - x[:-1]

        return rise / run

    def intercepts(x, y):
        return y[1:] - slopes(x, y) * x[1:]

    def crossings(x, y, q):
        crossings = (q - intercepts(x, y)) / slopes(x, y)

        return crossings[(crossings > x[:-1]) & (crossings < x[1:])]

    def interval(x, y, q, p, precision=2):
        points = crossings(x, y, q)
        for low, high in [points[i:i + 2] for i in range(0, len(points), 2)]:
            if p > low and p < high:
                return (round(low, precision) + 0, round(high, precision) + 0)  # turn -0.00 -> 0.00

    def intervals(x, y, q):
        points = crossings(x, y, q)

        return ((points[i:i + 2], [q, q]) for i in range(0, len(points), 2))

    res = {}
    for operator in config['operators']:
        data = root2array(os.path.join(config['outdir'], 'scans', '{}.total.root'.format(operator)))
        data = data[data['deltaNLL'] < 5]
        _, unique = np.unique(data[operator], return_index=True)

        x = data[unique][operator]
        y = 2 * data[unique]['deltaNLL']
        minima = scipy.signal.argrelmin(y)
        threshold = y[minima] - min(y) < 0.1

        res[operator] = {
            'x': x,
            'y': y,
            'best fit': [],
            'one sigma': [],
            'two sigma': []
        }

        for xbf, ybf in zip(x[minima][threshold], y[minima][threshold]):
            res[operator]['best fit'].append((xbf, ybf))
            if interval(x, y, 1.0, xbf) not in res[operator]['one sigma']:
                res[operator]['one sigma'].append(interval(x, y, 1.0, xbf))
            if interval(x, y, 3.84, xbf) not in res[operator]['two sigma']:
                res[operator]['two sigma'].append(interval(x, y, 3.84, xbf))

    return res


def xsecs(config, plotter):
    nll = fit_nll(config)
    data = {}
    series_labels = {}

    fn = os.path.join(config['outdir'], 'cross_sections.npy')
    for process, info in np.load(fn)[()].items():
        coefficients = info['coefficients']
        cross_section = info['cross section']
        sm_coefficients = np.array([tuple([0.0] * len(coefficients.dtype))], dtype=coefficients.dtype)
        sm_cross_section = np.mean(cross_section[coefficients == sm_coefficients])

        for operator in coefficients.dtype.names:
            x = coefficients[operator][coefficients[operator] != 0]
            y = cross_section[coefficients[operator] != 0]

            try:
                data[operator].append((x, y / sm_cross_section))
                series_labels[operator].append(process)
            except KeyError:
                data[operator] = [(x, y / sm_cross_section)]
                series_labels[operator] = [process]

    for operator in data.keys():
        with plotter.saved_figure(
                label[operator],
                '$\sigma_{NP+SM} / \sigma_{SM}$',
                os.path.join(config['outdir'], 'plots', 'cross_sections', 'ratios', operator)) as ax:
            for (x, y), l in zip(data[operator], series_labels[operator]):
                ax.plot(x, y, 'o', label=l)

            if operator in config['operators']:
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
            ax.legend(loc='upper center')

        # FIXME: add fits
        # x = np.linspace(-1, 1, 100)
        # y = fits[operator]['ttZ'](x)
        # plt.plot(x, y)


def nll(config, plotter):
    data = fit_nll(config)

    for operator, info in data.items():
        with plotter.saved_figure(label[operator], '-2 $\Delta$ ln L', os.path.join(config['outdir'], 'plots', 'nll', operator), '') as ax:
            ax.plot(info['x'], info['y'], 'o')

            for x, y in info['best fit']:
                ax.plot(x, y, 'o', c='#fc4f30', label='best fit: {:03.2f}'.format(x))

            for low, high in info['one sigma']:
                ax.plot([low, high], [1.0, 1.0], '-', label='$1\sigma$ CL [{:03.2f}, {:03.2f}]'.format(low, high))

            for low, high in info['two sigma']:
                ax.plot([low, high], [3.84, 3.84], '-', label='$2\sigma$ CL [{:03.2f}, {:03.2f}]'.format(low, high))

            ax.legend(loc='upper center')


def ttZ_ttW_2D(config, ax):
    from root_numpy import root2array
    limits = root2array(os.path.join('scans', 'ttZ_ttW_2D.total.root'))

    x = limits['r_ttW'] * cross_sections['ttW']
    y = limits['r_ttZ'] * cross_sections['ttZ']
    z = 2 * limits['deltaNLL']

    extent = (0, 1800, 0, 2200)
    x_min, x_max, y_min, y_max = extent

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
        marker="+",
        linestyle='None'
    )
    handles.append(bf)
    labels.append('2D best fit')
    print 'best fit ', x[z.argmin()], y[z.argmin()]

    ttW_theory_xsec = plt.axvline(x=cross_sections['ttW'], linestyle='--', color='black')
    ttW_theory_error = ax.axvspan(
        cross_sections['ttW'] - cross_sections['ttW'] * 0.12,
        cross_sections['ttW'] + cross_sections['ttW'] * 0.10,
        edgecolor='#555555',
        fill=False,
        linewidth=0.0,
        hatch='//'
    )
    handles.append((ttW_theory_xsec, ttW_theory_error))
    labels.append('{} theory'.format(label['ttW']))

    ttZ_theory_xsec = plt.axhline(y=cross_sections['ttZ'], linestyle='--', color='black')
    ttZ_theory_error = ax.axhspan(
        cross_sections['ttZ'] - cross_sections['ttZ'] * 0.12,
        cross_sections['ttZ'] + cross_sections['ttZ'] * 0.10,
        edgecolor='#555555',
        fill=False,
        linewidth=0.0,
        zorder=2,
        hatch='\\'
    )
    handles.append((ttZ_theory_xsec, ttZ_theory_error))
    labels.append('{} theory'.format(label['ttZ']))

    return handles, labels


def ttZ_ttW_2D_1D_ttZ_1D_ttW(config, plotter):
    from root_numpy import root2array
    with plotter.saved_figure(
            label['sigma ttW'],
            label['sigma ttZ'],
            'plots/ttZ_ttW_2D_1D_ttZ_1D_ttW',
            header=config['plot header']) as ax:
        handles, labels = ttZ_ttW_2D(config, ax)

        data = root2array('ttW.root')

        print 'ttW 1D: ', data['limit'][0] * cross_sections['ttW']
        ttW_1D_xsec = plt.axvline(x=data['limit'][0] * cross_sections['ttW'], color='black')
        ttW_1D_error = ax.axvspan(
            data['limit'][1] * cross_sections['ttW'],
            data['limit'][2] * cross_sections['ttW'],
            alpha=0.5,
            color='#FA6900',
            linewidth=0.0
        )
        handles.append((ttW_1D_xsec, ttW_1D_error))
        labels.append('{} 1D $\pm$ $1\sigma$'.format(label['ttW']))

        data = root2array('ttZ.root')

        print 'ttZ 1D: ', data['limit'][0] * cross_sections['ttZ']
        ttZ_1D_xsec = plt.axhline(y=data['limit'][0] * cross_sections['ttZ'], color='black')
        ttZ_1D_error = ax.axhspan(
            data['limit'][1] * cross_sections['ttZ'],
            data['limit'][2] * cross_sections['ttZ'],
            color='#69D2E7',
            alpha=0.5,
            linewidth=0.0
        )
        handles.append((ttZ_1D_xsec, ttZ_1D_error))
        labels.append('{} 1D $\pm$ $1\sigma$'.format(label['ttZ']))

        plt.legend(handles, labels, fontsize=15)


def load_fits(config):
    fits = {}

    fn = os.path.join(config['outdir'], 'cross_sections.npy')
    print '{:10} {:14} {:12} {:14}'.format('process', '       NLO', '   LO', 'scale factor')
    for process in ['ttZ', 'ttW']:
        info = np.load(fn)[()][process]
        coefficients = info['coefficients']
        cross_section = info['cross section'] * 1000.0
        sm_coefficients = np.array([tuple([0.0] * len(coefficients.dtype))], dtype=coefficients.dtype)
        sm_cross_section = np.mean(cross_section[coefficients == sm_coefficients])

        for operator in config['operators']:
            x = coefficients[operator][coefficients[operator] != 0]
            y = cross_section[coefficients[operator] != 0] * cross_sections[process] / sm_cross_section

            if operator == 'cpHQ':
                print '{:10} {:-10.0f} {:10.0f} {:10.2f}'.format(process, cross_sections[process], sm_cross_section, cross_sections[process] / sm_cross_section)
            try:
                fits[operator][process] = Polynomial.fit(x, y, 2)
            except KeyError:
                fits[operator] = {process: Polynomial.fit(x, y, 2)}

    return fits


def ttZ_ttW_2D_1D_eff_op(config, plotter):
    from root_numpy import root2array
    fits = load_fits(config)

    for operator in config['operators']:
        with plotter.saved_figure(
                label['sigma ttW'],
                label['sigma ttZ'],
                'plots/ttZ_ttW_2D_1D_{}'.format(operator),
                header=config['plot header']) as ax:
            handles, labels = ttZ_ttW_2D(config, ax)

            data = root2array('best-fit-{}.root'.format(operator))

            # p = np.linspace(data[operator][1], data[operator][2], 1000)
            # x = fits[operator]['ttW'](p)
            # y = fits[operator]['ttZ'](p)
            # band = plt.fill_between(x, y - y * 0.1, y + y * 0.1)
            # line = plt.plot(x, y, color='#A8D46F')
            # plt.fill_betweenx(y, x - x * 0.1, x + x * 0.1)

            bf, = plt.plot(
                fits[operator]['ttW'](data[operator][0]),
                fits[operator]['ttZ'](data[operator][0]),
                color='black',
                mew=3,
                markersize=17,
                marker="*",
                linestyle='None'
            )
            handles.append(bf)
            labels.append(
                'best fit {}\n{:03.2f} [{:03.2f}, {:03.2f}]'.format(
                    label[operator],
                    round(data[operator][0], 2) + 0,
                    data[operator][1],
                    data[operator][2]
                )
            )

            plt.legend(handles, labels, fontsize=15)


def wilson_coefficients_in_window(config, plotter):

    fits = load_fits(config)

    for operator in config['operators']:
        with plotter.saved_figure(
                label['sigma ttW'],
                label['sigma ttZ'],
                'plots/ttZ_ttW_2D_cross_section_with_sampled_{}'.format(operator),
                header=config['plot header']) as ax:
            ttZ_ttW_2D(config, ax)

            ax.set_color_cycle([plt.cm.cool(i) for i in np.linspace(0, 1, 28)])

            high = min(max((fits[operator]['ttW'] - 1000).roots().real), max((fits[operator]['ttZ'] - 1000).roots().real))
            low = max(min((fits[operator]['ttW'] - 1000).roots().real), min((fits[operator]['ttZ'] - 1000).roots().real))
            coefficients = np.linspace(low, high, 28)
            # avoid overlapping due to symmetric points
            for coefficient in np.hstack([coefficients[:14:2], coefficients[14::2]]):
                plt.plot(
                    fits[operator]['ttW'](coefficient),
                    fits[operator]['ttZ'](coefficient),
                    marker='x',
                    markersize=17,
                    mew=3,
                    label='{}={:03.2f}'.format(operator, coefficient),
                    linestyle="None"
                )

            ax.legend(numpoints=1, bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0., frameon=False)


def plot(args, config):

    from root_numpy import root2array
    plotter = NumPyPlotter(config)

    nll(config, plotter)
    xsecs(config, plotter)


    wilson_coefficients_in_window(config, plotter)
    ttZ_ttW_2D_1D_ttZ_1D_ttW(config, plotter)
    ttZ_ttW_2D_1D_eff_op(config, plotter)
