import atexit
import contextlib
from datetime import datetime
import glob
import jinja2
import logging
import os
import shutil
import tarfile
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.mlab import griddata
import numpy as np


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
                    [tfile.add(f) for f in [os.path.join(path, name) for name in pngs+pdfs]]
                    tfile.close()

                dirs = ['..']
                files = []
                names = [x for x in os.listdir(path) if not x.startswith('.') and not '#' in x and x != 'index.html']
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
    def saved_figure(self, x_label, y_label, title, name):
        fig, ax = plt.subplots(figsize=(11,11))

        try:
            yield ax

        finally:
            logging.info('saving {}'.format(name))
            plt.title(title)
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.savefig(os.path.join(self.config['outdir'], '{}.pdf'.format(name)), bbox_inches='tight')
            plt.savefig(os.path.join(self.config['outdir'], '{}.png'.format(name)), bbox_inches='tight')
            plt.close()

class NumPyPlotter(Plotter):
    def hist(self, data, num_bins, xlim, xlabel, title, name):
        data[data > xlim] = xlim
        data[data < (xlim * -1)] = xlim * -1
        info = u"$\mu$ = {0:.3g}\n$\sigma$ = {1:.3g}\nmedian = {2:.3g}".format(np.average(data), np.std(data), np.median(data))

        with self.saved_figure(xlabel, 'counts', title, name) as ax:
            # ax.set_xlim(xlim)
            ax.hist(data, bins=num_bins)
            ax.text(0.95, 0.95, info,
                     horizontalalignment='right', verticalalignment='top', transform=ax.transAxes, backgroundcolor='white')
            # ax.text(0.95, 0.01, 'colored text in axes coords', verticalalignment='bottom', horizontalalignment='right', transform=ax.transAxes, fontsize=15)

    def plot(self, data, xlabel, ylabel, title, name, series_labels=None):
        with self.saved_figure(xlabel, ylabel, title, name) as ax:
            if series_labels:
                for (x, y), l in zip(data, series_labels):
                    ax.plot(x, y, 'o', label=l)
                ax.legend(numpoints=1)
            else:
                for (x, y) in data:
                    ax.plot(x, y, 'o')

    def plot_2d(self, xy, z, xlabel, ylabel, zlabel, title, name, cmap):
        with self.saved_figure(xlabel, ylabel, title, name) as ax:
            x = xy[:,0]
            y = xy[:,1]
            z = z[:,0]

            extent = (min(x), max(x), min(y), max(y))
            xi, yi = np.mgrid[extent[0]:extent[1]:500j, extent[2]:extent[3]:500j]
            zi = griddata(x, y, z, xi, yi)

            plt.imshow(zi.T, extent=extent, aspect='auto', origin='lower', cmap=cmap)
            bar = plt.colorbar()
            bar.set_label(zlabel)

    def scatter_3d(self, xy, z, xlabel, ylabel, zlabel, title, name, **kwargs):
        with self.saved_figure(xlabel, ylabel, title, name) as ax:
            x = xy[:,0]
            y = xy[:,1]
            z = z[:,0]

            plt.scatter(x, y, c=z, s=100, **kwargs)
            if zlabel:
                bar = plt.colorbar()
                bar.set_label(zlabel)

    def scatter(self, x, y, color, xlabel, ylabel, title, name, **kwargs):
        with self.saved_figure(xlabel, ylabel, title, name) as ax:
            plt.scatter(x, y, c=color, s=100, **kwargs)


