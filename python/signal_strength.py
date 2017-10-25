from collections import defaultdict
import os

import numpy as np
from numpy.polynomial import Polynomial
import tabulate

from EffectiveTTVProduction.EffectiveTTVProduction.cross_sections import CrossSectionScan
# TODO only run this once
# TODO do this with roofit instead of numpy, for simpler PhysicsModel

def load(config):
    fn = os.path.join(config['outdir'], 'cross_sections.npz')
    scan = CrossSectionScan([fn])

    return scan.points, scan.cross_sections

def load_mus(config):
    mus = defaultdict(dict)
    fn = os.path.join(config['outdir'], 'cross_sections.npz')
    scan = CrossSectionScan([fn])

    for coefficients in scan.points:
        if len(coefficients) > 1:
            raise NotImplementedError
        for process in scan.points[coefficients]:
            if coefficients == 'sm':
                continue
            x = scan.points[coefficients][process].flatten()
            y = scan.signal_strengths[coefficients][process]

            # FIXME
            # # mu=1 when coefficient=0, make sure the fit goes through that point
            # weights = [1 if (i != 1) else 100000000 for i in y]
            weights = [1 for i in y]
            try:
                mus[coefficients[0]][process] = Polynomial.fit(x, y, 2, w=weights, window=[min(x), max(x)])
            except Exception as e:
                print 'failed to fit {} for process {}: {}'.format(coefficients, process, e)

            # if mus[coefficients][process](1.) / mus[coefficients][process](0.) <= (1 + np.std(y)):
            #     # We can't tell the difference between this and a straight line, let's keep things simple
            #     mus[coefficients][process].coef = (1., 0., 0.)

    return mus

# FIXME improve naming for clarity, maybe 'mus' -> 'scaling'
# FIXME dump a table of coefficient values
def dump_mus(args, config):
    mus = load_mus(config)

    np.save(os.path.join(config['outdir'], 'mus.npy'), mus)

    table = []
    for c in mus:
        for p in mus[c]:
            table.append([c, p] + list(mus[c][p].coef))

    with open(os.path.join(config['outdir'], 'mus.txt'), 'w') as f:
        f.write(tabulate.tabulate(table, headers=['Wilson coefficients', 'process', 's0', 's1', 's2']))

