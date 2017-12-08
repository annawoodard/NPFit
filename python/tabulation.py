import os
import tabulate as tb
import numpy as np
from NPFit.NPFit.makeflow import multidim_np
from NPFit.NPFit.parameters import conversion, label
from root_numpy import root2array
from NPFitProduction.NPFitProduction.utils import sorted_combos


class CLIntervals(object):
    def __init__(self, dimension, dimensionless=False, levels=[0.68, 0.95]):
        self.dimension = dimension
        self.dimensionless = dimensionless
        self.levels = levels
        self.base = 'cl_intervals_{}d_{}{}'.format(
            dimension,
            '_'.join(['cl{}'.format(x) for x in levels]),
            '_dimensionless' if dimensionless else ''
        )

    def specify(self, config, spec, index):
        inputs = multidim_np(config, spec, self.dimension, tasks=None, cl=self.levels)
        outputs = [os.path.join(config['outdir'], self.base + extension) for extension in ['.tex', '.txt']]
        spec.add(inputs, outputs, ['run', 'tabulate', '-i', index, config['fn']])

    def write(self, config, args):
        precision = 3 if self.dimensionless else 2
        if self.dimension == 2:
            tables = {}
            indexes = dict((c, i) for i, c in enumerate(sorted(config['coefficients'])))
            headers = sorted(config['coefficients'])

            def get_text(low, high, table, tablefmt, cellfmt, headers):
                txt_table = []
                for row in table:
                    txt_row = [row[0]]
                    for cell in row[1:]:
                        if cell is None:
                            txt_row += ['-']
                        else:
                            txt_row += [cellfmt.format(*cell[low:high], p=precision)]
                    txt_table += [txt_row]

                return tb.tabulate(txt_table, headers=headers, tablefmt=tablefmt)

            for level in self.levels:
                tables[level] = np.empty(
                    (len(config['coefficients']) - 1, len(config['coefficients'])),
                    dtype=object
                )
                tables[level][:, 0] = headers[:-1]

            for level in self.levels:
                for coefficients in sorted_combos(config['coefficients'], 2):
                    tag = '_'.join(coefficients)
                    x = coefficients[0]
                    y = coefficients[1]
                    data = root2array(
                        os.path.join(config['outdir'], 'cl_intervals/{}-{}.root'.format(tag, level))
                    )
                    cell = (
                        data[x][0] * (1. if self.dimensionless else conversion[x]),
                        data[y][0] * (1. if self.dimensionless else conversion[y]),
                        data[x][1] * (1. if self.dimensionless else conversion[x]),
                        data[x][2] * (1. if self.dimensionless else conversion[x]),
                        data[y][3] * (1. if self.dimensionless else conversion[y]),
                        data[y][4] * (1. if self.dimensionless else conversion[x])
                    )
                    tables[level][indexes[x]][indexes[y]] = cell

            with open(os.path.join(config['outdir'], self.base + '.txt'), 'w') as f:
                cell = '[{:.{p}f}, {:.{p}f}] / [{:.{p}f}, {:.{p}f}]'
                table_headers = [""] + [x + ('' if self.dimensionless else '/Lambda^2') for x in headers[1:]]
                for level, table in tables.items():
                    f.write('\n\n{line} cl={level} {line}\n{text}'.format(
                        line='#' * 20,
                        level=level,
                        text=get_text(2, 6, table.tolist(), 'plain', cell, table_headers))
                    )
                f.write('\n\n{line} best fit {line}\n{text}'.format(
                    line='#' * 20,
                    text=get_text(0, 2, table.tolist(), 'plain', '{:.{p}f} / {:.{p}f}', table_headers))
                )
            with open(os.path.join(config['outdir'], self.base + '.tex'), 'w') as f:
                cell = '\\diagbox{{[{:.{p}f}, {:.{p}f}]}}{{\\textcolor{{red}}{{[{:.{p}f}, {:.{p}f}]}}}}'
                table_headers = [""] + [
                    '\\textcolor{{red}}{{{}}}'.format(label[x] + ('' if self.dimensionless else '$/\Lambda^2$'))
                    for x in headers[1:]
                ]
                for level, table in tables.items():
                    table[:, 0] = [label[x] for x in table[:, 0]]
                    f.write('\n\n{line} cl={level} {line}\n{text}'.format(
                        line='#' * 20,
                        level=level,
                        text=get_text(2, 6, table.tolist(), 'latex_raw', cell, table_headers)
                        )
                    )
                cell = '\\diagbox{{{:.{p}f}}}{{\\textcolor{{red}}{{{:.{p}f}}}}}'
                f.write('\n\n{line} best fit {line}\n{text}'.format(
                    line='#' * 20,
                    text=get_text(0, 2, table.tolist(), 'plain', cell, table_headers)
                    )
                )
        else:
            table = []
            for coefficients in sorted_combos(config['coefficients'], self.dimension):
                for index, coefficient in enumerate(sorted(coefficients)):
                    row = [coefficient]
                    for level in self.levels:
                        tag = '_'.join(coefficients)
                        data = root2array(
                            os.path.join(config['outdir'], 'cl_intervals/{}-{}.root'.format(tag, level))
                        )
                        template = '[{:.{p}f}, {:.{p}f}]'
                        conversion_factor = 1. if self.dimensionless else conversion[coefficient]
                        low = data[coefficient][(index + 1) * 2 - 1] * conversion_factor
                        high = data[coefficient][(index + 1) * 2] * conversion_factor
                        row += [template.format(low, high, p=precision)]
                    row += ['{:.{p}f}'.format(data[coefficient][0], p=precision + 1)]
                    table.append(row)

            headers = [''] + ['CL={}'.format(cl) for cl in self.levels] + ['best fit']
            with open(os.path.join(config['outdir'], self.base + '.txt'), 'w') as f:
                f.write('dimension: {} {}\n'.format(self.dimension, '' if self.dimensionless else '(all values in [TeV^(-2)])'))
                f.write(tb.tabulate(
                    [[x[0] + ('' if self.dimensionless else '/\Lambda^2')] + x[1:] for x in table],
                    headers,
                    tablefmt='plain') + '\n'
                )
            with open(os.path.join(config['outdir'], self.base + '.tex'), 'w') as f:
                table = [[label[x[0]] + ('' if self.dimensionless else '$/\Lambda^2$')] + x[1:] for x in table]
                f.write('dimension: {} {}\n'.format(self.dimension, '' if self.dimensionless else '(all values in [TeV^(-2)])'))
                f.write(tb.tabulate(table, headers, tablefmt='latex_raw') + '\n')

def tabulate(args, config):
    if args.index is not None:
        config['tables'][args.index].write(config, args)
    else:
        for p in config['tables']:
            p.write(config, args)
