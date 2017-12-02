import glob
import logging
import os
import re
import shlex
import shutil
import subprocess

import numpy as np

from NPFit.NPFit.actionable import annotate
from NPFitProduction.NPFitProduction.utils import sorted_combos


class MakeflowSpecification(object):

    def __init__(self, config):
        self.rules = []
        self.config = config

    def add(self, inputs, outputs, cmd='run'):
        if isinstance(inputs, basestring):
            inputs = [inputs]
        if isinstance(outputs, basestring):
            outputs = [outputs]
        if isinstance(cmd, basestring):
            cmd = shlex.split(cmd)

        inputs = [self.config] + inputs
        inputs.sort()
        ins = ' '.join(inputs)
        if isinstance(outputs, dict):
            # FIXME hack, check https://github.com/cooperative-computing-lab/cctools/issues/1680 to see if issue with
            # remote file naming + shared-fs gets fixed
            # outs = ' '.join(["{}->{}".format(v, k) for k, v in outputs.items()])
            outs = ' '.join(outputs.values())
            cmd += ['; mv {} {}'.format(k, v) for k, v in outputs.items()]
        else:
            outs = ' '.join(outputs)

        res = (outs, ins, cmd)
        if res not in self.rules:
            self.rules.append(res)

    def dump(self, makefile):
        for outs, ins, cmd in self.rules:
            frag = """\n{out}: {ins}\n\t{cmd}\n"""
            with open(makefile, 'a') as f:
                s = frag.format(
                    out=outs,
                    ins=ins,
                    cmd=' '.join([str(x) for x in cmd]))
                f.write(s)


def prepare_cards(args, config, cardify):
    for analysis, path in config['cards'].items():
        if os.path.isdir(path):
            subprocess.call('combineCards.py {} > {}'.format(os.path.join(path, '*.txt'), cardify(analysis)), shell=True)
        elif os.path.isfile(path):
            shutil.copy(path, cardify(analysis))

    with open(cardify('4l'), 'r') as f:
        card = f.read()
    with open(cardify('4l'), 'w') as f:
        f.write(card[:card.find('nuisance parameters') + 19])
        f.write('''
----------------------------------------------------------------------------------------------------------------------------------
shapes *      ch1  FAKE
shapes *      ch2  FAKE''')
        f.write(card[card.find('nuisance parameters') + 19:])

    subprocess.call('combineCards.py {} {} > {}'.format(cardify('3l'), cardify('4l'), cardify('ttZ')), shell=True)
    subprocess.call('cp {} {}'.format(cardify('2l'), cardify('ttW')), shell=True)
    subprocess.call('combineCards.py {} {} > {}'.format(cardify('ttZ'), cardify('ttW'), cardify('ttV_np')), shell=True)

    with open(cardify('ttV_np'), 'r') as f:
        card = f.read()

    processes = re.compile(r'\nprocess.*')

    for index, process in enumerate(['ttW', 'ttZ']):
        names, numbers = processes.findall(card)
        for column in [i for i, name in enumerate(names.split()) if name == process]:
            number = numbers.split()[column]
            card = card.replace(numbers, numbers.replace(number, '{}'.format(index * -1)))

    jmax = re.search('jmax (\d*)', card).group(0)
    card = card.replace(jmax, 'jmax {}'.format(len(set(names.split()[1:])) - 1))

    with open(cardify('ttW-ttZ'), 'w') as f:
        f.write(card)

    systematics = {}
    for label, info in config['systematics'].items():
        systematics[label] = '\n{label}                  {dist}     '.format(label=label, dist=info['distribution'])

    def compose(kappa):
        if kappa['-'] == kappa['+']:
            return str(kappa['+'])
        else:
            return '{}/{}'.format(kappa['-'], kappa['+'])

    for name in names.split()[1:]:
        for label, info in config['systematics'].items():
            systematics[label] += '{:15s}'.format(compose(info['kappa'][name]) if name in info['kappa'] else '-')

    kmax = re.search('kmax (\d*)', card).group(0)
    card = card.replace(kmax, 'kmax {}'.format(int(re.search('kmax (\d*)', card).group(1)) + 4))

    for line in card.split('\n'):
        if line.startswith('ttX'):
            card = re.sub(line, '#' + line, card)

    with open(cardify('ttV_np'), 'w') as f:
        f.write(card[:card.find('\ntheo group')])
        for line in systematics.values():
            f.write(line)


def max_likelihood_fit(analysis, spec, config):
    workspace = os.path.join(config['outdir'], 'workspaces', '{}.root'.format(analysis))
    card = os.path.join(config['outdir'], '{}.txt'.format(analysis))
    spec.add(card, workspace, ['text2workspace.py', card, '-o', workspace])
    best_fit = os.path.join(config['outdir'], 'best-fit-{}.root'.format(analysis))
    fit_result = os.path.join(config['outdir'], 'fit-result-{}.root'.format(analysis))
    cmd = 'combine -M MaxLikelihoodFit {a} >& {a}.fit.log'.format(a=card)
    outputs = {
        'higgsCombineTest.MaxLikelihoodFit.mH120.root': best_fit,
        'fitDiagnostics.root': fit_result
    }
    spec.add(workspace, outputs, cmd)

    return [best_fit, fit_result]


def multi_signal(signals, tag, spec, config):
    workspace = os.path.join(config['outdir'], 'workspaces', '{}.root'.format(tag))
    card = os.path.join(config['outdir'], '{}.txt'.format(tag))
    cmd = [
        'text2workspace.py', card,
        '-P', 'HiggsAnalysis.CombinedLimit.PhysicsModel:multiSignalModel',
        '-o', workspace
    ] + ['--PO map=.*/{signal}:r_{signal}[1,0,4]'.format(signal=s) for s in signals]
    spec.add([card], workspace, cmd)

    best_fit = os.path.join(config['outdir'], 'best-fit-{}.root'.format(tag))
    fit_result = os.path.join(config['outdir'], 'fit-result-{}.root'.format(tag))
    outputs = {
        'higgsCombineTest.MultiDimFit.mH120.root': best_fit,
        'multidimfit.root': fit_result
    }
    cmd = 'combine -M MultiDimFit {} --autoBoundsPOIs=* --saveFitResult --algo=cross >& {}.fit.log'.format(workspace, card)
    spec.add(workspace, outputs, cmd)

    return [best_fit, fit_result]


def multidim_grid(config, tag, points, chunksize, spec):
    workspace = os.path.join(config['outdir'], 'workspaces', '{}.root'.format(tag))
    lowers = np.arange(1, points - 1, chunksize)
    uppers = np.arange(chunksize, points, chunksize)
    scans = []
    for index, (first, last) in enumerate(zip(lowers, uppers)):
        filename = 'higgsCombine_{}_{}.MultiDimFit.mH120.root'.format(tag, index)
        scan = os.path.join(config['outdir'], 'scans', filename)
        scans.append(scan)

        cmd = [
            'combine',
            '-M', 'MultiDimFit',
            '--saveFitResult',
            workspace,
            '--algo=grid',
            '--points={}'.format(points),
            '-n', '_{}_{}'.format(tag, index),
            '--firstPoint {}'.format(first),
            '--lastPoint {}'.format(last),
            '--autoBoundsPOIs=*'
        ]

        spec.add(workspace, {filename: scan}, cmd)

    outfile = os.path.join(config['outdir'], 'scans', '{}.total.root'.format(tag))
    spec.add(scans, outfile, ['hadd', '-f', outfile] + scans)

    return [outfile]


def multidim_np(config, spec, tasks):
    outfiles = []
    for coefficients in sorted_combos(config['coefficients'], config['dimension']):
        label = '_'.join(coefficients)
        workspace = os.path.join(config['outdir'], 'workspaces', '{}.root'.format(label))
        cmd = [
            'text2workspace.py', os.path.join(config['outdir'], 'ttV_np.txt'),
            '-P', 'NPFit.NPFit.models:eft',
            '--PO', 'scan={}'.format(os.path.join(config['outdir'], 'cross_sections.npz')),
            ' '.join(['--PO process={}'.format(x) for x in config['processes']]),
            ' '.join(['--PO poi={}'.format(x) for x in coefficients]),
            '-o', workspace
        ]

        spec.add(['cross_sections.npz'], workspace, cmd)

        best_fit = os.path.join(config['outdir'], 'best-fit-{}.root'.format(label))
        fit_result = os.path.join(config['outdir'], 'fit-result-{}.root'.format(label))
        cmd = ['run', 'combine'] + list(coefficients) + [config['fn']]
        spec.add([workspace], [best_fit, fit_result], cmd)

        scans = []
        for index in range(int(tasks)):
            scan = os.path.join(config['outdir'], 'scans', '{}_{}.root'.format(label, index))
            scans.append(scan)
            cmd = ['run', 'combine'] + list(coefficients) + ['-i', str(index), config['fn']]

            spec.add(['cross_sections.npz', workspace], scan, cmd)

        outfile = os.path.join(config['outdir'], 'scans', '{}.total.root'.format(label))
        spec.add(scans, outfile, ['hadd', '-f', outfile] + scans)
        outfiles += [outfile]

    return outfiles


def fluctuate(config, spec):
    outfiles = []
    for coefficients in sorted_combos(config['coefficients'], config['dimension']):
        label = '_'.join(coefficients)
        fit_result = os.path.join(config['outdir'], 'fit-result-{}.root'.format(label))
        cmd = ['run', 'fluctuate', label, config['fluctuations'], config['fn']]
        outfile = os.path.join(config['outdir'], 'fluctuations-{}.npy'.format(label))
        spec.add([fit_result], outfile, cmd)
        outfiles += [outfile]

    return outfiles


def make(args, config):
    def cardify(name):
        return os.path.join(config['outdir'], '{}.txt'.format(name))

    if os.path.isfile(os.path.join(config['outdir'], 'config.py')):
        raise ValueError('refusing to overwrite outdir {}'.format(config['outdir']))

    shutil.copy(args.config, config['outdir'])

    prepare_cards(args, config, cardify)

    makefile = os.path.join(config['outdir'], 'Makeflow')
    logging.info('writing Makeflow file to {}'.format(config['outdir']))

    annotate(args, config)

    spec = MakeflowSpecification(config['fn'])

    # adding annotate to the makeflow file without inputs or outputs
    # forces makeflow to run it everytime makeflow is run: this way new
    # code changes are always picked up
    spec.add([], [], ['run', 'annotate', config['fn']])

    files = glob.glob(os.path.join(config['indir'], '*.root'))
    for f in files:
        outputs = os.path.join('cross_sections', os.path.basename(f).replace('.root', '.npz'))
        spec.add([], outputs, ['run', '--parse', f, config['fn']])

    inputs = [os.path.join('cross_sections', os.path.basename(f).replace('.root', '.npz')) for f in files]
    inputs += glob.glob(os.path.join(config['indir'], '*.npz'))
    spec.add(inputs, 'cross_sections.npz', ['run', 'concatenate', config['fn']])

    for index, plot in enumerate(config['plots']):
        plot.specify(config, spec, index)

    spec.dump(makefile)
