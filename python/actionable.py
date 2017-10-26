
import glob
import logging
import itertools
import numpy as np
import os
import re
import shlex
import shutil
import subprocess
import stat
import yaml

from EffectiveTTV.EffectiveTTV.signal_strength import dump_mus

from EffectiveTTVProduction.EffectiveTTVProduction.cross_sections import CrossSectionScan

def annotate(args, config):
    """Annotate the output directory with a README

    The README includes instructions to reproduce the current version of
    the code. Any unstaged code changes will be saved as a git patch.
    """
    start = os.getcwd()
    os.chdir(os.path.dirname(__file__))
    head = subprocess.check_output(shlex.split('git rev-parse --short HEAD')).strip()
    diff = subprocess.check_output(shlex.split('git diff'))
    os.chdir(start)

    shared_filesystem = []
    if config['indir shared']:
        if not config['indir'].startswith('/'):
            raise Exception('absolute path required for shared filesystems: {}'.format(config['indir']))
        shared_filesystem += ["--shared-fs '/{}'".format(config['indir'].split('/')[1])]
    if config['outdir shared']:
        if not config['outdir'].startswith('/'):
            raise Exception('absolute path required for shared filesystems: {}'.format(config['outdir']))
        shared_filesystem += ["--shared-fs '/{}'".format(config['outdir'].split('/')[1])]

    info = """
    # to run, issue the following commands:
    cd {outdir}
    nohup work_queue_factory -T {batch_type} -M {label} -C {factory} >& factory.log &

    # then keep running this command until makeflow no longer submits jobs (may take a few tries):
    makeflow -T wq -M {label} --wrapper ./w.sh --wrapper-input w.sh {shared}

    # to reproduce the code:
    cd {code_dir}
    git checkout {head}
    """.format(
            batch_type=config['batch type'],
            outdir=config['outdir'],
            label=config['label'],
            factory=os.path.join(os.environ['LOCALRT'], 'src', 'EffectiveTTV', 'EffectiveTTV', 'data', 'factory.json'),
            shared=' '.join(shared_filesystem),
            code_dir=os.path.dirname(__file__),
            head=head
        )


    if diff:
        with open(os.path.join(config['outdir'], 'patch.diff'), 'w') as f:
            f.write(diff)
        info += 'git apply {}\n'.format(os.path.join(config['outdir'], 'patch.diff'))

    with open(os.path.join(config['outdir'], 'README.txt'), 'w') as f:
        f.write(info)
    logging.info(info)

def make(args, config):
    def cardify(name):
        return os.path.join(config['outdir'], '{}.txt'.format(name))

    def prepare_cards(args, config):
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
        subprocess.call('combineCards.py {} {} > {}'.format(cardify('ttZ'), cardify('ttW'), cardify('2d')), shell=True)
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

        with open(cardify('2d'), 'w') as f:
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

    # Makeflow is a bit picky about whitespace
    wrap = """#!/bin/sh

    source /cvmfs/cms.cern.ch/cmsset_default.sh
    cd {0}
    cmsenv
    cd -
    exec "$@"
    """.format(os.environ["LOCALRT"])

    if os.path.isfile(os.path.join(config['outdir'], 'config.py')):
        raise ValueError('refusing to overwrite outdir {}'.format(config['outdir']))

    configfile = os.path.join(config['outdir'], 'config.py')
    shutil.copy(args.config, configfile)

    wrapfile = os.path.join(config['outdir'], 'w.sh')
    with open(wrapfile, 'w') as f:
        f.write(wrap)
    os.chmod(wrapfile, os.stat(wrapfile).st_mode | stat.S_IEXEC)

    data = os.path.join(os.environ['LOCALRT'], 'src', 'EffectiveTTV', 'EffectiveTTV', 'data')
    shutil.copy(os.path.join(data, 'matplotlibrc'), config['outdir'])

    prepare_cards(args, config)

    makefile = os.path.join(config['outdir'], 'Makeflow')
    logging.info('writing Makeflow file to {}'.format(config['outdir']))

    annotate(args, config)

    frag = """\n{out}: {ins}\n\t{cmd}\n"""
    def makeflowify(inputs, outputs, cmd='run'):
        # FIXME make sure everything works without shared-fs, automate shared-fs
        if isinstance(inputs, basestring):
            inputs = [inputs]
        if isinstance(outputs, basestring):
            outputs = [outputs]
        if isinstance(cmd, basestring):
            cmd = shlex.split(cmd)

        ins = ' '.join(inputs)
        if isinstance(outputs, dict):
            # FIXME hack, check https://github.com/cooperative-computing-lab/cctools/issues/1680 to see if issue with
            # remote file naming + shared-fs gets fixed
            # outs = ' '.join(["{}->{}".format(v, k) for k, v in outputs.items()])
            outs = ' '.join(outputs.values())
            cmd += ['; mv {} {}'.format(k, v) for k, v in outputs.items()]
        else:
            outs = ' '.join(outputs)

        with open(makefile, 'a') as f:
            s = frag.format(
                out=outs,
                ins=ins,
                cmd=' '.join(cmd))
            f.write(s)

    # adding annotate to the makeflow file without inputs or outputs
    # forces makeflow to run it everytime makeflow is run: this way new
    # code changes are always picked up
    makeflowify([], [], ['run', 'annotate', 'config.py'])

    files = glob.glob(os.path.join(config['indir'], '*.root'))
    for f in files:
        outputs = os.path.join('cross_sections', os.path.basename(f).replace('.root', '.npz'))
        makeflowify(['config.py'], outputs, ['run', '--parse', f, 'config.py'])

    inputs = [os.path.join('cross_sections', os.path.basename(f).replace('.root', '.npz')) for f in files] + ['config.py']
    inputs += glob.glob(os.path.join(config['indir'], '*.npz'))
    outputs = 'cross_sections.npz'
    makeflowify(inputs, outputs, ['run', 'concatenate', 'config.py'])

    # makeflowify('ttZ.txt', {'higgsCombineTest.MaxLikelihoodFit.mH120.root': 'ttZ.root'}, 'combine -M MaxLikelihoodFit ttZ.txt')
    # makeflowify('ttW.txt', {'higgsCombineTest.MaxLikelihoodFit.mH120.root': 'ttW.root'}, 'combine -M MaxLikelihoodFit ttW.txt')

    for analysis in ['2l', '3l', '4l', 'ttZ']:
        workspace = os.path.join(config['outdir'], 'workspaces', '{}.root'.format(analysis))
        card = cardify(analysis)
        makeflowify(card, workspace, ['text2workspace.py', card, '-o', workspace])
        best_fit = os.path.join(config['outdir'], 'best-fit-{}.root'.format(analysis))
        fit_result = os.path.join(config['outdir'], 'fit-result-{}.root'.format(analysis))
        cmd = 'combine -M MaxLikelihoodFit {a} >& {a}.fit.log'.format(a=cardify(analysis))
        outputs = {
            'higgsCombineTest.MaxLikelihoodFit.mH120.root': best_fit,
            'fitDiagnostics.root': fit_result
        }
        makeflowify(workspace, outputs, cmd)

    workspace = os.path.join(config['outdir'], 'workspaces', '2d.root')
    cmd = [
        'text2workspace.py', cardify('2d'),
        '-P', 'HiggsAnalysis.CombinedLimit.PhysicsModel:multiSignalModel',
        '--PO', 'map=.*/ttZ:r_ttZ[1,0,4]',
        '--PO', 'map=.*/ttW:r_ttW[1,0,4]',
        '-o', workspace
    ]

    makeflowify([cardify('2d')], workspace, cmd)

    best_fit = os.path.join(config['outdir'], 'best-fit-2d.root')
    fit_result = os.path.join(config['outdir'], 'fit-result-2d.root')
    outputs = {
        'higgsCombineTest.MultiDimFit.mH120.root': best_fit,
        'multidimfit.root': fit_result
    }
    cmd = 'combine -M MultiDimFit {} --saveFitResult --algo=singles >& {}.fit.log'.format(workspace, cardify('2d'))
    makeflowify(workspace, outputs, cmd)

    lowers = np.arange(1, config['2d points'], config['chunk size'])
    uppers = np.arange(config['chunk size'], config['2d points'] + config['chunk size'], config['chunk size'])

    if config['make 2d sm contours']:
        scans = []
        for index, (first, last) in enumerate(zip(lowers, uppers)):
            filename = 'higgsCombine_ttW_ttZ_2D_part_{}.MultiDimFit.mH120.root'.format(index)
            scan = os.path.join(config['outdir'], 'scans', filename)
            scans.append(scan)

            cmd = [
                'combine',
                '-M', 'MultiDimFit',
                '--saveFitResult',
                workspace,
                '--algo=grid',
                '--points={}'.format(config['2d points']),
                '-n', '_ttW_ttZ_2D_part_{}'.format(index),
                '--firstPoint {}'.format(first),
                '--lastPoint {}'.format(last),
            ]

            makeflowify(workspace, {filename: scan}, cmd)

        outfile = os.path.join(config['outdir'], 'scans', '2d.total.root')
        makeflowify(scans, outfile, ['hadd', '-f', outfile] + scans)

    makeflowify(['config.py', 'cross_sections.npz'], 'mus.npy', ['run', 'scale', 'config.py'])

    combinations = [sorted(list(x)) for x in itertools.combinations(config['coefficients'], config['dimension'])]
    for coefficients in combinations:
        label = '_'.join(coefficients)
        workspace = os.path.join(config['outdir'], 'workspaces', '{}.root'.format(label))
        cmd = [
            'text2workspace.py', os.path.join(config['outdir'], 'ttV_np.txt'),
            '-P', 'EffectiveTTV.EffectiveTTV.models:eff_op',
            '--PO', 'scaling={}'.format(os.path.join(config['outdir'], 'mus.npy')),
            ' '.join(['--PO process={}'.format(x) for x in config['processes']]),
            ' '.join(['--PO poi={}'.format(x) for x in coefficients]),
            '-o', workspace
        ]

        makeflowify('mus.npy', workspace, cmd)

        best_fit = os.path.join(config['outdir'], 'best-fit-{}.root'.format(label))
        fit_result = os.path.join(config['outdir'], 'fit-result-{}.root'.format(label))
        cmd = ['run', 'combine'] + list(coefficients) + ['config.py']
        makeflowify(['config.py', workspace, 'mus.npy'], [best_fit, fit_result], cmd)

        cmd = ['run', 'fluctuate', label, '150000', 'config.py']
        makeflowify(['config.py', fit_result], os.path.join(config['outdir'], 'fluctuations-{}.npy'.format(label)), cmd)

        scans = []
        chunks = np.ceil(config['coefficient scan points'] / float(config['chunk size']))
        for index in range(int(chunks)):
            scan = os.path.join(config['outdir'], 'scans', '{}_{}.root'.format(label, index))
            scans.append(scan)
            cmd = ['run', 'combine'] + list(coefficients) + ['-i', str(index), 'config.py']

            makeflowify(['config.py', workspace], scan, cmd)

        outfile = os.path.join(config['outdir'], 'scans', '{}.total.root'.format(label))
        makeflowify(scans, outfile, ['hadd', '-f', outfile] + scans)

    inputs = [os.path.join(config['outdir'], 'scans', '{}.total.root'.format('_'.join(o))) for o in combinations]
    inputs += ['cross_sections.npz', 'config.py']
    for operator in config['coefficients']:
        fluctuations = os.path.join(config['outdir'], 'fluctuations-{}.npy'.format(operator))
        cmd = ['run', 'plot', operator, 'config.py']
        makeflowify(inputs + [fluctuations], [], cmd)

def combine(args, config):
    print args
    mus = np.load(os.path.join(config['outdir'], 'mus.npy'))[()]

    # convergence of the loop expansion requires c < (4 * pi)^2
    # see section 7 in https://arxiv.org/pdf/1205.4231.pdf
    c_max = (4 * np.pi) ** 2
    pmin = -1 * c_max
    pmax = c_max
    label = '_'.join(args.coefficient)
    for p in config['processes']:
        if (mus[label][p](c_max) > config['scale window']):
            pmin = max([(mus[label][p] - config['scale window']).roots().min(), pmin])
            pmax = min([(mus[label][p] - config['scale window']).roots().max(), pmax])

    cmd = [
        'combine', '-M', 'MultiDimFit', ' --saveFitResult', '{}'.format(os.path.join(config['outdir'], 'workspaces', '{}.root'.format(label))),
        '--setParameters', '{}'.format(','.join(['{}=0.0'.format(x) for x in args.coefficient])),
        '--setParameterRanges', '{}'.format(':'.join(['{}={},{}'.format(x, pmin, pmax) for x in args.coefficient]))
    ]
    if config['asimov data']:
        cmd += ['-t', '-1']
    if args.index is not None:
        lowers = np.arange(1, config['coefficient scan points'], config['chunk size'])
        uppers = np.arange(config['chunk size'], config['coefficient scan points'] + config['chunk size'], config['chunk size'])
        first, last = zip(lowers, uppers)[args.index]
        cmd += [
            '--algo=grid',
            '--points={}'.format(config['coefficient scan points']),
            '--firstPoint={}'.format(first),
            '--lastPoint={}'.format(last)
        ]
    else:
        cmd += ['--algo=singles']

    # FIXME: do I still need this?
    # '--autoRange={}'.format('15' if config['asimov data'] else '20'),
    print ' '.join(cmd)
    subprocess.call(' '.join(cmd), shell=True)

    if args.index is not None:
        shutil.move(
            'higgsCombineTest.MultiDimFit.mH120.root',
            os.path.join(config['outdir'], 'scans', '{}_{}.root'.format(label, args.index)))
    else:
        shutil.move(
            'higgsCombineTest.MultiDimFit.mH120.root',
            os.path.join(config['outdir'], 'best-fit-{}.root'.format(label)))
        shutil.move(
            'multidimfit.root',
            os.path.join(config['outdir'], 'fit-result-{}.root'.format(label)))

def parse(args, config):
    import DataFormats.FWLite

    result = CrossSectionScan()

    def get_collection(run, ctype, label):
        handle = DataFormats.FWLite.Handle(ctype)
        try:
            run.getByLabel(label, handle)
        except:
            raise

        return handle.product()

    logging.info('parsing {}'.format(args.file))

    for run in DataFormats.FWLite.Runs(args.file):
        cross_section = get_collection(run, 'LHERunInfoProduct', 'externalLHEProducer::LHE').heprup().XSECUP[0]
        coefficients = get_collection(run, 'vector<string>', 'annotator:wilsonCoefficients:LHE')
        process = str(get_collection(run, 'std::string', 'annotator:process:LHE'))
        point = np.array(get_collection(run, 'vector<double>', 'annotator:point:LHE'))

        result.add(point, cross_section, process, coefficients)

    outfile = os.path.join(config['outdir'], 'cross_sections', os.path.basename(args.file).replace('.root', '.npz'))
    result.dump(outfile)


def concatenate(args, config):
    files = glob.glob(os.path.join(config['outdir'], 'cross_sections', '*.npz'))
    if 'indir' in config:
        files += glob.glob(os.path.join(config['indir'], '*.npz'))

    result = CrossSectionScan(files)
    for coefficients in result.points:
        for process in result.points[coefficients]:
            result.deduplicate(coefficients, process)

    outfile = os.path.join(config['outdir'], 'cross_sections.npz')
    result.dump(outfile)

