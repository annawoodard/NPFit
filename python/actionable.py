
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


def make(args, config):
    # Makeflow is a bit picky about whitespace
    frag = """\n{out}: {ins}\n\t{cmd}\n"""

    wrap = """#!/bin/sh

    source /cvmfs/cms.cern.ch/cmsset_default.sh
    cd {0}
    cmsenv
    cd -
    exec "$@"
    """.format(os.environ["LOCALRT"])

    wrapfile = os.path.join(config['outdir'], 'w.sh')
    with open(wrapfile, 'w') as f:
        f.write(wrap)
    os.chmod(wrapfile, os.stat(wrapfile).st_mode | stat.S_IEXEC)

    configfile = os.path.join(config['outdir'], 'run.yaml')
    with open(configfile, 'w') as f:
        yaml.dump(config, f)

    data = os.path.join(os.environ['LOCALRT'], 'src', 'EffectiveTTV', 'EffectiveTTV', 'data')
    shutil.copy(os.path.join(data, 'matplotlibrc'), config['outdir'])

    def cardify(name):
        return os.path.join(config['outdir'], '{}.txt'.format(name))

    for name, card in config['cards'].items():
        shutil.copy(card, cardify(name))

    subprocess.call('combineCards.py {} {} > {}'.format(cardify('3l'), cardify('4l'), cardify('ttZ')), shell=True)
    subprocess.call('combineCards.py {} {} > {}'.format(cardify('ttZ'), cardify('2lss'), cardify('ttV')), shell=True)
    
    with open(cardify('ttV'), 'r') as f:
        card = f.read()

    processes = re.compile(r'\nprocess.*')
    for index, process in enumerate(['ttW', 'ttZ']):
        names, numbers = processes.findall(card)
        for column in [i for i, name in enumerate(names.split()) if name == process]:
            number = numbers.split()[column]
            card = card.replace(numbers, numbers.replace(number, '{}'.format(index * -1)))

    jmax = re.search('jmax (\d*)', card).group(0)
    card = card.replace(jmax, 'jmax {}'.format(len(set(names.split()[1:])) - 1))

    with open(cardify('ttV'), 'w') as f:
        f.write(card)

    makefile = os.path.join(config['outdir'], 'Makeflow')
    logging.info('writing Makeflow file to {}'.format(config['outdir']))
    with open(makefile, 'w') as f:
        factory = os.path.join(data, 'factory.json')
        msg = ('# to run, issue the following commands:\n'
        '# cd {}\n'
        '# makeflow -T wq -M ttV_FTW --wrapper ./w.sh --wrapper-input w.sh\n'
        '# nohup work_queue_factory -T condor -M ttV_FTW -C {} >& makeflow_factory.log &\n'
        ).format(config['outdir'], factory)
        
        f.write(msg)
        logging.info(msg.replace('# ', ''))

    def makeflowify(inputs, outputs, cmd='run', rename=False):
        if isinstance(inputs, basestring):
            inputs = [inputs]
        if isinstance(outputs, basestring):
            outputs = [outputs]
        if isinstance(cmd, basestring):
            cmd = shlex.split(cmd)

        outs = ' '.join(outputs)
        ins = ' '.join(inputs)
        if rename:
            outs = ' '.join(["{0}->{1}".format(p, os.path.basename(p)) for p in outputs])

        with open(makefile, 'a') as f:
            s = frag.format(
                    out=outs,
                    ins=ins,
                    cmd=' '.join(cmd))
            f.write(s)

    if 'indir' in config:
        files = glob.glob(os.path.join(config['indir'], '*.root'))
        for f in files:
            outputs = os.path.join('cross_sections', os.path.basename(f).replace('.root', '.npy'))
            makeflowify(['run.yaml', f], outputs, ['run', '--parse', f, 'run.yaml'])

        inputs = [os.path.join('cross_sections', os.path.basename(f).replace('.root', '.npy')) for f in files] + ['run.yaml']
        outputs = 'cross_sections.npy'
        makeflowify(inputs, outputs, ['run', '--concatenate', 'run.yaml'])
    elif 'cross sections' in config:
        shutil.copy(config['cross sections'], os.path.join(config['outdir'], 'cross_sections.npy'))
    else:
        raise RuntimeError('must specify either `indir` or `cross sections`')

    makeflowify('ttZ.txt', 'ttZ.root', 'combine -M MaxLikelihoodFit ttZ.txt; mv higgsCombineTest.MaxLikelihoodFit.mH120.root ttZ.root')
    makeflowify('2lss.txt', 'ttW.root', 'combine -M MaxLikelihoodFit 2lss.txt; mv higgsCombineTest.MaxLikelihoodFit.mH120.root ttW.root')

    lowers = np.arange(1, config['2d points'], config['chunk size'])
    uppers = np.arange(config['chunk size'], config['2d points'] + config['chunk size'], config['chunk size'])

    workspace = os.path.join('workspaces', 'ttW_ttZ_2D.root')
    cmd = [
        'mkdir workspaces;',
        'text2workspace.py', os.path.join(config['outdir'], 'ttV.txt'),
        '-P', 'HiggsAnalysis.CombinedLimit.PhysicsModel:multiSignalModel',
        '--PO', 'map=.*/ttZ:r_ttZ[1,0,4]',
        '--PO', 'map=.*/ttW:r_ttW[1,0,4]',
        '-o', workspace
    ]

    makeflowify([], workspace, cmd)
    
    scans = []
    for index, (first, last) in enumerate(zip(lowers, uppers)):
        cmd = [
            'combine',
            '-M', 'MultiDimFit',
            workspace,
            '--algo=grid',
            '--points={}'.format(config['2d points']),
            '-n', '_ttW_ttZ_2D_part_{}'.format(index),
            '--firstPoint {}'.format(first),
            '--lastPoint {}'.format(last)
        ]

        scan = os.path.join('scans', 'higgsCombine_ttW_ttZ_2D_part_{}.MultiDimFit.mH120.root'.format(index))
        scans.append(scan)

        makeflowify(workspace, scan, cmd, rename=True)

    outfile = 'scans/ttZ_ttW_2D.total.root'
    makeflowify(scans, outfile, ['LOCAL', 'hadd', '-f', outfile] + scans)

    lowers = np.arange(1, config['1d points'], config['chunk size'])
    uppers = np.arange(config['chunk size'], config['1d points'] + config['chunk size'], config['chunk size'])

    combinations = [sorted(list(x)) for x in itertools.combinations(config['operators'], config['dimension'])]
    for operators in combinations:
        label = '_'.join(operators)
        workspace = os.path.join('workspaces', '{}.root'.format(label))
        cmd = [
            'mkdir workspaces;',
            'text2workspace.py', os.path.join(config['outdir'], 'ttV.txt'),
            '-P', 'EffectiveTTV.EffectiveTTV.models:eff_op',
            '--PO', 'data={}'.format(os.path.join(config['outdir'], 'cross_sections.npy')),
            ' '.join(['--PO process={}'.format(x) for x in config['processes']]),
            ' '.join(['--PO poi={}'.format(x) for x in operators]),
            '--PO', 'plots={}'.format(os.path.join(config['outdir'], 'plots')),
            '-o', workspace
        ]

        makeflowify('cross_sections.npy', workspace, cmd)

        scans = []
        for index, (first, last) in enumerate(zip(lowers, uppers)):
            cmd = [
                'combine',
                '-M', 'MultiDimFit',
                workspace,
                '--algo=grid',
                '--points={}'.format(config['1d points']),
                '--setPhysicsModelParameters', ','.join(['{}=0.0'.format(x) for x in operators]),
                '--setPhysicsModelParameterRanges', ':'.join(['{}=-1,1'.format(x) for x in operators]),
                '-n', '_{}_part_{}'.format(label, index),
                '--firstPoint {}'.format(first),
                '--lastPoint {}'.format(last)
            ]

            scan = os.path.join('scans', 'higgsCombine_{}_part_{}.MultiDimFit.mH120.root'.format(label, index))
            scans.append(scan)

            makeflowify(workspace, scan, cmd, rename=True)

        outfile = '{}.total.root'.format(label)
        makeflowify(scans, outfile, ['LOCAL', 'hadd', '-f', outfile] + scans)

    inputs = ['{}.total.root'.format('_'.join(o)) for o in combinations] + ['cross_sections.npy', 'run.yaml'] 
    makeflowify(inputs, [], ['LOCAL', 'sh', os.path.join(data, 'env.sh'), 'run', '--plot', 'run.yaml'])

def parse(args, config):
    import DataFormats.FWLite

    def get_collection(run, ctype, label):
        handle = DataFormats.FWLite.Handle(ctype)
        try:
            run.getByLabel(label, handle)
        except:
            raise

        return handle.product()

    logging.info('parsing {}'.format(args.parse))

    for run in DataFormats.FWLite.Runs(args.parse):
        cross_section = get_collection(run, 'GenRunInfoProduct', 'generator::GEN').crossSection()
        operators = np.array(get_collection(run, 'vector<string>', 'annotator:operators:LHE'))
        process = str(get_collection(run, 'std::string', 'annotator:process:LHE'))
        dtype=[(name, 'f8') for name in operators]
        coefficients = np.array(tuple(get_collection(run, 'vector<double>', 'annotator:wilsonCoefficients:LHE')), dtype=dtype)

        row = np.array((coefficients, cross_section), dtype=[('coefficients', coefficients.dtype, coefficients.shape), ('cross section', 'f8')])

        try:
            cross_sections = np.vstack([cross_sections, row])
        except UnboundLocalError:
            cross_sections = row

    outfile = os.path.join(config['outdir'], 'cross_sections', os.path.basename(args.parse).replace('.root', '.npy'))
    np.save(outfile, {process: cross_sections})


def concatenate(args, config):
    files = glob.glob(os.path.join(config['outdir'], 'cross_sections', '*.npy'))
    res = {}
    for f in files:
        info = np.load(f)[()]
        for process, cross_sections in info.items():
            try:
                res[process] = np.vstack([res[process], cross_sections])
            except KeyError:
                res[process] = cross_sections

    outfile = os.path.join(config['outdir'], 'cross_sections.npy')
    np.save(outfile, res)
