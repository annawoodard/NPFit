
import glob
import logging
import itertools
import numpy as np
import os
import shutil
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
    local_config = config.copy()
    local_config['outdir'] = ''
    with open(configfile, 'w') as f:
        yaml.dump(local_config, f)

    data = os.path.join(os.environ['LOCALRT'], 'src', 'EffectiveTTV', 'EffectiveTTV', 'data')
    shutil.copy(os.path.join(data, 'matplotlibrc'), config['outdir'])

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


    inputs = [os.path.join('cross_sections', os.path.basename(f).replace('.root', '.npy')) for f in files] + ['run.yaml']
    outputs = 'cross_sections.npy'
    makeflowify(inputs, outputs, ['run', '--concatenate', 'run.yaml'])

    inputs = ['cross_sections.npy', 'run.yaml']
    makeflowify(inputs, [], ['LOCAL', 'run', '--plot', 'run.yaml'])

    lowers = np.arange(0, config['points'], config['chunk size'])
    uppers = np.arange(config['chunk size'], config['points'] + config['chunk size'], config['chunk size'])

    combinations = [sorted(list(x)) for x in itertools.combinations(config['operators'], config['dimension'])]
    for operators in combinations:
        label = '_'.join(operators)
        workspace = os.path.join('workspaces', '{}.root'.format(label))
        cmd = [
            'mkdir workspaces;',
            'text2workspace.py', config['card'],
            '-P', 'EffectiveTTV.EffectiveTTV.models:eff_op',
            '-m', '125',
            '--PO', 'data={}'.format(config['outdir']),
            ' '.join(['--PO process={}'.format(x) for x in config['processes']]),
            ' '.join(['--PO poi={}'.format(x) for x in operators]),
            '-o', workspace
        ]

        makeflowify('cross_sections.npy', workspace, cmd)

        scans = []
        for index, (first, last) in enumerate(zip(lowers, uppers)):
            cmd = [
                'combine',
                '-M', 'MultiDimFit',
                workspace,
                '-m', '125',
                '--algo=grid',
                '--points={}'.format(config['points']),
                '--setPhysicsModelParameters', ','.join(['{}=0.0'.format(x) for x in operators]),
                '--setPhysicsModelParameterRanges', ':'.join(['{}=-1,1'.format(x) for x in operators]),
                '-n', '_{}_part_{}'.format(label, index),
                '--firstPoint {}'.format(first),
                '--lastPoint {}'.format(last)
            ]

            scan = os.path.join('scans', 'higgsCombine_{}_part_{}.MultiDimFit.mH125.root'.format(label, index))
            scans.append(scan)

            makeflowify(workspace, scan, cmd, rename=True)

        outfile = '{}.total.root'.format(label)
        makeflowify(scans, outfile, ['LOCAL', 'hadd', '-f', outfile] + scans)

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

def plot(args, config):
    from EffectiveTTV.EffectiveTTV.plotting import plot_xsecs, plot_nll

    # plot_xsecs(config)
    plot_nll(config)
