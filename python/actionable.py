
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
    source /afs/crc.nd.edu/user/a/awoodard/.profile
    cd -
    exec "$@"
    """.format(os.environ["LOCALRT"])

    wrapfile = os.path.join(config['outdir'], 'w.sh')
    with open(wrapfile, 'w') as f:
        f.write(wrap)
    os.chmod(wrapfile, os.stat(wrapfile).st_mode | stat.S_IEXEC)
    shutil.copy(args.config, outdir)

    makefile = os.path.join(config['outdir'], 'Makeflow')
    logging.info('writing Makeflow file to {}'.format(config['outdir']))
    with open(makefile, 'w') as f:
        factory = os.path.join(os.environ['LOCALRT'], 'src', 'EffectiveTTV', 'EffectiveTTV', 'data', 'factory.json')
        f.write("# to run, issue the following commands:\n")
        f.write("# makeflow -T wq -M ttV_FTW --wrapper ./w.sh --wrapper-input w.sh\n")
        f.write("# nohup work_queue_factory -T condor -M ttV_FTW -C {} >& makeflow_factory.log &\n".format(factory))

    def makeflowify(inputs, outputs, cmd='run', rename=True):
        if isinstance(outputs, basestring):
            outputs = [outputs]
        if isinstance(cmd, basestring):
            cmd = [cmd]

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

    files = glob.glob(os.path.join(config['indir'], '*.root'))
    for f in files:
        inputs = [os.path.basename(args.config)]
        outputs = os.path.join('cross_sections', os.path.basename(f).replace('.root', '.npy'))
        makeflowify(inputs, outputs, ['run', '-p', f, os.path.basename(args.config)], False)

    inputs = [os.path.join('cross_sections', os.path.basename(f).replace('.root', '.npy')) for f in files] + [os.path.basename(args.config)]
    outputs = 'cross_sections.npy'
    makeflowify(inputs, outputs, ['run', '-c', os.path.basename(args.config)], False)

    for path in ('workspaces', 'scans'):
        if not os.path.exists(os.path.join(config['outdir'], path)):
            os.makedirs(os.path.join(config['outdir'], path))

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

        makeflowify(['cross_sections.npy'], [workspace], cmd, False)

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

            makeflowify([workspace], [scan], cmd)

        outfile = '{}.total.root'.format(label)
        makeflowify(scans, [outfile], ['hadd', '-f', outfile] + scans)

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

    outdir = os.path.join(config["outdir"], 'cross_sections')
    if not os.path.exists(outdir):
        os.makedirs(outdir)

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
    from EffectiveTTV.EffectiveTTV.plotting import NumPyPlotter
    plotter = NumPyPlotter(config)

    data = {}
    labels = {}

    fn = os.path.join(config['outdir'], 'cross_sections.npy')
    for process, info in np.load(fn)[()].items():
        coefficients = info['coefficients']
        cross_section = info['cross section']
        sm_coefficients = np.array([tuple([0.0] * len(coefficients.dtype))], dtype=coefficients.dtype)
        sm_cross_section = np.mean(cross_section[coefficients == sm_coefficients])

        for operator in coefficients.dtype.names:
            x = coefficients[operator][coefficients[operator] != 0]
            y = cross_section[coefficients[operator] != 0] / sm_cross_section
            try:
                data[operator].append((x, y))
                labels[operator].append(process)
            except KeyError:
                data[operator] = [(x, y)]
                labels[operator] = [process]

    for operator in data.keys():
        plotter.plot(data[operator], operator, '$\sigma_{NP+SM} / \sigma_{SM}$', '', 'ratios_{}'.format(operator), series_labels=labels[operator])

