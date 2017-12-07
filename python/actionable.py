import glob
import logging
import numpy as np
import os
import shlex
import shutil
import subprocess

from NPFitProduction.NPFitProduction.cross_sections import CrossSectionScan, get_perimeter

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
    if 'shared-fs' in config:
        for directory in config['shared-fs']:
            if not directory.startswith('/'):
                raise Exception('absolute path required for shared filesystems: {}'.format(directory))
            shared_filesystem += ["--shared-fs '/{}'".format(directory.split('/')[1])]

    info = """
    # to run, issue the following commands:
    cd {outdir}
    nohup work_queue_factory -T {batch_type} -M {label} -C {factory} >& factory.log &

    # then keep running this command until makeflow no longer submits jobs (may take a few tries):
    makeflow -T wq -M {label} {shared}

    # to reproduce the code:
    cd {code_dir}
    git checkout {head}
    """.format(
        batch_type=config['batch type'],
        outdir=config['outdir'],
        label=config['label'],
        factory=os.path.join(os.environ['LOCALRT'], 'src', 'NPFit', 'NPFit', 'data', 'factory.json'),
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
    if args.files is not None:
        files = sum([glob.glob(x) for x in args.files], [])
    else:
        files = glob.glob(os.path.join(config['outdir'], 'cross_sections', '*.npz'))
        if 'indirs' in config:
            for indir in config['indirs']:
                for root, _, filenames in os.walk(indir):
                    files += [os.path.join(root, fn) for fn in filenames if fn.endswith('.npz')]

    result = CrossSectionScan(files)
    for coefficients in result.points:
        for process in result.points[coefficients]:
            try:
                result.deduplicate(coefficients, process)
            except (RuntimeError, KeyError) as e:
                print(e)
                result.prune(process, coefficients)
    result.fit()

    outfile = os.path.join(config['outdir'], args.output)
    result.dump(outfile)


def combine(args, config):
    label = '_'.join(args.coefficients)

    scan = CrossSectionScan([os.path.join(config['outdir'], 'cross_sections.npz')])
    mins = np.amin(scan.points[tuple(args.coefficients)][config['processes'][-1]], axis=0)
    maxes = np.amax(scan.points[tuple(args.coefficients)][config['processes'][-1]], axis=0)

    def call_combine(postfix):
        cmd = [
            'combine', '-M', 'MultiDimFit', '{}'.format(os.path.join(config['outdir'], 'workspaces', '{}.root'.format(label))),
            '--setParameters', '{}'.format(','.join(['{}=0.0'.format(x) for x in args.coefficients])),
        ]
        if tuple(args.coefficients) in config['autorange']:
            cmd += ['--autoRange={}'.format(config['autorange'][tuple(args.coefficients)])]
        else:
            cmd += [
                '--setParameterRanges', ':'.join(['{c}={low},{high}'.format(c=c, low=low, high=high) for c, low, high in zip(args.coefficients, mins, maxes)]),
                '--autoBoundsPOIs=*', '--autoMaxPOIs=*'# , '--verbose=1'
            ]
        if config['asimov data']:
            cmd += ['-t', '-1']

        cmd += postfix

        print 'calling ', ' '.join(cmd)
        subprocess.call(' '.join(cmd), shell=True)

    if args.index is None:
        if len(args.coefficients) == 1:
            call_combine(['--saveFitResult', '--algo=singles'])
            shutil.move(
                'multidimfit.root',
                os.path.join(config['outdir'], 'fit-result-{}.root'.format(label)))
        else:
            call_combine(['--algo=cross'])
            shutil.move(
                'higgsCombineTest.MultiDimFit.mH120.root',
                os.path.join(config['outdir'], 'best-fit-{}.root'.format(label)))
    else:
        lowers = np.arange(1, config['np points'], config['np chunksize'])
        uppers = np.arange(config['np chunksize'], config['np points'] + config['np chunksize'], config['np chunksize'])
        first, last = zip(lowers, uppers)[args.index]
        postfix = [
            '--algo=grid',
            '--points={}'.format(config['np points']),
            '--firstPoint={}'.format(first),
            '--lastPoint={}'.format(last)
        ]
        call_combine(postfix)
        shutil.move(
            'higgsCombineTest.MultiDimFit.mH120.root',
            os.path.join(config['outdir'], 'scans', '{}_{}.root'.format(label, args.index)))
