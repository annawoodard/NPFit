
import glob
import logging
import itertools
import numpy as np
import os
import re
import shutil
import stat


def make(args, config):
    outdir = config['outdir']
    indir = config['indir']
    config = os.path.basename(args.config)

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

    wrapfile = os.path.join(outdir, 'w.sh')
    with open(wrapfile, 'w') as f:
        f.write(wrap)
    os.chmod(wrapfile, os.stat(wrapfile).st_mode | stat.S_IEXEC)
    shutil.copy(args.config, outdir)

    makefile = os.path.join(outdir, 'Makeflow')
    logging.info('writing Makeflow file to {}'.format(outdir))
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
            ins = ' '.join(["{0}->{1}".format(p, os.path.basename(p)) for p in inputs])

        with open(makefile, 'a') as f:
            s = frag.format(
                    out=outs,
                    ins=ins,
                    cmd=' '.join(cmd))
            f.write(s)

    files = glob.glob(os.path.join(indir, '*.root'))
    for f in files:
        inputs = [config]
        outputs = os.path.join('cross_sections', os.path.basename(f).replace('.root', '.npy'))
        makeflowify(inputs, outputs, ['run', '-p', f, config], False)

    inputs = [os.path.join('cross_sections', os.path.basename(f).replace('.root', '.npy')) for f in files] + [config]
    outputs = 'cross_sections.npy'
    makeflowify(inputs, outputs, ['run', '-c', config], False)


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
        coefficients = np.array(get_collection(run, 'vector<double>', 'annotator:wilsonCoefficients:LHE'))
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

