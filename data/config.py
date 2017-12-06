import NPFit.NPFit.plotting as plotting

config = {
    'indirs': ['/hadoop/store/user/$USER/ttV/cross_sections/18/'],
    'outdir': '~/www/.private/ttV/80/',  # Output directory; iterate the version each time you make changes
    'shared-fs': ['/afs', '/hadoop'],  # Declare filesystems the batch system can access-- files will not be copied (faster)
    'coefficients': ['cuW', 'cuB', 'cH', 'tc3G', 'c3G', 'cHu', 'c2G', 'cuG'],
    # 'dimension': 1,
    # 'plots': [
    #     # Wildcards are accepted, for example:
    #     # plotting.FitErrors(['/hadoop/store/user/$USER/ttV/cross_sections/1/1d/final*/*npz'], dimensions=[1]),
    #     plotting.NewPhysicsScaling([ ('ttW', 'x', 'blue'), ('ttZ', '+', '#2fd164'), ('ttH', 'o', '#ff321a')],
    #         match_nll_window=False,
    #         subdir='scaling_free_window_dimensionless',
    #         dimensionless=True
    #     ),
    #     plotting.NLL(),
    #     plotting.TwoProcessCrossSectionSM(
    #         subdir='.',
    #         signals=['ttW', 'ttZ'],
    #         theory_errors={'ttW': (0.1173, 0.1316), 'ttZ': (0.1164, 0.10)},
    #         numpoints=500, chunksize=250, contours=True),
    #     plotting.TwoProcessCrossSectionSMAndNP(
    #         subdir='.',
    #         signals=['ttW', 'ttZ'],
    #         theory_errors={'ttW': (0.1173, 0.1316), 'ttZ': (0.1164, 0.10)})
    # ],
    # uncomment below for scanning two coefficients at a time
    'plots': [
        plotting.FitErrors(dimensions=[2], fitpoints=list(range(6, 500, 1)),
        plotting.NewPhysicsScaling2D(['ttZ', 'ttH', 'ttW'], maxnll=70),
        plotting.NLL2D(scatter=True, maxnll=70),
        plotting.NLL2D(scatter=True, subdir='nll2ddimensionless', maxnll=70, dimensionless=True)
    ],
    'asimov data': False,  # Calculate expected values with MC data only (Asimov dataset), false for real data.
    'cards': {
        '2l': '/afs/cern.ch/user/a/awoodard/public/TOP-17-005/2L',
        '3l': '/afs/cern.ch/user/a/awoodard/public/TOP-17-005/3L',
        '4l': '/afs/cern.ch/user/a/awoodard/public/TOP-17-005/4L.txt'
    },
    'luminosity': 36,
    'scale window': 10,  # maximum scaling of any scaled process at which to set the scan boundaries
    'autorange': {('c2G', 'c3G'): 2},  # set parameter ranges to +/- this many standard deviations for these coefficient groups instead of using the scan ranges-- mostly used as  a fallback when the usual guess does a bad job
    'processes': ['ttH', 'ttZ', 'ttW'],  # processes to scale
    'fluctuations': 10000,
    'header': 'preliminary',
    'interp points': 100,
    'np points': 40000,
    'np chunksize': 300,
    'systematics': {  # below, list any additional (NP-specific, beyond what is in `cards`) systematics to apply
        'PDF_gg': {
            'kappa': {  # https://arxiv.org/pdf/1610.07922.pdf page 160
                'ttZ': {'-': 1.028, '+': 1.028},
                'ttH': {'-': 1.03, '+': 1.03}
            },
            'distribution': 'lnN'
        },
        'PDF_qq': {
            'kappa': {'ttW': {'-': 1.0205, '+': 1.0205}},
            'distribution': 'lnN'
        },
        'Q2_ttH': {
            'kappa': {'ttH': {'-': 1.092, '+': 1.058}},
            'distribution': 'lnN'
        },
        'Q2_ttZ': {
            'kappa': {'ttZ': {'-': 1.113, '+': 1.096}},
            'distribution': 'lnN'
        },
        'Q2_ttW': {
            'kappa': {'ttW': {'-': 1.1155, '+': 1.13}},
            'distribution': 'lnN'
        }
    },
    'label': 'ttV_FTW',  # label to use for batch submission: no need to change this between runs
    'batch type': 'condor',  # batch type for makeflow, can be local, wq, condor, sge, torque, moab, slurm, chirp, amazon, dryrun
}
