config = {
    'outdir': '~/www/ttV/1',  # Output directory; iterate the version each time you make changes.
    'outdir shared': True,  # Can the batch system machines access this directory? If so, files will not be copied (this is faster)
    'indir': '/hadoop/store/user/$USER/ttV/cross_sections/1/final_pass*/',
    'indir shared': True,  # Can the batch system machines access this directory? If so, files will not be copied (this is faster)
    'asimov data': False,  # Calculate expected values with MC data only (Asimov dataset), false for real data.
    'dimension': 1,
    'cards': {
        '2l': '/afs/cern.ch/user/a/awoodard/Public/TOP-17-005/2L',
        '3l': '/afs/cern.ch/user/a/awoodard/Public/TOP-17-005/3L',
        '4l': '/afs/cern.ch/user/a/awoodard/Public/TOP-17-005/4L.txt'
    },
    'coefficient scan points': 300,
    '2d points': 500,
    'chunk size': 500,
    'luminosity': 36,
    'scale window': 5,  # maximum scaling of any scaled process at which to set the scan boundaries
    'processes': ['ttH', 'ttZ', 'ttW'], # processes to scale
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
    # 'coefficients': ['c2B', 'c2G', 'c2W', 'c3G', 'c3W', 'c6', 'cA', 'cB', 'cG', 'cH', 'cHB', 'cHL', 'cHQ', 'cHW', 'cHd', 'cHe', 'cHu', 'cHud', 'cT', 'cWW', 'cd', 'cdB', 'cdG', 'cdW', 'cl', 'clB', 'clW', 'cpHL', 'cpHQ', 'cu', 'cuB', 'cuG', 'cuW', 'tc3G', 'tc3W', 'tcA', 'tcG', 'tcHB', 'tcHW']
    'coefficients': ['cuW', 'cuB', 'cH', 'tc3G', 'c3G', 'cHu', 'c2G', 'cuG'],
    'label': 'ttV_FTW',  # label to use for batch submission: no need to change this between runs
    'batch type': 'condor',  # batch type for makeflow, can be local, wq, condor, sge, torque, moab, slurm, chirp, amazon, dryrun
    'make 2d sm contours': False  # run an additional scan to calculate the 2D SM contours
}
