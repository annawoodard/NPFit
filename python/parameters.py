import numpy as np

nlo = {
    'ttW': 601.,
    'ttZ': 839.,
    'ttH': 496.
}

# https://cds.cern.ch/record/2150771/files/LHCHXSWG-DRAFT-INT-2016-008.pdf
kappa = {
    'Q2_ttZ': {'-': 1.113, '+': 1.096},
    'Q2_ttW': {'-': 1.1155, '+': 1.13},
    'Q2_ttH': {'-': 1.092, '+': 1.058},
    'PDF_gg': {'ttZ': {'-': 1.028, '+': 1.028}, 'ttH': {'-': 1.03, '+': 1.03}},
    'PDF_qq': {'-': 1.0205, '+': 1.0205}
}

operators = ['c2W', 'c3G', 'c3W', 'cA', 'cB', 'cG', 'cHB', 'cHQ', 'cHW',
             'cHd', 'cHu', 'cHud', 'cT', 'cWW', 'cpHQ', 'cu', 'cuB',
             'cuG', 'cuW', 'tc3G', 'tc3W', 'tcG', 'tcHW']

process_groups = {
    'ttX': ['tZq'],
    'ttother': ['tZq'],
    'rare': ['WZZ', 'ZZZ', 'WWW', 'WWZ'],
    'charge': ['tt'],
    'fake': ['tt', 'DY'],
    'Fake': ['tt', 'DY'],
    'WZ': ['WZ'],
    'ZZ': ['ZZ'],
    'ttH': ['ttH'],
    'ttZ': ['ttZ'],
    'ttW': ['ttW'],
}

names = {
    'Fake': 'fake',
    'WZ': 'WZ',
    'ZZ': 'ZZ',
    'charge': 'charge',
    'fake': 'fake',
    'rare': 'rare',
    'ttH': 'ttH',
    'ttW': 'ttW',
    'ttX': 'ttX',
    'ttZ': 'ttZ',
    'ttother': 'ttX'
}

label = {
    'sigma ttW': r'$\sigma_{\mathrm{t\bar{t}W}}$ $\mathrm{[fb]}$',
    'sigma ttZ': r'$\sigma_{\mathrm{t\bar{t}Z}}$ $\mathrm{[fb]}$',
    'ttW': r'$\mathrm{t\bar{t}W}$',
    'ttZ': r'$\mathrm{t\bar{t}Z}$',
    'ttH': r'$\mathrm{t\bar{t}H}$',
    'tt': r'$\mathrm{t\bar{t}}$',
    'c2B': r'$\bar{c}_{2B}$',
    'c2G': r'$\bar{c}_{2G}$',
    'c2W': r'$\bar{c}_{2W}$',
    'c3G': r'$\bar{c}_{3G}$',
    'c3W': r'$\bar{c}_{3W}$',
    'c6': r'$\bar{c}_{6}$',
    'cA': r'$\bar{c}_{A}$',
    'cB': r'$\bar{c}_{B}$',
    'cG': r'$\bar{c}_{G}$',
    'cH': r'$\bar{c}_{H}$',
    'cHB': r'$\bar{c}_{HB}$',
    'cHL': r'$\bar{c}_{HL}$',
    'cHQ': r'$\bar{c}_{HQ}$',
    'cHW': r'$\bar{c}_{HW}$',
    'cHd': r'$\bar{c}_{Hd}$',
    'cHe': r'$\bar{c}_{He}$',
    'cHu': r'$\bar{c}_{Hu}$',
    'cHud': r'$\bar{c}_{Hud}$',
    'cT': r'$\bar{c}_{T}$',
    'cWW': r'$\bar{c}_{WW}$',
    'cd': r'$\bar{c}_{d}$',
    'cdB': r'$\bar{c}_{dB}$',
    'cdG': r'$\bar{c}_{dG}$',
    'cdW': r'$\bar{c}_{dW}$',
    'cl': r'$\bar{c}_{l}$',
    'clB': r'$\bar{c}_{lB}$',
    'clW': r'$\bar{c}_{lW}$',
    'cpHL': r"$\bar{c}'_{HL}$",
    'cpHQ': r"$\bar{c}'_{HQ}$",
    'cu': r'$\bar{c}_{u}$',
    'cuB': r'$\bar{c}_{uB}$',
    'cuG': r'$\bar{c}_{uG}$',
    'cuW': r'$\bar{c}_{uW}$',
    'tc3G': r'$\widetilde{c}_{3G}$',
    'tc3W': r'$\widetilde{c}_{3W}$',
    'tcA': r'$\widetilde{c}_{A}$',
    'tcG': r'$\widetilde{c}_{G}$',
    'tcHB': r'$\widetilde{c}_{HB}$',
    'tcHW': r'$\widetilde{c}_{HW}$',
}

# see http://feynrules.irmp.ucl.ac.be/attachment/wiki/HEL/HEL.fr
vev = 0.246 # TeV
mass_w = 0.080385 # TeV
ymt = 0.172 # TeV
yu = np.sqrt(ymt) / vev # up-type Yukawa couplings (the first two generations have been removed from the model)
ymb = 0.0047 # TeV
yb = np.sqrt(ymb) / vev # up-type Yukawa couplings (the first two generations have been removed from the model)
aeW = 1. / 127.9 # electroweak coupling contant
ee = np.sqrt(4 * np.pi * aeW) # electric coupling constant
sw = ee * vev / (2 * mass_w) # sine of the Weinberg angle
gw = ee / sw # weak coupling constant at the Z pole
g1 = ee / (1 - sw * sw) # U(1)Y coupling constant at the Z pole

# see https://arxiv.org/pdf/1310.5150v2.pdf equations 2.4 - 2.14
conversion = {
    'cu': yu / (vev * vev),
    'cHu': 1 / (2 * vev * vev),
    'cuB': (2 * g1 * yu) / (mass_w * mass_w),
    'cuW': (gw * yu) / (mass_w * mass_w)
}

# see http://feynrules.irmp.ucl.ac.be/attachment/wiki/HEL/HEL.fr
vev = 0.246 # TeV
mass_w = 0.080385 # TeV
mass_higgs = 0.125 # TeV
higgs_quartic_coupling = np.power(mass_higgs, 2) / (2 * np.power(vev, 2))  # see http://sns.ias.edu/~pgl/SMB/Higgs_update.pdf equation 7.196
ymt = 0.172 # TeV
yu = np.sqrt(ymt) / vev # up-type Yukawa couplings (the first two generations have been removed from the model)
ymb = 0.0047 # TeV
yb = np.sqrt(ymb) / vev # up-type Yukawa couplings (the first two generations have been removed from the model)
aeW = 1. / 127.9 # electroweak coupling contant
ee = np.sqrt(4 * np.pi * aeW) # electric coupling constant
sw = ee * vev / (2 * mass_w) # sine of the Weinberg angle
gw = ee / sw # weak coupling constant at the Z pole
g1 = ee / (1 - sw * sw) # U(1)Y coupling constant at the Z pole
aS = 0.1184
gs = np.sqrt(4 * np.pi * aS) # strong coupling constant at the Z pole


# see https://arxiv.org/pdf/1310.5150v2.pdf equations 2.4 - 2.14
cutoff = {
    'cH': np.sqrt(2) * vev,
    'cT': np.sqrt(2) * vev,
    'c6': vev / np.sqrt(higgs_quartic_coupling),
    'cu': vev / np.sqrt(yu),
    'cWW': mass_w / np.sqrt(gw),
    'cB': np.sqrt(2) * mass_w / np.sqrt(g1),
    'cHW': mass_w / np.sqrt(gw),
    'cHB': mass_w / np.sqrt(g1),
    'cA': mass_w / g1,
    'cG': mass_w,
    'cHQ': vev,
    'cpHQ': vev / 2,
    'cHu': np.sqrt(2) * vev,
    'cHd': np.sqrt(2) * vev,
    'cHud': vev,
    'cHe': np.sqrt(2) * vev,
    'cHL': vev,
    'cuB': mass_w / np.sqrt(g1 * yu),
    'cuW': mass_w / np.sqrt(gw * yu),
    'cuG': mass_w / np.sqrt(gs * yu),
    'c3W': mass_w / np.sqrt(gw * gw * gw),
    'c3G': mass_w / np.sqrt(gw * gw * gw),
    'c2W': mass_w,
    'c2B': mass_w,
    'c2G': mass_w,
    'tcHW': mass_w / np.sqrt(gw),
    'tcHB': np.sqrt(2) * mass_w / np.sqrt(g1),
    'tcA': np.sqrt(2) * mass_w / g1,
    'tc3W': np.sqrt(2) * mass_w / np.sqrt(gw * gw * gw),
    'tc3G': np.sqrt(2) * mass_w / np.sqrt(gs * gs * gs),
    'cpHL': vev / 2,
    'tcG': np.sqrt(2) * mass_w / gs
}

