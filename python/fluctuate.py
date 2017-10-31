import os
import sys

import numpy as np
import ROOT

from EffectiveTTV.EffectiveTTV.parameters import nlo
from EffectiveTTV.EffectiveTTV.signal_strength import load, load_mus


def fluctuate(args, config):
    ROOT.RooRandom.randomGenerator().SetSeed(0)

    coefficient = args.coefficient
    mus = load_mus(config)

    file = ROOT.TFile.Open(os.path.join(config['outdir'], 'fit-result-{}.root'.format(coefficient)))
    fit = file.Get('fit_mdf')

    data = np.empty(
            (int(args.perturbations) + 1,),
            dtype=[('x_sec_{}'.format(process), 'f8') for process in config['processes']] +
                [('r_{}'.format(process), 'f8') for process in config['processes']] +
                [(coefficient, 'f8')] +
                [(s, 'f8') for s in config['systematics'].keys()]
    )

    pars = fit.floatParsFinal()
    for theta in config['systematics'].keys() + [args.coefficient]:
        data[0][theta] = float(pars.selectByName(theta)[0].getVal())

    def get_cross_sections(pars):
        cross_sections = {}

        for process in config['processes']:
            cross_sections[process] = ROOT.ProcessNormalization('xsec_{}'.format(process), '', nlo[process])
            for systematic, info in config['systematics'].items():
                if process in info['kappa']:
                    if not info['distribution'] == 'lnN':
                        raise NotImplementedError
                    if info['kappa'][process]['+'] == info['kappa'][process]['-']:
                        cross_sections[process].addLogNormal(info['kappa'][process]['+'], pars.selectByName(systematic)[0])
                    else:
                        cross_sections[process].addAsymmLogNormal(
                            info['kappa'][process]['-'],
                            info['kappa'][process]['+'],
                            pars.selectByName(systematic)[0]
                        )

        return cross_sections

    cross_sections = get_cross_sections(pars)
    for process in config['processes']:
        data[0]['x_sec_{}'.format(process)] = cross_sections[process].getVal() * mus[coefficient][process](data[coefficient][0])

    pars = fit.randomizePars()
    cross_sections = get_cross_sections(pars)
    for i in range(1, int(args.perturbations)):
        pars = fit.randomizePars()
        for par in config['systematics'].keys() + [coefficient]:
            data[i][par] = pars.selectByName(par)[0].getVal()
        for process in config['processes']:
            data[i]['x_sec_{}'.format(process)] = cross_sections[process].getVal() * mus[coefficient][process](data[i][coefficient])

    for process in config['processes']:
        data['r_{}'.format(process)] = mus[coefficient][process](data[coefficient])

    np.save(os.path.join(config['outdir'], 'fluctuations-{}.npy'.format(args.coefficient)), data)
