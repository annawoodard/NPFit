import os
import re

import numpy as np
from numpy.polynomial import Polynomial
import ROOT
import yaml

from HiggsAnalysis.CombinedLimit.PhysicsModel import PhysicsModel
from HiggsAnalysis.CombinedLimit.SMHiggsBuilder import SMHiggsBuilder

from EffectiveTTV.EffectiveTTV.signal_strength import load, load_mus
from EffectiveTTV.EffectiveTTV.parameters import process_groups

class EffectiveOperatorModel(PhysicsModel):

    def setPhysicsOptions(self, options):
        # FIXME use argparse on options
        self.pois = []
        self.processes = []
        for option, value in [x.split('=') for x in options]:
            if option == 'poi':
                self.pois.append(value)
            if option == 'process':  # processes which will be scaled
                self.processes.append(value)
            if option == 'config':
                with open(value) as f:
                    self.config = yaml.load(f)

    def setup(self):
        mus = load_mus(self.config)
        coefficients, cross_sections = load(self.config)
        for group in self.processes:  # FIXME: naming
            self.modelBuilder.out.var(group)

            functions = []
            for process in process_groups[group]:
                self.modelBuilder.out.var(process)
                for poi in self.pois:
                    x = self.modelBuilder.out.var(poi)
                    name = 'r_{0}_{1}'.format(process, poi)
                    if not self.modelBuilder.out.function(name):
                        functions += [name]
                        template = "expr::{name}('{a0} + ({a1} * {poi}) + ({a2} * {poi} * {poi})', {poi})"
                        a0, a1, a2 = mus[poi][process]
                        quadratic = self.modelBuilder.factory_(template.format(name=name, a0=a0, a1=a1, a2=a2, poi=poi))
                        self.modelBuilder.out._import(quadratic)

            self.modelBuilder.factory_('sum::r_{0}({1})'.format(group, ', '.join(functions)))

    def quadratic(self, x, xi, yi):
        fit = Polynomial.fit(xi, yi, 2)

        a0 = ROOT.RooRealVar("a0", "a0", fit.coef[0])
        a1 = ROOT.RooRealVar("a1", "a1", fit.coef[1])
        a2 = ROOT.RooRealVar("a2", "a2", fit.coef[2])

        p = ROOT.RooPolynomial("p", "p", x, ROOT.RooArgList(a0, a1, a2))

        return p

    def doParametersOfInterest(self):
        # FIXME change the range here and try (4pi)^2
        # it looks like utils::setModelParameterRanges isn't working
        for poi in self.pois:
            self.modelBuilder.doVar('{0}[0, -5, 5]'.format(poi))

        self.modelBuilder.doSet('POI', ','.join(self.pois))

        self.setup()

    def getYieldScale(self, bin, process):
        if process not in self.processes:
            return 1
        else:
            name = 'r_{0}'.format(process)

            return name


eff_op = EffectiveOperatorModel()
