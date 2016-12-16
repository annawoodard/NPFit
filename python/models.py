import os
import re

import numpy as np
from numpy.polynomial import Polynomial
import ROOT

from HiggsAnalysis.CombinedLimit.PhysicsModel import PhysicsModel
from HiggsAnalysis.CombinedLimit.SMHiggsBuilder import SMHiggsBuilder


class EffectiveOperatorModel(PhysicsModel):

    def setPhysicsOptions(self, options):
        self.pois = []
        self.processes = []
        for option, value in [x.split('=') for x in options]:
            if option == 'poi':
                self.pois.append(value)
            if option == 'process':  # processes which will be scaled
                self.processes.append(value)
            if option == 'data':
                self.data = value
            if option == 'plots':
                self.plots = value

    def setup(self):
        info = np.load(self.data)[()]
        for process in self.processes:
            self.modelBuilder.out.var(process)
            coefficients = info[process]['coefficients']
            cross_section = info[process]['cross section']
            sm_coefficients = np.array([tuple([0.0] * len(coefficients.dtype))], dtype=coefficients.dtype)
            sm_cross_section = np.mean(cross_section[coefficients == sm_coefficients])

            functions = []
            for poi in self.pois:
                x = self.modelBuilder.out.var(poi)
                name = 'x_sec_{0}_{1}'.format(process, poi)
                if not self.modelBuilder.out.function(name):
                    functions += [name]
                    xi = coefficients[poi][coefficients[poi] != 0]
                    yi = cross_section[coefficients[poi] != 0] / sm_cross_section
                    fit = Polynomial.fit(xi, yi, 2)

                    template = "expr::{name}('{a0} + ({a1} * {poi}) + ({a2} * {poi} * {poi})', {poi})"
                    quadratic = self.modelBuilder.factory_(template.format(name=name, a0=fit.coef[0], a1=fit.coef[1], a2=fit.coef[2], poi=poi))
                    self.modelBuilder.out._import(quadratic)

            self.modelBuilder.factory_('sum::x_sec_{0}({1})'.format(process, ', '.join(functions)))

    def doParametersOfInterest(self):
        for poi in self.pois:
            self.modelBuilder.doVar('{0}[0, -1, 1]'.format(poi))

        self.modelBuilder.doSet('POI', ','.join(self.pois))

        self.setup()

    def getYieldScale(self, bin, process):
        if process not in self.processes:
            return 1
        else:
            name = 'x_sec_{0}'.format(process)

            return name


eff_op = EffectiveOperatorModel()
