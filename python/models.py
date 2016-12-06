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
            if option == 'process': # processes which will be scaled
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
                    
                    c1 = ROOT.TCanvas()
                    proj = quadratic.plotOn(x.frame())
                    graph = ROOT.TGraph(len(xi), xi, yi)
                    graph.SetMarkerStyle(20)
                    graph.SetMarkerSize(1)
                    graph.SetMarkerColor(6)
                    graph.GetXaxis().SetRangeUser(-1, 1)
                    graph.SetTitle("")
                    graph.GetXaxis().SetTitle(poi)
                    graph.GetYaxis().SetTitle("#frac{#sigma_{NP+SM}}{#sigma_{SM}}")
                    graph.Draw()
                    proj.Draw("same")
                    ROOT.gPad.BuildLegend()
                    path = os.path.join(self.plots, 'cross_sections', process, '{}.pdf'.format(poi))
                    c1.SaveAs(path)

                    del c1

            self.modelBuilder.factory_('sum::x_sec_{0}({1})'.format(process, ', '.join(functions)))

    def doParametersOfInterest(self):
        for poi in self.pois:
            self.modelBuilder.doVar('{0}[0, -1, 1]'.format(poi))

        self.modelBuilder.doSet('POI', ','.join(self.pois))

        self.setup()

    def getYieldScale(self, bin, process):
        card_names = {
            'TTH': 'ttH',
            'TTZ': 'ttZ',
            'TTW': 'ttW'
        }

        if process in card_names:
            process = card_names[process]
        if process not in self.processes:
        # if not self.DC.isSignal[process]:
            return 1
        else:
            name = 'x_sec_{0}'.format(process)

            return name


class TTbarWTTbarZSignalModel(PhysicsModel):
    def doParametersOfInterest(self):
        """Create POI out of signal strength"""
        self.modelBuilder.doVar("r_ttW[0.5,0,2]")
        self.modelBuilder.doVar("r_ttZ[0.5,0,2]")
        self.modelBuilder.doSet("POI",'r_ttW,r_ttZ')

    def getYieldScale(self, bin, process):
        if process == 'TTW':
            return 'r_ttW'
        elif process == 'TTZ':
            return 'r_ttZ'
        else:
            return 1

class CvCaZSimpleModel(PhysicsModel):
    def doParametersOfInterest(self):
        self.modelBuilder.doVar("cV[0.244,-1.3,1.3]")
        self.modelBuilder.doVar("cA[-0.601,-1,1]")
        self.modelBuilder.doSet("POI","cV,cA")
    def getYieldScale(self, bin, process):
        if process == 'TTZ':
            expr = 'expr::r_ttZ("sqrt(0.5 * (@0 / 0.244)^2 + 0.5 * (@1 / -0.0601)^2)", cV, cA)'
            self.modelBuilder.factory_(expr)
            self.modelBuilder.out.Print()
            return 'r_ttZ'
        else:
            return 1

class CvCaModel(PhysicsModel):
    def doParametersOfInterest(self):
        self.modelBuilder.doVar("cV[0.244,-3,3]")
        self.modelBuilder.doVar("cA[-0.601,-3,3]")
        self.modelBuilder.doSet("POI","cV,cA")
    def getYieldScale(self, bin, process):
        if process == 'TTZ':
            expr = 'expr::r_ttZ("(1 / 206) * (74.61 + @0 * 0.504 + @0^2 * 189.4 - @1 * 16.265 + @1^2 * 359.7)", cV, cA)'
            self.modelBuilder.factory_(expr)
            self.modelBuilder.out.Print()
            return 'r_ttZ'
        else:
            return 1

ttW_ttZ_signal_model = TTbarWTTbarZSignalModel()
eff_op = EffectiveOperatorModel()
cVcAZ = CvCaZSimpleModel()
cVcA = CvCaModel()
