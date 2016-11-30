from array import array
from HiggsAnalysis.CombinedLimit.PhysicsModel import PhysicsModel
from HiggsAnalysis.CombinedLimit.SMHiggsBuilder import SMHiggsBuilder
import numpy as np
import os
import re
import ROOT


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

    def setup(self):
        fn = os.path.join(self.data, 'cross_sections.npy')
        info = np.load(fn)[()]
        for process in self.processes:
            self.modelBuilder.out.var(process)
            coefficients = info[process]['coefficients']
            cross_section = info[process]['cross section']
            sm_coefficients = np.array([tuple([0.0] * len(coefficients.dtype))], dtype=coefficients.dtype)
            sm_cross_section = np.mean(cross_section[coefficients == sm_coefficients])

            functions = []
            for poi in self.pois:
                x_var = self.modelBuilder.out.var(poi)
                name = 'x_sec_{0}_{1}'.format(process, poi)
                if not self.modelBuilder.out.function(name):
                    functions += [name]
                    x = coefficients[poi][coefficients[poi] != 0]
                    y = cross_section[coefficients[poi] != 0] / sm_cross_section
                    spline = ROOT.RooSpline1D(name, poi, x_var, len(x), x, y)
                    self.modelBuilder.out._import(spline)

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
            self.modelBuilder.factory_('expr::{}("@0", {})'.format(name, name))
            self.modelBuilder.out.Print()

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

class TwoSignalModel(PhysicsModel):
    def setPhysicsOptions(self, options):
        self.poi_map = {'r_ttH': 'TTH', 'r_ttW': 'TTW', 'r_ttZ': 'TTZ'}
        self.pois = []
        for option, value in [x.split('=') for x in options]:
            if option == 'signals':
                self.pois.extend(value.split(','))

    def doParametersOfInterest(self):
        """Create POI out of signal strength"""
        for poi in self.pois:
            self.modelBuilder.doVar("%s[0.5,0,2]" % poi)
        self.modelBuilder.doSet("POI",','.join(self.pois))

    def getYieldScale(self, bin, process):
        for poi in self.pois:
            if process == self.poi_map[poi]:
                return poi
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
two_signal_model = TwoSignalModel()
