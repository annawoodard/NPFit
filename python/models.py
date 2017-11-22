from HiggsAnalysis.CombinedLimit.PhysicsModel import PhysicsModel

from NPFitProduction.NPFitProduction.cross_sections import CrossSectionScan
from NPFitProduction.NPFitProduction.utils import sorted_combos


class EFTScaling(PhysicsModel):
    """Apply process scaling due to EFT operators.

    This class takes a `CrossSectionScan`,  performs a fit to describe how processes are
    scaled as a function of an EFT operator's Wilson coefficient and adds it to
    the workspace.

    """

    def setPhysicsOptions(self, options):
        self.pois = []
        self.processes = []
        for option, value in [x.split('=') for x in options]:
            if option == 'poi':
                self.pois.append(value)
            if option == 'process':  # processes which will be scaled
                self.processes.append(value)
            if option == 'scan':
                self.scan = CrossSectionScan(value)

    def setup(self):
        dim = len(self.pois)
        for process in self.processes:
            if process not in self.scan.fit_constants[self.pois]:
                raise RuntimeError('no fit provided for process {}'.format(process))
            self.modelBuilder.out.var(process)
            name = 'r_{0}'.format(process)

            pairs = sorted_combos(range(0, dim), 2)

            constant = ['1.0']
            linear = self.pois
            quad = ['{p} * {p}'.format(p=p) for p in self.pois]
            mixed = ['{p0} * {p1}'.format(p0=self.pois[p0], p1=self.pois[p1]) for p0, p1 in pairs]
            info = zip(self.scan.fit_constants[tuple(self.pois)][process], constant + linear + quad + mixed)
            terms = ['({s} * {c})'.format(s=s, c=c) for s, c in info]
            template = 'expr::{name}("{terms}", {pois})'
            print 'building ', template.format(name=name, terms=' + '.join(terms), pois=', '.join(self.pois))

            scale = self.modelBuilder.factory_(template.format(name=name, terms=' + '.join(terms), pois=', '.join(self.pois)))
            self.modelBuilder.out._import(scale)

    def doParametersOfInterest(self):
        for poi in self.pois:
            # user should call combine with `--setPhysicsModelParameterRanges` set to sensible ranges
            self.modelBuilder.doVar('{0}[0, -inf, inf]'.format(poi))

        self.modelBuilder.doSet('POI', ','.join(self.pois))

        self.setup()

    def getYieldScale(self, bin, process):
        if process not in self.processes:
            return 1
        else:
            name = 'r_{0}'.format(process)

            return name


eft = EFTScaling()
