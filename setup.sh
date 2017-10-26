#!/bin/sh

cat <<EOF
=========================================================================
this script creates a working directory for the EffectiveTTV analysis
output is in setup.log
=========================================================================
EOF

(
set -e
set -o xtrace

if ! type "$makeflow" > /dev/null; then
   echo "cctools is required; for instructions please visit http://ccl.cse.nd.edu/software/downloadfiles.php"
fi

export SCRAM_ARCH=slc6_amd64_gcc530
cmsrel CMSSW_8_1_0
cd CMSSW_8_1_0/src
cmsenv
git clone https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit.git HiggsAnalysis/CombinedLimit
cd HiggsAnalysis/CombinedLimit
git fetch origin
git checkout v7.0.3
cd $CMSSW_BASE/src
git clone https://github.com/cms-analysis/CombineHarvester.git CombineHarvester
git clone git@github.com:annawoodard/EffectiveTTV.git EffectiveTTV/EffectiveTTV

scramv1 b clean
scramv1 b -j 16

pip install --user tabulate
pip install --user tempdir

) > setup.log

cat <<EOF
=========================================================================
output is in setup.log
=========================================================================
EOF
