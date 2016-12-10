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

export SCRAM_ARCH=slc6_amd64_gcc491
scramv1 project CMSSW_7_4_7
cd CMSSW_7_4_7/src
set +o xtrace
eval $(scramv1 runtime -sh)
set -o xtrace

git clone -b v6.3.0 https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit.git HiggsAnalysis/CombinedLimit
git clone https://github.com/cms-analysis/CombineHarvester.git CombineHarvester
git clone git@github.com:annawoodard/EffectiveTTV.git EffectiveTTV/EffectiveTTV

scram b -j 32

cd ../..
scramv1 project CMSSW_8_1_0_pre16 # needed for a harmonious numpy environment
ln -s  CMSSW_7_4_7/src/EffectiveTTV/EffectiveTTV/python CMSSW_8_1_0_pre16/python/EffectiveTTV/EffectiveTTV
) > setup.log


cat <<EOF
=========================================================================
output is in setup.log
=========================================================================
EOF
