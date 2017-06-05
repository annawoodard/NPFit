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

pip install --user tabulate

cd ../..
scramv1 project CMSSW_8_1_0_pre16 # needed for a harmonious numpy environment
ln -s  CMSSW_7_4_7/src/EffectiveTTV/EffectiveTTV/python CMSSW_8_1_0_pre16/python/EffectiveTTV/EffectiveTTV
ln -s  CMSSW_7_4_7/src/EffectiveTTV/EffectiveTTV/data CMSSW_8_1_0_pre16/src/EffectiveTTV/EffectiveTTV
ln -s /afs/crc.nd.edu/user/a/awoodard/releases/effective-ttV/CMSSW_7_4_7/python/EffectiveTTV/EffectiveTTV /afs/crc.nd.edu/user/a/awoodard/releases/effective-ttV/CMSSW_8_1_0_pre16/python/EffectiveTTV/EffectiveTTV
# FIXME need to use absolute paths in ln -s above otherwise they won't
# work
) > setup.log

pip install --upgrade --user matplotlib
pip install --user rootpy
export PYTHONPATH=/afs/crc.nd.edu/user/a/awoodard/.local/lib/python2.7/site-packages:$PYTHONPATH

cat <<EOF
=========================================================================
output is in setup.log
=========================================================================
EOF
