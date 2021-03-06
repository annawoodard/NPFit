#!/bin/sh

cat <<EOF
=========================================================================
this script creates a working directory for the NPFit analysis
output is in setup.log
=========================================================================
EOF

(
set -e
set -o xtrace

if ! type makeflow > /dev/null; then
   echo "cctools is required; for instructions please visit http://ccl.cse.nd.edu/software/downloadfiles.php"
   exit 1
fi

export SCRAM_ARCH=slc6_amd64_gcc530
cmsrel CMSSW_8_1_0
cd CMSSW_8_1_0/src
cmsenv
git clone https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit.git HiggsAnalysis/CombinedLimit
cd HiggsAnalysis/CombinedLimit
git fetch origin
git checkout 81x-root606
cd $CMSSW_BASE/src
git clone https://github.com/cms-analysis/CombineHarvester.git CombineHarvester
git clone git@github.com:annawoodard/NPFit.git NPFit/NPFit

git clone git@github.com:annawoodard/NPFitProduction.git NPFitProduction/NPFitProduction
cd NPFitProduction/NPFitProduction
git config core.sparsecheckout true
echo "python/" >> .git/info/sparse-checkout
echo "scripts/" >> .git/info/sparse-checkout
git checkout master
cd $CMSSW_BASE/src

scramv1 b clean
scramv1 b -j 2

pip install --user tabulate
pip install --user seaborn

) > setup.log

cat <<EOF
=========================================================================
output is in setup.log
=========================================================================
EOF
