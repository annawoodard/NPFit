This is the code used for the Effective Field Theory interpretation in the CMS TOP-17-005 ([CADI](http://cms.cern.ch/iCMS/analysisadmin/cadilines?line=TOP-17-005)) analysis. Code used for running Madgraph to calculate cross sections is located in a separate repository which can be found [here](https://github.com/annawoodard/EffectiveTTVProduction).

## Installation

CCTools is required. At Notre Dame, add this to your login script:

    cctools=lobster-148-c1a7ecbd-cvmfs-0941e442
    export PYTHONPATH=$PYTHONPATH:/afs/crc.nd.edu/group/ccl/software/x86_64/redhat6/cctools/$cctools/lib/python2.6/site-packages
    export PATH=/afs/crc.nd.edu/group/ccl/software/x86_64/redhat6/cctools/$cctools/bin:$PATH

If you are not running at Notre Dame, you will need to setup [cctools](https://ccl.cse.nd.edu/software/) yourself:

    cd $HOME
    slc=$(egrep "Red Hat Enterprise|Scientific|CentOS" /etc/redhat-release | sed 's/.*[rR]elease \([0-9]*\).*/\1/')
    wget http://ccl.cse.nd.edu/software/files/cctools-lobster-148-c1a7ecbd-cvmfs-0941e442-x86_64-redhat$slc.tar.gz
    tar xvf cctools-lobster-148-c1a7ecbd-cvmfs-0941e442-x86_64-redhat$slc.tar.gz
    # add following line to your login script
    export PATH=$HOME/cctools-lobster-148-c1a7ecbd-cvmfs-0941e442-x86_64-redhat$slc/bin:$PATH

Now set up a working area:

    curl https://raw.githubusercontent.com/annawoodard/EffectiveTTV/master/setup.sh|sh -

## Quick start
To reproduce the TOP-17-005 plots, no modification of the configuration file is necessary (with caveats [1]) After `cmsenv`ing in your working area, setup the output directory and produce the Makeflow specification:

    run make data/config.py

Follow the instructions which will be printed to the screen. They should look similar to this:

    # to run, issue the following commands:
    cd /afs/crc.nd.edu/user/a/awoodard/www/ttV/1
    nohup work_queue_factory -T condor -M ttV_FTW -C /afs/crc.nd.edu/user/a/awoodard/releases/CMSSW_8_1_0/src/EffectiveTTV/EffectiveTTV/data/factory.json >& factory.log &
    makeflow -T wq -M ttV_FTW --shared-fs '/afs'
Note that you only need to run the `work_queue_factory` command once; you can leave it running and it will only submit jobs as they are needed for any makeflow process matching `label` in [test/config.py](test/config.py) (there is no reason to change the label between runs).

[1] Caveats:

* If you have a `www` directory which is not located at `$HOME/www`, you should modify `outdir` in the config. A web-enabled output directory is not a requirement but makes viewing plots more convenient.
* If you are not using the HTCondor batch system, modify `batch type` (supported systems: local, wq, condor, sge, torque, moab, slurm, chirp, amazon).
* If you are submitting to machines which will not have your output directory mounted, change `outdir shared` to `False`.

## More details
### Main idea

### Reproducibility
Calling `run make data/config.py` produces an output directory (specified as `outdir` in the [config](test/config.py)) which contains:
1) all of the inputs (starting from datacards and including the config itself),
2) all of the outputs (plots, etc),
3) all of the commands run (specified in `Makefile`), and
4) a git patch in order to exactly reproduce the code as it was the last time `makeflow` was called. Instructions for reproducing the code are saved to `outdir` in a text file called `README.txt`, which should contain something like the following:
    ```
    cd /afs/crc.nd.edu/user/a/awoodard/releases/CMSSW_8_1_0/python/EffectiveTTV/EffectiveTTV
    git checkout d7bb6f7
    git apply /afs/crc.nd.edu/user/a/awoodard/www/ttV/1/patch.diff

    ```
It is recommended that you iterate the version number at the end of `outdir` periodically. In this way, you should be able to keep track of differences as the analysis evolves. Note that `makeflow` works similarly to a makefile; if you delete an output and run makeflow again, all of the outputs which depend on that file will also be re-run.
