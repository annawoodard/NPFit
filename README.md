## Installation

CCTools is required. To install it:

    cd $HOME
    slc=$(egrep "Red Hat Enterprise|Scientific|CentOS" /etc/redhat-release | sed 's/.*[rR]elease \([0-9]*\).*/\1/')
    wget http://ccl.cse.nd.edu/software/files/cctools-6.0.10-x86_64-redhat$slc.tar.gz
    gunzip cctools-6.0.10-x86_64-redhat6.tar.gz
    tar xvf cctools-6.0.16-x86_64-redhat$slc.tar
    # add following line to your login script
    export PATH=$HOME/cctools-6.0.16-x86_64-redhat$slc/bin:$PATH

Now set up a working area:

    curl https://raw.githubusercontent.com/annawoodard/EffectiveTTV/master/setup.sh|sh -


