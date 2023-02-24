# aiida-QECpWorkChain

Car-Parrinello Work Chain. This code was used to perform the simulation work of this [PhD thesis](https://iris.sissa.it/handle/20.500.11767/130650)

## Usage

 - setup your aiida postgresql database, rabbitmq, and AiiDA. This can be done following [the official documentation](https://aiida.readthedocs.io/projects/aiida-core/en/latest/intro/install_conda.html#intro-get-started-conda-install)
 - install the package via `pip install .`
 - setup aiida-quantumespresso by configuring the remote computers and the remote code (`cp.x` and `pw.x`) as described in [the official documentation](https://aiida-quantumespresso.readthedocs.io/en/latest/installation/index.html). Load the pseudopotential in the database
 - as a starting point you can have a look at the example in [examples/example1.py](examples/example1.py), and change it according to your need. In particular you will need to modify the remote computer configuration and the pseudopotential family, and of course the starting configuration.