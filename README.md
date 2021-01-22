[![Build Status](https://github.com/rikigigi/aiida-QECpWorkChain/workflows/ci/badge.svg?branch=master)](https://travis-ci.org/rikigigi/aiida-QECpWorkChain/actions)
[![Coverage Status](https://coveralls.io/repos/github/rikigigi/aiida-QECpWorkChain/badge.svg?branch=master)](https://coveralls.io/github/rikigigi/aiida-QECpWorkChain?branch=master)
[![Docs status](https://readthedocs.org/projects/aiida-QECpWorkChain/badge)](http://aiida-QECpWorkChain.readthedocs.io/)
[![PyPI version](https://badge.fury.io/py/aiida-QECpWorkChain.svg)](https://badge.fury.io/py/aiida-QECpWorkChain)

# aiida-QECpWorkChain

Car-Parrinello Work Chain

## Usage

Here goes a complete example of how to submit a test calculation using this plugin.

A quick demo of how to submit a calculation:
```python
import pymatgen as pmg
import numpy as np
import copy
from aiida import load_profile
load_profile()
from aiida.orm import Code
import aiida.orm
from aiida.orm.nodes.data.upf import get_pseudos_from_structure
from aiida.engine import submit
from aiida.orm.nodes.data import *
import re
from aiida.engine import submit

from aiida.plugins import DataFactory, WorkflowFactory
Qecp=WorkflowFactory('qecpworkchain.cp')
StructureData = DataFactory('structure')


#load your initial structure
aiida_structure=aiida.orm.load_node(29073)
 
#eventually build a supercell
supercell=aiida_structure.get_pymatgen_structure()
supercell.make_supercell([3,3,3])
aiida_structure=aiida.orm.StructureData(pymatgen_structure=supercell)



build=Qecp.get_builder()

#cluster parameters
pw=Code.get_from_string('pw-6.7@your_cluster')
cp=Code.get_from_string('cp-6.7@your_cluster')
#for cp code
resources={
    'resources' : {
        'num_machines' : 4,
        'num_mpiprocs_per_machine' : 48,
    },
    'wallclock' : 60*60*12,
    'queue' : 'your_queue',
    'account': 'your_account',
}
#for pw code
resourcespw={
    'resources' : {
        'num_machines' : 1,
        'num_mpiprocs_per_machine' : 48,
    },
    'wallclock' : 600,
    'queue' : 'skl_usr_prod',
    'account': 'Sis20_baroni',
}
additional_parameters_cp={
    'SYSTEM' :{ 'nr1b': 23, 'nr2b': 23, 'nr3b': 23 },
    'CONTROL' : {'isave' : 3000, },
}
build.cp_code=cp
build.pw_code=pw
build.cp_resources_cp_list=List(list=[resources])
build.cp_resources_cg_list=List(list=[resources])
build.pw_resources_list=List(list=[resourcespw])
build.structure=aiida_structure
build.ecutwfc=Float(85.0)
build.pseudo_family=Str('oncvpsp-4.0.1') #your pseudopotential
build.target_force_ratio=Float(0.95)
build.additional_parameters_cp=Dict(dict=additional_parameters_cp)
build.emass_list=List(list=[90]) #put here a list of emass to test
build.dt_start_stop_step=List(list=[4.0,5.9,2.0]) #as in np.arange
build.number_of_pw_per_trajectory=Int(15)
build.nve_required_picoseconds=Float(10.0) #length of the production trajectory
#list of T,P points where the workchain will do the thermobarostatation, then will check the equilibration with a nve simulation
#if equilibrated will go to the next point
build.thermobarostat_points=List(list=[
                                     {"temperature_K": 300, "pressure_KBar": 100 , "equilibration_time_ps": 5.0, "thermostat_time_ps": 5.0},
                                     {"temperature_K": 400, "pressure_KBar": 150 , "equilibration_time_ps": 5.0, "thermostat_time_ps": 5.0},
                                     {"temperature_K": 600, "pressure_KBar": 300 , "equilibration_time_ps": 5.0, "thermostat_time_ps": 5.0},
                                     {"temperature_K": 800, "pressure_KBar": 300 , "equilibration_time_ps": 5.0, "thermostat_time_ps": 5.0}
                                       ])
build.metadata.label="your_label@800K,30GPa"
build.metadata.description="Your long description"
res=submit(build)
print ('workflow {} submitted (cross your fingers)'.format(res))
```
