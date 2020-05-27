import sys
import pymatgen as pmg
import numpy as np
import copy



print ('current path:')
for p in sys.path:
    print (p)



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

aiida_structure=aiida.orm.load_node(42) #some node where you have a structure

supercell=aiida_structure.get_pymatgen_structure() # eventually replicate the structure
supercell.make_supercell([2,2,2])
aiida_structure=aiida.orm.StructureData(pymatgen_structure=supercell)




build=Qecp.get_builder()


#define computing resources
pw=Code.get_from_string('qe-6.5@ulysses')
cp=Code.get_from_string('cp-6.5@ulysses')
resources={
    'resources' : {
        'num_machines' : 1,
        'num_mpiprocs_per_machine' : 20,
    },
    'wallclock' : 3600*3,
    'queue' : 'regular1',
}
resourcespw={
    'resources' : {
        'num_machines' : 1,
        'num_mpiprocs_per_machine' : 20,
    },
    'wallclock' : 600,
    'queue' : 'regular1',
}

additional_parameters_cp={
    'SYSTEM' :{ 'nr1b': 23, 'nr2b': 23, 'nr3b': 23, 'ibrav':1 },
    'CONTROL' : {'isave' : 3000, },
}
build.cp_code=cp
build.pw_code=pw
build.cp_resources_cp_list=List(list=[resources])
build.cp_resources_cg_list=List(list=[resources])
build.pw_resources_list=List(list=[resourcespw])
build.structure=aiida_structure
build.ecutwfc=Float(85.0)
build.pressure=Float(700.0)
build.pseudo_family=Str('oncvpsp-4.0.1') # define pseudo family
build.target_force_ratio=Float(0.9)
build.tempw_initial_random=Float(1000.0)
build.tempw_initial_nose=Float(500.0)
build.tempw=Float(1000.0)
build.nthermo_cycle=Int(3)
build.additional_parameters_cp=Dict(dict=additional_parameters_cp)
build.emass_start_stop_step_mul=List(list=[1.0,5.0,2.0,75]) #try some emasses
build.dt_start_stop_step=List(list=[4.0,10.0,2.0]) #try some dts
build.number_of_pw_per_trajectory=Int(15)
build.nve_required_picoseconds=Float(10.0)
res=submit(build)
print ('workflow {} submitted (cross your fingers)'.format(res))
