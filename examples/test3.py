import aiida_QECpWorkChain.workflow3 as wf2


import sys
import pymatgen.core
import pymatgen as pmg
import numpy as np
import copy


def struct_gen_original(what='IV'):
    # structure from
    # https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.76.74
    if what=='IV':
        ortho_abc=np.array((3.1868,5.5433,5.2559))
        lattice=pmg.core.lattice.Lattice.from_parameters(*ortho_abc,90.0,90.0,90.0)

        coord=np.array([
        [0.2429, 0.3496, 0.2562],
        [0.3667, 0.1779, 0.2301],
        [-0.0469, 0.3216, 0.3212],
        [0.2114, 0.4290, 0.0852]
        ])
    elif what=='V':
        ortho_abc=np.array((2.9215,5.0921,4.8056))
        lattice=pmg.core.lattice.Lattice.from_parameters(*ortho_abc,90.0,90.0,90.0)

        coord=np.array([
        [0.2388, 0.3363,0.2552],
        [0.3508, 0.1415, 0.2321],
        [-0.0823, 0.3145, 0.3226],
        [0.2550, 0.4559, 0.0731]
        ])        
    structure=pmg.core.structure.Structure.from_spacegroup('P212121',lattice,['N','H','H','H'],coord,coords_are_cartesian=False)
    return structure



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
StructureData = DataFactory('core.structure')

aiida_structure_IV = aiida.orm.StructureData(pymatgen_structure=struct_gen_original(what='IV'))
aiida_structure_IV.store()

wg=wf2.build_and_test(code=Code.get_from_string('cp@localhost'),pw_code = Code.get_from_string('pw@localhost'), pseudo_family='SSSP/1.3/PBE/efficiency',structure=aiida_structure_IV,ecutwfc=42.0,resources={'resources':{'num_machines':1,'num_mpiprocs_per_machine':1},'wallclock':60},additional_parameters={
    'SYSTEM': {
        'nr1b': 5,
        'nr2b': 5,
        'nr3b': 5,
        }
    },dt=6.0,nstep=5,ion_velocities=None)
wg.submit()
