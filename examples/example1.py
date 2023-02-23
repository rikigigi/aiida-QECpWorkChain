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
Qecp=WorkflowFactory('qecpworkchain.cp')
StructureData = DataFactory('core.structure')

aiida_structure_IV = aiida.orm.StructureData(pymatgen_structure=struct_gen_original(what='IV'))
aiida_structure_IV.store()
input_structures=[
    aiida_structure_IV
]
alphas = [
'phase IV'
]
submitted_pks=[]
for alpha,aiida_structure in zip(alphas,input_structures):

    build=Qecp.get_builder()
    #* pk 165117 - cp-gpu-m100@m100
    #* pk 165118 - pw-gpu-m100@m100
    
    pw=Code.get_from_string('pw@localhost')
    cp=Code.get_from_string('cp@localhost')
    resources={
        'resources' : {
            'num_machines' : 1,
            'num_mpiprocs_per_machine' : 1
        },
        'wallclock' : 60*60*12,
        'queue' : 'myqueue',
        'account': 'dummy',
    }
    resourcespw={
        'resources' : {
            'num_machines' : 1,
            'num_mpiprocs_per_machine' : 1,
        },
        'wallclock' : 60*60*12,
        'queue' : 'myqueue',
        'account': 'dummy',
    }
    additional_parameters_cp={
        'SYSTEM' :{ 'nr1b': 18, 'nr2b': 18, 'nr3b': 18 },
        'CONTROL' : {'isave' : 3000, },
    }
    build.cp_code=cp
    build.pw_code=pw
    build.cp_resources_cp_list=List([resources])
    build.cp_resources_cg_list=List([resources])
    build.pw_resources_list=List([resourcespw])
    build.structure=aiida_structure
    build.ecutwfc=Float(70.0)
    #build.skip_emass_dt_test=Bool(True)
    build.skip_parallel_test=Bool(True)
    build.pseudo_family=Str('pseudodojo')
    build.target_force_ratio=Float(0.95)
    build.additional_parameters_cp=Dict(additional_parameters_cp)
    build.emass_list=List([20])
    build.max_slope_min_emass=Float(20.0)
    build.dt_start_stop_step=List([2.8,2.9,1.0])
    build.number_of_pw_per_trajectory=Int(15)
    build.nve_required_picoseconds=Float(1.0)
    build.nstep_initial_cg=Int(10)
    eq_t=0.05
    nh_t=0.1
    build.thermobarostat_points=List([
                                         {"temperature_K": 200, "pressure_KBar": 100, "equilibration_time_ps": eq_t, "thermostat_time_ps": nh_t},
                                         {"temperature_K": 700, "pressure_KBar": 100, "equilibration_time_ps": eq_t, "thermostat_time_ps": nh_t},
                                           ])
    build.temperature_tolerance=Float(100.0) #K
    build.pressure_tolerance=Float(100.0) #Kbar
    build.metadata.label=f'10GPa 200-700K {alpha}'
    build.metadata.description="Phase IV is almost HPC"
    res=submit(build)
    print ('workflow {} submitted (cross your fingers)'.format(res))
    submitted_pks.append(res.pk)
import numpy as np
with open('submitted.pks', "ab") as f:
	np.savetxt(f, submitted_pks, fmt='%i')
