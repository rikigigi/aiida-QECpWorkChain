import aiida_workgraph
import aiida_workgraph.socket
from .workflow import *

import typing

from aiida_quantumespresso.calculations.cp import CpCalculation
from aiida_quantumespresso.calculations.pw import PwCalculation


def link_restart_cp(
    wg: aiida_workgraph.WorkGraph,
    link_from: aiida.orm.CalcJobNode,
    name: str,
    code: aiida.orm.AbstractCode,
    pseudos: dict,
    metadata: dict,
    dt: float = 2.0,
    dtchange: bool = False,
    mu: float = 50,
    mucut: float = 2.5,
    nstep: int = 9999999,
    copy_mu_mucut: bool = False,
    remove_autopilot: bool = False,
    additional_parameters: None | dict = None,
    cg: bool = False,
    remove_parameters_namelist: aiida_workgraph.socket.Any | None = None,
    cmdline: None | aiida_workgraph.socket.Any = None,
    ttot_ps: float | None = None,
    stepwalltime_s: float | None = None,
    tstress: bool = False,
    from_scratch: bool=False,
    structure: aiida.orm.StructureData | None = None,
):
    #prepare the parameter input dictionary by copying it from the link_from calcjob
    parameters=copy.deepcopy(link_from.inputs.parameters.value.get_dict())
    if remove_parameters_namelist is not None:
        for itemtodel in remove_parameters_namelist:
            parameters.pop(itemtodel)
    parameters.setdefault('CONTROL',{})
    parameters['CONTROL']['calculation'] = 'cp'
    parameters['CONTROL']['restart_mode'] = 'from_scratch' if from_scratch else  'restart'
    parameters['CONTROL']['tstress'] = tstress
    parameters['CONTROL']['tprnfor'] = True
    parameters['IONS']['ion_velocities'] = 'from_input' if from_scratch else 'default' 
    parameters['ELECTRONS']['electron_velocities'] = 'default'
    if not cg:
        parameters['ELECTRONS']['orthogonalization'] = 'ortho'
        parameters['ELECTRONS']['electron_dynamics'] = 'verlet'
    else:
        parameters['ELECTRONS']['orthogonalization'] = 'Gram-Schmidt'
        parameters['ELECTRONS']['electron_dynamics'] = 'cg'
    if not copy_mu_mucut:
        parameters['ELECTRONS']['emass'] = mu
        parameters['ELECTRONS']['emass_cutoff'] = mucut
    elif not 'emass' in link_from.inputs.parameters.value['ELECTRONS']:
        raise ValueError('emass parameter not found in input dictionary!')
    if dt is not None:
        if not from_scratch:
            if abs(float(parameters['CONTROL']['dt']) - dt) > 1e-5 and dtchange:
                parameters['IONS']['ion_velocities'] = 'change_step'
                parameters['IONS']['tolp'] = parameters['CONTROL']['dt']
                parameters['ELECTRONS']['electron_velocities'] = 'change_step'
        parameters['CONTROL']['dt'] = dt
    if additional_parameters is not None:
        for key in additional_parameters.keys():
            for subkey in additional_parameters[key].keys():
                parameters.setdefault(key,{})[subkey]=additional_parameters[key][subkey]
    if ttot_ps is not None:
        dt_ps=parameters['CONTROL']['dt']*qeunits.timeau_to_sec*1.0e12
        #note: nstep is increased by iprint so the trajectory is written
        nstep=int(ttot_ps/dt_ps+1)
        if 'iprint' in parameters['CONTROL']:
            nstep=nstep+2*parameters['CONTROL']['iprint']
        else:
            nstep=nstep+2*10
        print('nstep={}, dt = {} ps'.format(nstep,dt_ps))
    wallclock=metadata['options']['max_wallclock_seconds']
    if stepwalltime_s is not None:
        wallclock_max=wallclock
        wallclock=nstep*stepwalltime_s*1.25+120.0
        wallclock=wallclock if wallclock < wallclock_max else wallclock_max
        print('wallclock requested: {} s'.format(wallclock))
   
    parameters['CONTROL']['nstep'] = nstep
    parameters['CONTROL']['max_seconds'] = int(wallclock*0.9)

    #settings dictionary
    if 'settings' in link_from.inputs:
        settings = copy.deepcopy(link_from.inputs.settings.value.get_dict())
    else:
        settings = {}
    if remove_autopilot and 'AUTOPILOT' in settings:
        del settings['AUTOPILOT']
        print ('removed AUTOPILOT input')
    if cmdline is not None:
        settings['cmdline']=cmdline
        
    #metadata dictionary: put it in the input
        
    new_task = wg.add_task(CpCalculation, pseudos=pseudos, code=code, structure=structure, name=name, parameters=parameters, settings=settings, metadata=metadata)
    
    if from_scratch:
        #get the structure and velocities from the trajectory and do not set the parent_folder
        struct_velocities = wg.add_task(get_structure_and_velocities, name=f'get_structure_and_velocities_{name}',
                            trajectory_or_structure=link_from.outputs['output_trajectory'])

        wg.add_link(struct_velocities.outputs['start_structure'],new_task.inputs['structure'])
        
        #NOTE: we cannot do the following because settings.ATOMIC_VELOCITIES is not a valid port: it is a key in the settings dictionary  
        #wg.add_link(struct_velocities.outputs['start_velocities_A_au'],new_task.inputs['settings.ATOMIC_VELOCITIES'])
        
        generate_settings = wg.add_task(get_settings_dictionary, name=f'get_settings_dictionary_{name}',
                            settings=settings, atomic_velocities=struct_velocities.outputs['start_velocities_A_au'])
        wg.add_link(generate_settings.outputs['settings_dictionary'],new_task.inputs['settings']) #NOTE: this is overwriting the settings dictionary
        wg.add_link(struct_velocities.outputs['start_structure'],new_task.inputs['structure']) #NOTE: this is overwriting the structure
    else:
        wg.add_link(link_from.outputs['remote_folder'],new_task.inputs["parent_folder"])
        
    
    
    return new_task

@aiida_workgraph.task.calcfunction(outputs=[{'name':'settings_dictionary'}])
def get_settings_dictionary(settings: dict, atomic_velocities = None):
    if atomic_velocities is not None:
        settings['ATOMIC_VELOCITIES'] = atomic_velocities
    return {'settings_dictionary': settings}

#workgraph that takes a trajectory from a cp calculation and runs N pw calculations on N timesteps of it
#NOTE: workgraph dynamically created during the workflow execution. Number of tasks depend on the number of steps in the trajectory, known after execution
@aiida_workgraph.task.graph_builder(
    outputs=[{"name": "force_ratios", "from": "context.force_ratios"}]
)
def run_many_pw_on_trajectory(trajectory: aiida.orm.TrajectoryData, n: int,
                            pw_code: aiida.orm.Code, pseudos: dict, ecutwfc: float,
                            cmdline = None,
                            tstress: bool = False, nbnd: int | None =None, options: dict | None = None):
    parameters= {
        'CONTROL': {
            'calculation': 'scf',
            'restart_mode': 'from_scratch',
            'tstress': tstress,
            'tprnfor': True,
        },
        'SYSTEM': {
            'ecutwfc': ecutwfc,
        },
        'ELECTRONS': {
        },
    }
    if nbnd is not None:
        parameters['SYSTEM']['nbnd'] = nbnd
    settings = {'gamma_only': True}
    if cmdline is not None:
        settings['cmdline'] = cmdline
    wg = aiida_workgraph.WorkGraph()
    n_steps = trajectory.numsteps #this is known only after the calculation
    step = n_steps//n
    if step <= 0:
        step=1
    kpoints = aiida.orm.KpointsData()
    kpoints.set_kpoints_mesh([1,1,1])
    if options is None:
        options = {}
    compare_forces_many = []
    for i in range(0,n_steps,step):
        new_task = wg.add_task(PwCalculation, kpoints=kpoints, pseudos=pseudos, code=pw_code, structure=trajectory.get_step_structure(i), name=f'pw_on_trajectory_{i}', parameters=parameters, settings=settings, metadata={'options': options})
        compare_forces_task = wg.add_task(compare_forces, cp_trajectory=trajectory, pw_trajectory=new_task.outputs["output_trajectory"], cp_traj_idx=i)
        compare_forces_many.append(compare_forces_task)
        # save the result of compare_forces_task in the context.force_ratios dictionary
        compare_forces_task.set_context({f'force_ratios.step_{i}': 'result'})
    return wg
    
#function that compare the forces of the cp and pw calculations
@aiida_workgraph.task.calcfunction()
def compare_forces(cp_trajectory: aiida.orm.TrajectoryData, pw_trajectory: aiida.orm.TrajectoryData, cp_traj_idx: int):
    cp_forces = cp_trajectory.get_array('forces')[cp_traj_idx.value]
    pw_forces = pw_trajectory.get_array('forces')[0]
    forces_ratio = aiida.orm.List((np.array(pw_forces)/np.array(cp_forces)).tolist())
    return forces_ratio


@aiida_workgraph.task.calcfunction(outputs=[{'name': 'start_structure','identifier':'workgraph.aiida_structuredata'},{'name': 'start_velocities_A_au'}])
def get_structure_and_velocities(trajectory_or_structure : aiida.orm.TrajectoryData|aiida.orm.StructureData):
    outputs={
        'start_structure':trajectory_or_structure.get_step_structure(trajectory_or_structure.numsteps-1) if isinstance(trajectory_or_structure,aiida.orm.TrajectoryData) else trajectory_or_structure,
        'start_velocities_A_au':trajectory_or_structure.get_step_data(trajectory_or_structure.numsteps -1)['velocities'] if isinstance(trajectory_or_structure,aiida.orm.TrajectoryData) and 'velocities' in trajectory_or_structure.get_arraynames() else None
    }
    
def wg_configure_cp_builder_cg(wg: aiida_workgraph.WorkGraph, code : aiida.orm.Code,pseudo_family : str,
                            structure : aiida.orm.StructureData,
                            ecutwfc : float,
                            resources : dict,
                            additional_parameters : dict,
                            dt : float=3.0,
                            nstep : int=50,
                            temp: float | None = None,
                            ion_velocities: typing.Any|None = None,): #NOTE: List[List[float]] does not work
    builder = configure_cp_builder_cg(
        code,
        pseudo_family,
        structure,
        ecutwfc,
        resources=resources,
        additional_parameters=additional_parameters,
        dt=dt,
        nstep=nstep,
        ion_velocities=ion_velocities,
        tempw=temp if temp is not None else 300.0
    )
    
    builder_unwrapped = replace_data(builder._data)
    #wg.add_task(CpCalculation, name='cg_born_oppenheimer')
    return builder_unwrapped,wg.add_task(CpCalculation, **builder_unwrapped, name='cg_born_oppenheimer') #NOTE: builder_unwrapped is ugly


#function that recursively loops over the builder and replaces all the members that have the _data attribute with its content
def replace_data(dict_):
    for key in dict_.keys():
        if hasattr(dict_[key],'_data'):
            dict_[key]=replace_data(dict_[key]._data)
        elif isinstance(dict_[key],dict):
            dict_[key]=replace_data(dict_[key])
    return dict_

@aiida_workgraph.task.graph_builder(
    outputs=[{"name": "force_ratios", "from": "context.force_ratios"}]
)
def build_and_test(code : aiida.orm.Code, pw_code: aiida.orm.Code, pseudo_family : str,
                            structure : aiida.orm.StructureData,
                            ecutwfc : float,
                            resources : dict,
                            additional_parameters : dict,
                            dt : float=3.0,
                            nstep : int=50,
                            temp: float | None = None,
                            ion_velocities: typing.Any|None = None,): #NOTE: same input as an other workgraph
    wg = aiida_workgraph.WorkGraph()
    kwargs = {
        'code': code,
        'pseudo_family': pseudo_family,
        'structure': structure,
        'ecutwfc': ecutwfc,
        'resources': resources,
        'additional_parameters': additional_parameters,
        'dt': dt,
        'nstep': nstep,
        'temp': temp,
        'ion_velocities': ion_velocities
    }
    builder_unwrapped, added_task=wg_configure_cp_builder_cg(wg,**kwargs)
    kwargs2={
        'code': code,
        'nstep': 20,
    }
    test_dt_emass_list = [(25.0,2.0),(25.0,5.0), (50.0, 2.0),( 50.0, 5.0)]
    
    #link all the tasks
    for mu,dt in test_dt_emass_list:
        cp=link_restart_cp(wg, link_from=added_task, metadata=builder_unwrapped['metadata'],pseudos=builder_unwrapped['pseudos'], structure=structure, name=f'test_mu_dt_{mu}_{dt}'.replace('.','d'),nstep=20,code=code,mu=mu, dt=dt) #NOTE: if I start with a number, I have a error from the worker: 'ValueError: invalid link label `25d0_2d0`: not a valid python identifier'. It also cannot start with an underscore
        many_pw_task_wg = wg.add_task(run_many_pw_on_trajectory, trajectory=cp.outputs['output_trajectory'], n=10,
                                      pw_code=pw_code, pseudos=builder_unwrapped['pseudos'], ecutwfc=ecutwfc, 
                                      options={
                                            'resources': resources['resources'],    
                                      })
        # save the force ratios in the context
        key = f"{str(mu).replace('.','d')}_{str(dt).replace('.','d')}"
        many_pw_task_wg.set_context({f'force_ratios.{key}': 'force_ratios'})
    
    return wg

    