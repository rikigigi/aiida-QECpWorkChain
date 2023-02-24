import copy
from enum import Enum

import aiida.orm
from aiida.orm import Int, Float, Str, List, Dict, ArrayData, Bool
from aiida.engine import WorkChain, calcfunction, ToContext, append_, while_, if_, return_
from aiida.orm.nodes.data.upf import get_pseudos_from_structure
from aiida.plugins.factories import DataFactory
import numpy as np
import qe_tools
qeunits=qe_tools.CONSTANTS


from .utils import *

def validate_structure(structure_or_trajectory,port):
    return

def validate_pseudo_family(pseudo_family_str,port):
    return

def validate_cp_list(cp_code,port):
    return

def validate_pw_list(pw_code,port):
    return

def validate_ecutwfc(ecutwfc,port):
    if ecutwfc <= 0.0:
        return 'ecutwfc must be a positive number'
    return

def validate_tempw(tempw,port):
    if tempw <= 0.0:
        return 'tempw must be a positive number'
    return

def configure_cp_builder_cg(code,
                            pseudo_family,
                            aiida_structure,
                            ecutwfc,
                            tempw,
                            resources, #={ 'num_machines': 1, 'num_mpiprocs_per_machine': 20},
                            additional_parameters={},
                            nstep=50,
                            ion_velocities=None,
                            dt=3.0,
                            print=print
                            #queue,#='regular1', #inside resources
                            #wallclock=3600,
                            #account=None
                           ):
    builder=code.get_builder()
    builder.structure = aiida_structure
    #settings_dict = {
    #    'cmdline': ['-n', '16'],
    #}
    #builder.settings = Dict(dict=settings_dict)
    #The functions finds the pseudo with the elements found in the structure.
    builder.pseudos = aiida.orm.nodes.data.upf.get_pseudos_from_structure(aiida_structure,pseudo_family.value)
    parameters = {
    'CONTROL' : {
        'calculation': 'cp' ,
        'restart_mode' : 'from_scratch',
        'tstress': False ,
        'tprnfor': True,
        'dt': dt,
        'nstep' : nstep,
    },
    'SYSTEM' : {
        'ecutwfc': ecutwfc,
#        'nr1b' : nrb[0],
#        'nr2b' : nrb[1],
#        'nr3b' : nrb[2],
    },
    'ELECTRONS' : {
         'emass': 25,
         #'orthogonalization' : 'ortho',
         #'electron_dynamics' : 'verlet',
         'orthogonalization' : 'Gram-Schmidt',
         'electron_dynamics' : 'cg',
    },
    'IONS' : {
        'ion_dynamics' : 'verlet',
        #'ion_velocities' : 'default',
        'ion_velocities' : 'random' if ion_velocities is None else 'from_input',
        'tempw' : tempw , 
    },
#    'CELL' : {
#        'cell_dynamics' : 'none',
#    },
    }
    settings={}
    autopilot=None
    if ion_velocities is not None:
        settings['ATOMIC_VELOCITIES']=ion_velocities
    if nstep>22:
        autopilot = [
        {'onstep' : 7, 'what' : 'dt', 'newvalue' : 6.0 },
        {'onstep' : 14, 'what' : 'dt', 'newvalue' : 18.0},
        {'onstep' : 21, 'what' : 'dt', 'newvalue' : 60.0},
        {'onstep' : nstep-1, 'what' : 'dt', 'newvalue' : dt},
        ]
    elif nstep>10:
        autopilot = [
        {'onstep' : 7, 'what' : 'dt', 'newvalue' : 15.0 },
        {'onstep' : nstep-1, 'what' : 'dt', 'newvalue' : dt},
        ]
    if autopilot is not None:
        settings['AUTOPILOT']=autopilot
    
    for key in additional_parameters.keys():
        for subkey in additional_parameters[key].keys():
            parameters.setdefault(key,{})[subkey]=additional_parameters[key][subkey]

    builder.settings = Dict(settings)
    builder.parameters = Dict(parameters)
    builder.metadata.options.resources = resources['resources']
    builder.metadata.options.max_wallclock_seconds = resources['wallclock']
    builder.metadata.options.queue_name = resources['queue']
    if 'account' in resources:
        builder.metadata.options.account = resources['account']
    return builder


def get_resources(calc):
    start_from=get_node(calc)
    options=start_from.get_options()
    if 'account' in options:
        return options['resources'], options['queue_name'], options['max_wallclock_seconds'], options['account']
    else:
        return options['resources'], options['queue_name'], options['max_wallclock_seconds'], None

def configure_cp_builder_restart(code,
                                    start_from,
                                    dt=None,
                                    dtchange=True,
                                    mu=50,
                                    mucut=2.5,
                                    nstep=999999,
                                    resources=None,
                                    copy_mu_mucut=False,
                                    remove_autopilot=True,
                                    additional_parameters={},
                                    cg=False,
                                    remove_parameters_namelist=[],
                                    cmdline=None,
                                    structure=None,
                                    ttot_ps=None,
                                    stepwalltime_s=None,
                                    print=print,
                                    tstress=True,
                                    from_scratch=False
                                ):
    '''
    rescaling of atomic and wfc velocities is performed, if needed, using the dt found in the CONTROL namelist.
    If dt is changed with autopilot this can be wrong.
    max_seconds is changed according to 0.9*wallclock, where wallclock is the time requested to the scheduler.
    nstep can be calculated from the simulation time required if ttot_ps is given (in picoseconds)
    If stepwalltime is given, the time requested to the scheduler is calculated as nstep*stepwalltime*1.25+120.0, up to a maximum of walltime.
    It performs a restart. In general all parameters but the one that are needed to perform a CP dynamics
    are copied from the parent calculation. tstress and tprnfor are setted to True by default.
    additional_parameters is setted at the end, so it can override every other parameter setted anywhere before
    or during this function call.
    remove_parameters_namelist are removed at the beginning
    cmdline can be, for example, ['-ntg', '2'] or a longer list. if not specified it is copied from the old calculation
    If from_scratch is True, restart the calculation copying velocities and positions from last trajectory output step of start_from, and perform a cg. valid only with cg=True
    '''
    start_from = get_node(start_from)
    print(start_from)
    builder=code.get_builder()
    #settings_dict = {
    #    'cmdline': ['-n', '16'],
    #}
    #builder.settings = Dict(dict=settings_dict)
    if 'settings' in start_from.inputs:
        settings=start_from.inputs.settings.get_dict()
    else:
        settings={}
    if resources is not None:
        resources_=resources['resources']
        queue=     resources['queue']
        wallclock= resources['wallclock']
        if 'account' in resources:
            builder.metadata.options.account = resources['account']
    else: #get resources from old calculation
        resources_,queue,wallclock, account=get_resources(start_from)
        if account is not None:
            builder.metadata.options.account = account
    if not from_scratch:
        #note that the structure will not be used (it has the restart)
        if structure is None:
            builder.structure = start_from.inputs.structure
        else:
            builder.structure = structure
        builder.parent_folder = start_from.outputs.remote_folder
    else:
        if not cg:
            raise ValueError('from_scratch={} and cg={} are incompatible'.format(from_scratch,cg))
        else:
            #copy velocities and starting positions from last step
            last_traj_struct=start_from.outputs.output_trajectory.get_step_structure(-1)
            if structure is None:
                new_structure=start_from.inputs.structure.clone()
            else:
                new_structure=structure.clone()
            new_structure.cell=last_traj_struct.cell
            new_structure.clear_sites()
            for site in last_traj_struct.sites:
                new_structure.append_site(site) 
            builder.structure=new_structure
            #set velocities
            ion_velocities=(extract_velocities_from_trajectory(start_from.outputs.output_trajectory).get_array('velocities')*qeunits.timeau_to_sec*1.0e12).tolist()
            settings['ATOMIC_VELOCITIES']=ion_velocities

    if cmdline is not None:
        settings['cmdline']=cmdline
    
    builder.pseudos = start_from.inputs.pseudos
    parameters = copy.deepcopy(start_from.inputs.parameters.get_dict())
    for itemtodel in remove_parameters_namelist:
        parameters.pop(itemtodel,None)
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
    elif not 'emass' in start_from.inputs.parameters['ELECTRONS']:
        raise ValueError('emass parameter not found in input dictionary!')
    if dt is not None:
        if not from_scratch:
            if abs(float(parameters['CONTROL']['dt']) - dt) > 1e-5 and dtchange:
                parameters['IONS']['ion_velocities'] = 'change_step'
                parameters['IONS']['tolp'] = parameters['CONTROL']['dt']
                parameters['ELECTRONS']['electron_velocities'] = 'change_step'
        parameters['CONTROL']['dt'] = dt
    if remove_autopilot:
        try:
            del settings['AUTOPILOT']
            print ('removed AUTOPILOT input')
        except:
            print ('no AUTOPILOT input to remove')
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
    if stepwalltime_s is not None:
        wallclock_max=wallclock
        wallclock=nstep*stepwalltime_s*1.25+120.0
        wallclock=wallclock if wallclock < wallclock_max else wallclock_max
        print('wallclock requested: {} s'.format(wallclock))
   
    parameters['CONTROL']['nstep'] = nstep
    parameters['CONTROL']['max_seconds'] = int(wallclock*0.9)
    print(parameters)    
    builder.parameters = Dict(parameters)
    builder.settings = Dict(settings)
    builder.metadata.options.resources = resources_
    builder.metadata.options.max_wallclock_seconds = int(wallclock)
    builder.metadata.options.queue_name = queue
    return builder




#compare forces between reference pw calculation and cp one
def compare_forces(pwcalc,minpk=0):
    #get cp trajectory (I am assuming that the graph is as I made it)
    parent_traj=get_parent_trajectory(pwcalc.get_incoming().all_nodes(),minpk)
    #get calcfunctions (one of them has the index of the step)
    parent_calcfunctions=get_parent_calcfunction(pwcalc.get_incoming().all_nodes(),minpk)
    if len(parent_traj) != 1:
        raise RuntimeError('wrong number of direct parent trajectories in the graph ({})'.format(len(parent_traj)))
    #get parent cp calculation
    parent_cp=get_parent_calculation(parent_traj,minpk)
    if len(parent_cp) != 1:
        raise RuntimeError('wrong number of direct parent calculations in the graph ({})'.format(len(parent_cp)))
    emass=parent_cp[0].inputs.parameters['ELECTRONS']['emass']
    dt=parent_cp[0].inputs.parameters['CONTROL']['dt']
    pk=parent_cp[0].pk
    extract_index_calcfunction=[]
    for parent_calcfunction in parent_calcfunctions:
        if parent_calcfunction.function_name == 'extract_structure_from_trajectory' and hasattr(parent_calcfunction.inputs,'step_index'):
            extract_index_calcfunction.append(parent_calcfunction)
    if len(extract_index_calcfunction) != 1:
        raise RuntimeError('wrong number of direct parent extract_structure_from_trajectory in the graph ({})'.format(len(extract_index_calcfunction)))
    #index of the step of the trajectory
    step_index = extract_index_calcfunction[0].inputs.step_index.value
    #get atomic masks to differentiate the various atomic types
    atomic_masks=get_atomic_types_and_masks(pwcalc.outputs.output_trajectory.get_step_data(0)[3])
    #cp forces, of the correct step
    forces_cp=parent_traj[0].get_array('forces')[step_index]
    #pw forces
    forces_pw_arr=pwcalc.outputs.output_trajectory.get_array('forces')
    if forces_pw_arr.shape[0] != 1:
        raise RuntimeError('wrong number of steps in pw trajectory')
    res=[]
    for name,mask in atomic_masks:
        res.append( (name, forces_pw_arr[0][mask], forces_cp[mask]/forces_pw_arr[0][mask]) )
    return res, emass, pk, dt

def compare_forces_many(submitted, minpk=0):
    '''
    Given a list of submitted pw jobs, find the original cp trajectory and
    get the forces and the cp/pw force ratio for comparison.
    
    In the output you get a dictionary with one entry for every emass value of the original cp simulation.
    Then you will find in nested dictionaries all data for forces and ratios divided by atomic type.
    Everything is found by traversing the aiida graph.
    '''
    atom_forces={}
    for sub in submitted:
        if sub.is_finished_ok or sub.exit_status == 322 :
            list_ratios, emass, pk, dt = compare_forces(sub,minpk)
            atom=atom_forces.setdefault(emass,{}).setdefault(dt,{'fratios':{},'forces':{}})['fratios']
            forces=atom_forces.setdefault(emass,{}).setdefault(dt,{'fratios':{},'forces':{}})['forces']
            atom_forces[emass][dt].setdefault('PK',set()).add(pk)
            for element in list_ratios:
                atom.setdefault(element[0],[]).append(element[2])
                forces.setdefault(element[0],[]).append(element[1])
        else:
            print(sub.pk,'failed')
    #print(atom_forces)
    for emass_af in atom_forces.keys():
        for dt in atom_forces[emass_af].keys():
            for element in atom_forces[emass_af][dt]['fratios'].keys():
                atom_forces[emass_af][dt]['fratios'][element]=np.array(atom_forces[emass_af][dt]['fratios'][element])
            for element in atom_forces[emass_af][dt]['forces'].keys():
                atom_forces[emass_af][dt]['forces'][element]=np.array( atom_forces[emass_af][dt]['forces'][element])
    return atom_forces


def analyze_forces_ratio(pwcalcjobs,fthreshold=0.1,corrfactor=1.0,ax_=None,minpk=0, create_fig=None):
    '''
    Given a list of pw calcjobs, analyzes the force ratios between them and their cp ancestors.
    Produces a series of histogram plots and mean and standard deviation of the ratio in an output dictionary.
    '''
    if isinstance(ax_,list) or ax_ is None:
        ax=ax_
    else:
        ax=[ax_]
    atom_forces=compare_forces_many(pwcalcjobs, minpk)
    figs=[]
    res={}
    ax_counter=0
    for emass in atom_forces.keys():
        for dt in atom_forces[emass].keys():
            if ax is not None:
                if len(ax) <= ax_counter:
                    fig__,ax__=create_fig()
                    figs.append(fig__)
                    ax.append(ax__)
                ax[ax_counter].set_title('emass={}, dt={}, PK={}'.format(emass,dt,atom_forces[emass][dt]['PK']))
            for element in atom_forces[emass][dt]['fratios'].keys():
                mask=atom_forces[emass][dt]['forces'][element]>fthreshold
                res.setdefault(emass,{}).setdefault(dt,{})[element]={
                    'forces_std': float(atom_forces[emass][dt]['forces'][element][mask].std()),
                    'fratios_mean': float(atom_forces[emass][dt]['fratios'][element][mask].mean()*corrfactor),
                    'fratios_std': float(atom_forces[emass][dt]['fratios'][element][mask].std()*corrfactor),
                    'PK': atom_forces[emass][dt]['PK'],
                    'fig_idx': ax_counter
                }
                if ax is not None:
                    hist=ax[ax_counter].hist(
                    atom_forces[emass][dt]['fratios'][element][
                        mask
                    ]*corrfactor,
                    bins=100,alpha=0.5,label='{0:s} mean={1:.4f} std={2:.4f}'.format(element,res[emass][dt][element]['fratios_mean'],res[emass][dt][element]['fratios_std'])
                      )
                    res[emass][dt][element]['hist']=hist
            if ax is not None:
                ax[ax_counter].legend()
                ax_counter += 1
    if ax is not None:
        return res, ax,figs, ax_counter
    else:
        return res

def ekinc_const_motion_analysis(traj):
    return ekinc_const_motion_analysis_2(traj.get_array('times'),
                                         traj.get_array('electronic_kinetic_energy'),
                                         traj.get_array('energy_constant_motion'),
                                         traj.numsites)

def ekinc_const_motion_analysis_2(tps, ekinc, cmot, natoms):
    """Convert electronic kinetic energy in K and then does a fit.
    Returns angular coefficient in K/ps.
    This is the true quantity that you are interested in, since this
    is the temperature that the ionic system is going to lose because of 
    the coupling between electrons and ions
    Then does the same thing for the constant of motion.
    With those two numbers you are able to decide if your timestep
    and your emass are good or not.
    """
    #conversion factor from eV to K
    k_b=8.617333262145e-5 #eV/K
    eV_to_K=2/(3*k_b*natoms)
    
    fit_ekinc=np.polyfit(tps,ekinc*eV_to_K,1)
    fit_cmot=np.polyfit(tps,cmot*eV_to_K,1)
    return fit_ekinc, fit_cmot

#various calcfunctions

@calcfunction
def extract_structure_from_trajectory(traj, step_index=lambda: Int(-1)):
    if int(step_index) < traj.numsteps and int(step_index) >= -traj.numsteps:
        return traj.get_step_structure(int(step_index))
    else:
        raise ValueError('index {} out of range for trajectory {}'.format(step_index, traj))

@calcfunction
def extract_velocities_from_trajectory(traj, step_index=lambda: Int(-1)):
    if not 'velocities' in traj.get_arraynames():
        raise ValueError(' array velocities not found in trajectory {}'.format(traj))
    if abs(int(step_index)) < traj.numsteps:
        return to_ArrayData(traj.get_array('velocities')[int(step_index)],'velocities')
    else:
        raise ValueError('index {} out of range for trajectory {}'.format(step_index, traj))


@calcfunction
def get_maximum_frequency_vdos(traj):
    vel=traj.get_array('velocities')
    times=traj.get_array('times')
    vdos=np.abs(np.fft.fft(vel,axis=0)).mean(axis=1).mean(axis=1)
    idx=np.argmax(vdos)
    return Float(np.fft.fftfreq(vel.shape[0],times[-1]-times[-2])[idx])


def compare_pop(k1,k2,exclude_list):
    for k in list(k1.keys()):
        if k in exclude_list:
            k1.pop(k)
            k2.pop(k)
            continue
        if k1.pop(k) != k2.pop(k):
            return False
    if not k1 and not k2:
        return True
    else:
        raise KeyError('missing keys to compare!')

def are_same_kind(k1_,k2_):
    k1=copy.deepcopy(k1_)
    k2=copy.deepcopy(k2_)
    try:
       return compare_pop(k1,k2,['name']) 
    except KeyError:
        return False

@calcfunction
def collapse_kinds(structure):
    attr=structure.get_attribute('kinds')
    sites=structure.get_attribute('sites')

    #find uniques kind and put them in output_attr:
    kind_translation={}
    output_attr=[]
    for a in attr:
        new=True
        kind_name=a['name']
        for kind in output_attr:
            if are_same_kind(a,kind):
                new=False
                kind_name=kind['name']
                break
        nodigit_kind_name=''.join(i for i in kind_name if not i.isdigit())
        if new:
            output_attr.append(a) 
        kind_translation[a['name']]=nodigit_kind_name
        output_attr[-1]['name']=nodigit_kind_name
    #translate old kind names
    for site in sites:
        site['kind_name']=kind_translation[site['kind_name']]

    if not structure.is_stored:
        s=structure
    else:
        s=structure.clone()
    s.set_attribute('kinds',output_attr)
    s.set_attribute('sites',sites)
    return s

def get_structures_from_trajectory(traj,every=1,stop=-1):
    structures=[]
    for i in range(0,traj.numsteps if stop<0 else stop ,every):
        structures.append(extract_structure_from_trajectory(traj,Int(i)))
    return structures



def generate_pw_from_trajectory(pwcode, start_from,
                                    skip=1, numcalc=0,
                                    resources=None,
                                    traj=None,
                                    nbnd=None,
                                    pseudos = None,
                                    ecutwfc = None,
                                    stop=-1
                                ):
    if start_from is not None:
       start_from = get_node(start_from)
    if pseudos is None:
        pseudos = start_from.inputs.pseudos
    if traj is None:
        traj=start_from.outputs.output_trajectory
    if numcalc==0 and skip>0:
        pass
    elif numcalc>0 and skip==1:
        skip=int(traj.numsteps/numcalc)
        if skip==0:
            skip=1
    else:
        raise KeyError('cannot specify both skip and numcalc')
    structures=get_structures_from_trajectory(traj,skip,stop)
    builders=[]
    for structure in structures:
        builder=pwcode.get_builder()
        builder.structure = structure
        #settings_dict = {
        #    'cmdline': ['-n', '16'],
        #}
        #builder.settings = Dict(dict=settings_dict)
        #The functions finds the pseudo with the elements found in the structure.
        builder.pseudos = pseudos
        KpointsData = DataFactory('core.array.kpoints')
        kpoints=KpointsData()
        kpoints.set_kpoints_mesh([1,1,1])
        builder.kpoints = kpoints
        parameters = {
            
            'CONTROL' : {
                'calculation': 'scf' ,
                'restart_mode' : 'from_scratch',
                'tstress': True,
                'tprnfor': True,
               # 'disk_io': 'none',
            },
            'SYSTEM' : {
                'ecutwfc': start_from.inputs.parameters.get_dict()['SYSTEM']['ecutwfc'] if ecutwfc is None else ecutwfc,
            },
            'ELECTRONS' : {
            },
        }
        if nbnd is not None:
            parameters['SYSTEM']['nbnd'] = nbnd
        builder.settings = Dict({'gamma_only': True})
        builder.parameters = Dict(parameters)
        if resources is not None:
            builder.metadata.options.resources = resources['resources']
            builder.metadata.options.max_wallclock_seconds = resources['wallclock']
            builder.metadata.options.queue_name = resources['queue']
            if 'account' in resources:
                builder.metadata.options.account=resources['account']
        else:
            builder.metadata.options.resources,builder.metadata.options.queue_name,builder.metadata.options.max_wallclock_seconds,account=get_resources(start_from)
            if account is not None:
                builder.metadata.options.account=account
        
        builders.append(builder)
    return builders

def get_emass_dt_pk(calc):
    return calc.inputs.parameters['ELECTRONS']['emass'],calc.inputs.parameters['CONTROL']['dt'],calc.pk


def get_calc_from_emass_dt(res,emass,dt):
    candidate= get_node(max(res[emass][dt][dict_keys(res,level=2)[0]]['PK']))
    if not candidate.is_finished_ok:
        pk_max=-1
        for pk in res[emass][dt][dict_keys(res,level=2)[0]]['PK']:
            if get_node(pk).is_finished_ok and pk>pk_max:
                pk_max=pk
        if pk_max>=0:
            candidate=get_node(pk_max)
    return candidate

def get_parent_calc_from_emass_dt(res,emass,dt,minpk=0):
    startfrom=get_parent_calculation(get_calc_from_emass_dt(res,emass,dt).get_incoming().all_nodes(),minpk)
    if len(startfrom) != 1:
        raise RuntimeError('Bug: wrong logic')
    return startfrom

def load_nodes(pks):
    res=[] 
    for pk in pks:
        res.append(aiida.orm.load_node(pk))
    return res

def get_total_time(cps):
    if len(cps)==0:
        return 0.0
    tot=0.0
    for cp in cps:
        if 'output_trajectory' in cp.outputs:
            tot = tot + cp.outputs.output_trajectory.get_array('times')[-1] - cp.outputs.output_trajectory.get_array('times')[0]
    return tot
    


@calcfunction
def set_mass(new_mass,structure):
    new_structure=structure.clone()
    new_kinds=copy.deepcopy(structure.get_attribute('kinds'))
    for kind in new_kinds:
        if True:
            kind['mass']=new_mass[kind['name']]*kind['mass']
        else:
            kind['mass']=new_mass[kind['name']]
    new_structure.set_attribute('kinds',new_kinds)
    return new_structure

@calcfunction
def set_kinds(new_kinds,structure):
    new_structure=structure.clone()
    new_structure.set_attribute('kinds',list(new_kinds))
    return new_structure



def to_ArrayData(a,key):
    res=ArrayData()
    res.set_array(key,a)
    return res


from .generate_concatenate import *

def factors(nr):
    i = 2
    factor = []
    while i <= nr:
        if (nr % i) == 0:
            factor.append(i)
            nr = nr / i
        else:
            i = i + 1
    return factor

def unique(array): return [x for i, x in enumerate(array) if array.index(x) == i]

def apply_new_factor(fnode,inp):
    res2=[]
    for used_idx,res in inp:
        for r in res:
            for i,f in enumerate(fnode):
                if i in used_idx: continue
                res2=res2+[(used_idx+[i],[[r[0]*f,r[1]],[r[0],r[1]*f]])]
    return res2

def apply_many_new_factor(fnode,n=1,start=[[1,1]]):
    res=[([],start)]
    t=res
    for i in range(n):
        res=apply_new_factor(fnode,res)
        if len(res)==0: break
        t=t+res
    return t

def join_second(t):
    res=[]
    for _,r in t:
        res=res+r
    return res

def possible_ntg_nb(nnodes,procpernode):
    #do a prime factor decompsition of nnodes and procpernode
    #use the lowest factors to generate some ntg and nb, that takes factors in various way
    fnode=[1]+factors(nnodes)
    fproc=[1]+factors(procpernode)
    #generate some possible configurations
    #use only node factors
    c1=join_second(apply_many_new_factor(fnode,5))
    #use all nodes in band groups and factors from number of processors everywhere
    c2=join_second(apply_many_new_factor(fproc,start=[[nnodes,1]]))
    #use everything only in task groups
    fprocs=[]
    for i,f1 in enumerate(fproc):
        for f2 in fproc[:i]:
            fprocs.append(f1*f2)
    c3=[ [1,nnodes*t] for t in fprocs ] 
    #remove duplicated
    resu=unique(c1+c2+c3)
    resu.sort(key=lambda x: x[0]*x[1])

    return resu

def main_loop_line(c):
    s=c.outputs.retrieved.get_object_content('aiida.out')
    idx=s.find('main_loop')
    a=s[idx:].split('\n')[0].split()
    return [float(a[4][:-1]), int(a[7])]


def fake_function(*args,**kwargs):
    print(args,kwargs)
    return




class CpWorkChain(WorkChain):
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input('structure', required=True, valid_type=(aiida.orm.nodes.data.StructureData,aiida.orm.nodes.data.TrajectoryData),validator=validate_structure,
                   help='Input structure. If a trajectory is given, the workchain will use its last step to start the CG. If velocities are present, they will be used to initialize the simulation. Note that if you use a trajectory, usually kind information (like mass) are not included, so default values will be used. If you want to include kind information or override those provided with the input structure, use the input structure_kinds')
        spec.input('structure_kinds',valid_type=(List),required=False,
                   help='These kinds will be used to override or set the masses of the various atomic types. Note that the workflow, if skip_emass_dt_test is True, will calculate the ratio between cp forces and pw forces and adjust the provided masses automatically according to this ratio. So if you provide this input, make sure to set skip_emass_dt_test to True and set also the inputs emass and dt, or "bad things can happen"')
        spec.input('pseudo_family', required=True, valid_type=(Str),validator=validate_pseudo_family,
                   help='pseudopotential family to use, as in usual aiida operations')
        spec.input('ecutwfc', required=True, valid_type=(Float),validator=validate_ecutwfc,
                   help='wavefunction cutoff (Ry), like in the QE input')
        spec.input('pw_code', required=True, valid_type=(aiida.orm.nodes.data.code.Code),validator=validate_pw_list,
                  help='input pw code (used to calculate force ratio)')
        spec.input('cp_code',required=True, valid_type=(aiida.orm.nodes.data.code.Code), validator=validate_cp_list)
        spec.input('pw_resources_list',valid_type=(List),required=True,help='Same as cp_resources_cp_list but for pw.x code.')
        spec.input('cp_resources_cp_list',valid_type=(List),required=True,
                   help="""List of dictionary like the following:
{
 'resources' : {
   'num_machines' : 2,
   'num_mpiprocs_per_machine' : 48,
 },
 'wallclock' : 3600,
 'queue' : 'queue_name',
 'account': 'account_name',
}
c,porturrently only the first element of the list is used.
'wallclock' is the maximum time that can be requested to the scheduler. This code can decide to ask for less.
""")
        spec.input('cp_resources_cg_list',valid_type=(List),required=True,help='Same as cp_resources_cp_list but when doing a CG. The CG uses a different amount of resource and can use no band or task group parallelization.')
        spec.input('target_force_ratio', valid_type=(Float), default=lambda: Float(0.9), validator= lambda a,port: 'target_force_ratio must be between 0 and 1' if a>=1.0 or a<=0.0 else None, help='The forces calculated by the Car-Parrinello method are affected by two types of error: one is due to the oscillations of the electrons around the DFT energy minimum, and the second is due to the finite mass of the electronic fluid that produces a _sistematic_ error in the forces, as if the electrons add mass to the ionic core. This second kind of error is can be controlled by this parameter, that tries to adjust the electronic mass to obtain the desidered ratio between CP forces and true DFT forces. Then you may want to modify the ionic mass to correct the leading factor of this error.' )
        spec.input('additional_parameters_cp', valid_type=(Dict),default=lambda: Dict(dict={}),
                   help='parameters that will be included in the settings input of the QE CP plugin. These settings will be added on top of the default one. Same format as plugin input')
        spec.input('dt_start_stop_step', valid_type=(List), default=lambda: List(list=[2.0,4.0,20.0]),help='list of timesteps to try. Timesteps are changed to better integrate the equation of motion. When a new electronic mass is selected by this workchain timesteps are automatically adjusted.')
        spec.input('emass_list', valid_type=(List), default=lambda: List(list=[50.0, 75.0, 100.0]),help='list of electronic masses to try. The emass is selected in order to satisfy the requested CP/DFT force ratio.')
        spec.input('number_of_pw_per_trajectory', valid_type=(Int), default=lambda: Int(100),
                   help='Number of pw submitted for every trajectory during calculation of force ratio.')
        spec.input('skip_emass_dt_test',valid_type=(Bool), default=lambda: Bool(False))
        spec.input('adjust_ionic_mass', valid_type=(Bool), default=lambda: Bool(True), help='Multiply the mass of the ions by the corresponding force ration between the cp forces and pw forces -- that is less than 1. Note that averages of static properties do not depend on the ionic masses.')
        spec.input('skip_thermobarostat',valid_type=(Bool),  default=lambda: Bool(False))
        spec.input('nve_required_picoseconds',valid_type=(Float), default=lambda: Float(50.0),
                  help='The equilibrated NVE simulation will last at least this number of picoseconds. How much picoseconds do you want?')
        spec.input('tempw_initial_random',valid_type=(Float), required=False, help='If provided, sets the initial temperature when randomly initializing the starting velocities.')
        spec.input('thermobarostat_points',valid_type=(List),help='List of dicts, each with the format [ { "temperature_K": 1000, "pressure_KBar": 10 , "equilibration_time_ps": 5.0, "thermostat_time_ps": 5.0} ]. The simulation will loop over this list of dictionaries, in the same order, equilibrating for the specified time at the given P,T point. Every point is repeated if the average T and P are not within the specified ranges')
        spec.input('nstep_initial_cg',valid_type=(Int), default=lambda: Int(50), help='At the beginning of the simulation the CP algorithm is not used. This is the number of steps to do using Born-Oppenheimer molecular dynamics algorithm with a conjugate gradient minimization of the electronic ground state.')
        spec.input('initial_atomic_velocities_A_ps',valid_type=(ArrayData),required=False,help='optional input initial velocities in angstrom over picoseconds')
        spec.input('dt',valid_type=(Float),required=False,help='timestep in atomic units, if not automatically chosen.')
        spec.input('emass',valid_type=(Float),required=False,help='electronic mass, atomic mass units, if not automatically chosen')
        spec.input('cmdline_cp',valid_type=(List), required=False,help='additional command line parameters of the cp verlet caclulations only (for example parallelization options)')
        spec.input('skip_parallel_test',valid_type=(Bool),default=lambda: Bool(False), help='do not run run benchmarks to discover a good internal Quantum Espresso parallelization scheme for the current system')
        spec.input('nstep_parallel_test',valid_type=(Int), default=lambda: Int(200),help='the benchmark simulations will be that long, if performed')
        spec.input('benchmark_parallel_walltime_s',valid_type=(Float), default=lambda: Float(600.0),help='time requested to the scheduler during the test for finding the best parallelization parameters.')
        spec.input('benchmark_emass_dt_walltime_s',valid_type=(Float), default=lambda: Float(1200.0),help='same as benchmark_parallel_walltime_s but for dermining the best electronic mass and timestep.')
        spec.input('max_slope_ekinc',valid_type=(Float), default=lambda: Float(0.1),help='max slope in K/ps of the ekinc linear fit. If not satisfied try to change emass')
        spec.input('max_slope_const',valid_type=(Float), default=lambda: Float(0.05),help='max slope in K/ps of the constant of motion linear fit.')
        spec.input('max_slope_min_ps',valid_type=(Float), default=lambda: Float(1.0),help='minimum required lenght in ps of the last trajectory to do the linear fit on ekinc and const of motion')
        spec.input('max_slope_min_emass',valid_type=(Float), default=lambda: Float(50.0),help='minimum possible value of electronic mass that can be set by the max_slope correction routine. Will not go lower than that.')
        spec.input('min_traj_steps_vdos', valid_type=(Int), default=lambda: Int(100), help='minimum number of steps to consider the calculated vibrational spectrum maximum valid, to set the thermostat frequency')
        spec.input('default_nose_frequency', valid_type=(Float), default=lambda: Float(10.0), help='default nose frequency when a frequency cannot be estimated from the vibrational spectrum')
        spec.input('minimum_nose_frequency', valid_type=(Float), default=lambda: Float(0.1), help='minimum nose frequency: if the frequency estimated from the vibrational spectrum is lower than this value, this value is used')
        spec.input('temperature_tolerance', valid_type=(Float), default=lambda: Float(-1), help='Temperature tolerance in K used to say if the npt is equilibrated. If not setted, use the standard deviation of the T time series')
        spec.input('pressure_tolerance', valid_type=(Float), default=lambda: Float(-1), help='Pressure tolerance in kBar used to say if the npt is equilibrated. If not setted, use the standard deviation of the P time series')
       

        spec.outline(
            cls.setup,
            if_(cls.find_emass_dt)(
                cls.small_equilibration_cg,
                cls.emass_benchmark,
                cls.dt_benchmark,
                while_(cls.emass_dt_not_ok)(
                    cls.compare_forces_pw,
                    cls.analysis_step
                )
            ).else_(
                #start with some cg steps, then go on 
                cls.small_equilibration_cg
                #do some nve equilibration?
            ),
            cls.setup_check1, #everything will start from ctx.check1 
            #find best parallelization options
            cls.benchmark_parallelization_options, # sets, if test was performed, NEW MASSES for ions
            cls.benchmark_analysis, #overwrite ctx.check1 with the faster simulation
            #thermobarostatation
            #prepare ctx.last_nve
            cls.nose_setup, #setup ctx.last_nve
            while_(cls.equil_not_ok)(
                cls.nose_prepare, #sets nose counters
                while_(cls.check_nose)(
                    cls.nose # append to ctx.last_nve
                ),
                cls.final_cg,# append to ctx.last_nve
                cls.check_final_cg,
                while_(cls.check_nve_nose)(
                    cls.run_nve, # append to ctx.last_nve; eventually SET NEW MASSES of ion if self.ctx.find_new_ion_masses is True
                    # check slope of ekinc and econt, correct and eventually do a new final_cg
                    while_(cls.prepare_check_slope)(
                       cls.nothing
                    ),
                    cls.check_slope,
                    if_(cls.check_slope_not_ok)(
                        #change emass and dt
                        cls.final_cg,
                        cls.prepare_find_new_masses, # do a small cp
                        cls.compare_forces_pw, #takes trajectory from self.ctx.dt_benchmark!
                        cls.find_new_ion_masses #find new force ratios; ratio are in self.ctx.ionic_mass_corr
                    )
                )
            ),
            cls.setup_check2, #everything will start from ctx.last_nve[-1]
            #first production nve simulation index is self.ctx.first_prod_nve_idx
            #in array self.ctx.last_nve
            cls.final_cg,
            cls.check_final_cg,
            while_(cls.run_more)(
                cls.run_nve
            ),
            cls.get_result
        )
        spec.exit_code(401,'ERROR_INITIAL_CG_FAILED', message='The initial cg steps failed. I cannot start to work.')
        spec.exit_code(402, 'ERROR_NOSE_FAILED', message='Nose-Hoover thermostat failed.')
        spec.exit_code(403, 'ERROR_FINAL_CG_FAILED', message='Final cg after Nose-Hoover failed.')
        spec.exit_code(404, 'ERROR_NVE_FAILED', message='Error in the NVE simulation')
        spec.exit_code(405, 'ERROR_GARBAGE', message='The simulations are calculating very expensive random numbers. There is something wrong (cutoff? metal? boo?)')
        spec.exit_code(406, 'ERROR_WRONG_INPUT', message='Wrong input parameters')
        spec.exit_code(407, 'ERROR_PARALLEL_TEST', message='Parallel test was not succesful, maybe there is something more wrong.')
        spec.exit_code(408, 'ERROR_MULTIPLE_FAIL', message='Multiple errors in the simulation that cannot fix.')
        spec.exit_code(409, 'ERROR_WRONG_LOGIC', message='This is a bug in the workchain.')
        spec.output('nve_prod_traj')
        spec.output('full_traj')
        spec.output('dt')
        spec.output('emass')
        spec.output('cmdline_cp')
        spec.output('kinds')

    def nothing(self):
        pass

    def emass_dt_not_ok(self):
        return bool(self.ctx.emass_dt_not_ok)
 
    def set_cp_code(self, index):
        self.ctx.current_cp_code=self.inputs.cp_code
        self.ctx.current_cp_code_cp_resources=self.inputs.cp_resources_cp_list[index]
        self.ctx.current_cp_code_cg_resources=self.inputs.cp_resources_cg_list[index]

    def get_cp_code(self):
        return self.ctx.current_cp_code

    def get_cp_resources_cg(self):
        return self.ctx.current_cp_code_cg_resources

    def get_cp_resources_cp(self):
        return self.ctx.current_cp_code_cp_resources

    def set_pw_code(self, index):
        self.ctx.current_pw_code=self.inputs.pw_code
        self.ctx.current_pw_code_resources=self.inputs.pw_resources_list[index]

    def get_pw_code(self):
        return self.ctx.current_pw_code

    def def_pw_resources(self):
        return self.ctx.current_pw_code_resources

    def setup(self):
        #extract structure from last step of the trajectory, if necessary
        if isinstance(self.inputs.structure,aiida.orm.nodes.data.TrajectoryData ):
            self.ctx.start_structure = extract_structure_from_trajectory(self.inputs.structure)
            self.report('setting starting positions from last step of input trajectory')
            #get also velocities, if present, from the trajectory
            try:
                self.ctx.start_velocities_A_au = (extract_velocities_from_trajectory(self.inputs.structure).get_array('velocities')*qeunits.timeau_to_sec*1.0e12).tolist()
                self.report('setting starting velocities from last step of input trajectory')
            except ValueError:
                pass
        else:
            if 'inital_atomic_velocities_A_ps' in self.inputs:
                self.report('setting starting velocities from input')
                self.ctx.start_velocities_A_au = (self.inputs.initial_atomic_velocities_A_ps.get_array('velocities')*qeunits.timeau_to_sec*1.0e12).tolist()
            self.ctx.start_structure = self.inputs.structure
        self.ctx.start_structure = collapse_kinds(self.ctx.start_structure)
        if 'structure_kinds' in self.inputs:
            self.report('SETTING KINDS as in {}'.format(self.inputs.structure_kinds))
            self.ctx.start_structure = set_kinds(self.inputs.structure_kinds,self.ctx.start_structure)
            if not self.inputs.skip_emass_dt_test.value and self.inputs.adjust_ionic_mass:
                self.report('!! WARNING !! skip_emass_dt_test is False. The masses of atoms will be adjusted, by a rescaling of the current ones.')
        else:
            self.report('using default kinds')
            if self.inputs.skip_emass_dt_test.value:
                self.report('!! WARNING !! you are not testing against emass and you are not providing structure_kinds input: inertia of the atoms may be wrong')
        self.ctx.fnosep=10.0
        #set default codes 
        self.set_pw_code(0)
        self.set_cp_code(0)
        self.ctx.emass_dt_not_ok=True
        self.ctx.nve_count=0 
        if 'cmdline_cp' in self.inputs:
            self.ctx.cmdline_cp=self.inputs.cmdline_cp.get_list()
        else:
            self.ctx.cmdline_cp=['-ntg','1','-nb','1']
        if self.inputs.skip_emass_dt_test.value:
            if not 'dt' in self.inputs and not 'emass' in self.inputs:
                self.report('[setup] ERROR: if you want to skip the emass/dt benchmark, you must provide values for dt and emass!')
                return 406 
            self.out('dt', self.inputs.dt)
            self.out('emass',self.inputs.emass)
        else:
            #check that there are some values in the provided range
            if len(self.inputs.emass_list) <= 0:
                return 406 
            if len(np.arange(*self.inputs.dt_start_stop_step)) <= 0:
                return 406
        self.ctx.retry_count=0
        self.report('setup completed')

    def find_emass_dt(self):
        return not self.inputs.skip_emass_dt_test.value

    def small_equilibration_cg(self):
        #This is always the first step
        #build cg input
        tempw_initial = self.inputs.thermobarostat_points[0]['temperature_K']
        builder = configure_cp_builder_cg(
                                self.get_cp_code(),
                                self.inputs.pseudo_family,
                                self.ctx.start_structure,
                                self.inputs.ecutwfc,
                                self.inputs.tempw_initial_random if 'tempw_initial_random' in self.inputs else tempw_initial,
                                self.get_cp_resources_cg (),
                                additional_parameters=self.inputs.additional_parameters_cp.get_dict(),
                                ion_velocities = self.ctx.start_velocities_A_au if 'start_velocities_A_au' in self.ctx else None,
                                nstep=self.inputs.nstep_initial_cg.value,
                                print= lambda x : self.report('[small_equilibration_cg] [builder] {}'.format(x))
                               )
        
        node = self.submit(builder)
        self.to_context(initial_cg=node)
        
        self.report('[small_equilibration] pk={} cg sent to context'.format(node.pk))
        return

    def emass_benchmark(self):
        #check that initial cg is ok

        if not self.ctx.initial_cg.is_finished_ok:
            return 401

        #...

        #prepare multiple
        params=copy.deepcopy(self.get_cp_resources_cp())
        params['wallclock']=self.inputs.benchmark_emass_dt_walltime_s.value 
        #start,stop,step,fac=self.inputs.emass_list
        for mu in self.inputs.emass_list:
            calc=configure_cp_builder_restart(
                                               self.get_cp_code(),
                                               self.ctx.initial_cg,
                                               mu=mu,
                                               dt=float(np.arange(*self.inputs.dt_start_stop_step)[0]), 
                                               resources=params,
                                               cmdline=self.ctx.cmdline_cp,
                                               print= lambda x : self.report('[emass_benchmark] [builder] {}'.format(x))
                                             )
            sub=self.submit(calc)
            self.to_context(emass_benchmark=append_(sub))
            self.report('[emass_benchmark] pk={} emass={} sent to context'.format(sub.pk,mu))
           
        return

    def dt_benchmark(self):
        params=copy.deepcopy(self.get_cp_resources_cp())
        params['wallclock']=self.inputs.benchmark_emass_dt_walltime_s.value 
        for calc in self.ctx.emass_benchmark:
            mu,dt,pk=get_emass_dt_pk( calc) 
            if calc.is_finished_ok:
                for dt in np.arange(*self.inputs.dt_start_stop_step):
                    self.report('[dt_benchmark] trying calculation with emass={} dt={}'.format(mu,dt))
                    newcalc=configure_cp_builder_restart(
                                                   self.get_cp_code(),
                                                   calc,
                                                   resources=params,
                                                   copy_mu_mucut=True,
                                                   dt=dt,
                                                   cmdline=self.ctx.cmdline_cp,
                                                   print= lambda x : self.report('[dt_benchmark] [builder] {}'.format(x))
                                                   )
                    sub=self.submit(newcalc)
                    self.to_context(dt_benchmark=append_(sub))
                    self.report('[dt_benchmark] submitted pk={} emass={} dt={}'.format(sub.pk,mu,dt))
            else:
                self.report('[dt_benchmark] calculation with pk={} emass={} dt={} failed'.format(pk,mu,dt))

        return

    def compare_forces_pw(self):
        pwcode=self.get_pw_code()
        joblist=[]
        for calc in self.ctx.dt_benchmark:
            emass,dt,pk=get_emass_dt_pk(calc)
            if calc.is_finished_ok:
                children_calc=get_children_calculation(calc.get_outgoing().all_nodes())
                if len(children_calc)>0:  #maybe I calculated this stuff before
                    pwcont=0
                    for cc in children_calc:
                        if cc.process_type == 'aiida.calculations:quantumespresso.pw':
                            pwcont = pwcont + 1
                    if pwcont > 0:
                        continue
                self.report('[compare_forces_pw] comparing forces with PW for calc with pk={} emass={} dt={}'.format(pk,emass,dt))
                joblist=joblist+generate_pw_from_trajectory(
                                        pwcode,
                                        calc,
                                        numcalc=int(self.inputs.number_of_pw_per_trajectory)
                                    )
            else:
                self.report('[compare_forces_pw] calculation with pk={} emass={} dt={} failed'.format(pk,emass,dt))
        if len(joblist)==0:
            raise RuntimeError('What am I doing here?')
        for job in joblist:
            self.to_context(compare_pw=append_(self.submit(job)))




    def test_analysis_step(self, test_dt_traj_pks, test_pw_calc_pks, target_force_ratio,cp_code,pw_code,cp_code_res,pw_code_res ):
        #setup fake environment
        set_fake_something(self,'ctx','dt_benchmark',load_nodes(test_dt_traj_pks))
        set_fake_something(self,'ctx','compare_pw',load_nodes(test_pw_calc_pks))
        set_fake_something(self,'ctx','current_cp_code',cp_code)
        set_fake_something(self,'ctx','current_pw_code',pw_code)
        set_fake_something(self,'ctx','current_cp_code_cp_resources',cp_code_res)
        set_fake_something(self,'ctx','current_pw_code_resources',pw_code_res)
        set_fake_something(self,'ctx','compare_pw',load_nodes(test_pw_calc_pks))
        set_fake_something(self,'inputs', 'target_force_ratio', target_force_ratio)
        set_fake_something(self,'report',None,lambda s,a: print(a))
        set_fake_something(self,'to_context',None,fake_function) 
        set_fake_something(self,'submit',None,fake_function) 
        global append_
        append_ = lambda a: a
        print('starting fake ctx state')
        print('starting tests:')
        res = self.analysis_step()
        print('fake ctx state (remember that to_context does nothing):')
        print (self.ctx.__dict__)
        print ('return value from analysis_step: ',res)

    def prepare_check_slope(self):
        res=self.fix_last_nve(report=lambda x : self.report('[prepare_check_slope] {}'.format(x)))
        if res == 1:       
            return True
        self.ctx.check_slope_simulation=self.ctx.last_nve[-1]
        return False

    def test_check_slope_prepare(self, newctx):
        class CTX:
            def __init__(self,name=''):
               self.attrib_called=[]
               self.name=''
            def __getattribute__(self, name):
               if name=='attrib_called':
                  return super().__getattribute__(name)
               if name=='__str__':
                  res=f'{name} {[c.__str__() for c in self.attrib_called]}'
               newctx=CTX(name)
               self.attrib_called.append(newctx)
               return newctx

             
        self.ctx=CTX('ctx')

    def check_slope(self):
        self.report('[check_slope] beginning')
        self.ctx.max_slope_ok=True
        if 'check_slope_simulation' in self.ctx:
           sim=self.ctx.check_slope_simulation
           if not sim.is_finished_ok:
               self.report('[check_slope] simulation to check is not finished ok!')
               return 404
           self.ctx.max_slope_emass=sim.inputs.parameters['ELECTRONS']['emass']
           self.ctx.max_slope_dt=sim.inputs.parameters['CONTROL']['dt']
           self.ctx.max_slope_emass_cut=sim.inputs.parameters['ELECTRONS']['emass_cutoff']

           simulation_check_list = self.ctx.last_nve[self.ctx.after_last_cg_idx:]
           tps, ekinc, cmot = get_concat_arrays(['times','electronic_kinetic_energy','energy_constant_motion'], simulation_check_list) 
           natoms = self.ctx.check_slope_simulation.outputs.output_trajectory.numsites
           ek, cm = ekinc_const_motion_analysis_2(tps, ekinc, cmot, natoms)
           self.report('[check_slope] pk={} (ekinc, econs) linear fit = ({}, {})'.format(get_pk_list(simulation_check_list),ek, cm))
           tps=get_total_time(simulation_check_list)
           if tps<float(self.inputs.max_slope_min_ps):
               self.report('[check_slope] the simulation is not long enough: not considering the result of the linear fit')
           else:
               if ek[0] > float(self.inputs.max_slope_ekinc):
                   #try to decrease ekinc slope by decreasing emass
                   fac=2.0/3.0
                   new_emass=self.ctx.max_slope_emass*fac
                   if new_emass<float(self.inputs.max_slope_min_emass):
                       self.report('[max_slope] I will not go under the required minimum electronic mass specified in the input.')
                       if self.ctx.max_slope_emass > float(self.inputs.max_slope_min_emass):
                           fac=float(self.inputs.max_slope_min_emass)/self.ctx.max_slope_emass
                           self.report('[max_slope] setting minimum allowed electronic mass')
                       else:
                           return
                   self.ctx.max_slope_ok=False    
                   self.ctx.max_slope_emass=self.ctx.max_slope_emass*fac
                   self.ctx.max_slope_dt=self.ctx.max_slope_dt*(fac)**0.5
                   self.report('[check_slope] ekinc too steepy: correcting emass and dt to {} {}'.format(self.ctx.max_slope_emass,self.ctx.max_slope_dt))
                   self.ctx.dt_emass_off=(self.ctx.max_slope_dt,self.ctx.max_slope_emass,0)
                   return
               if abs(cm[0]) > float (self.inputs.max_slope_const):
                   self.ctx.max_slope_ok=False 
                   self.ctx.max_slope_dt=self.ctx.max_slope_dt*2.0/3.0
                   self.report('[check_slope] econs too steepy: correcting dt to {}'.format(self.ctx.max_slope_dt))
                   return
               self.report('[check_slope] slope ok: (ekinc, econs) = ({}, {})'.format(ek,cm))
           del(self.ctx.check_slope_simulation)
        else:
           self.report('[check_slope] nothing to do')

    def check_slope_not_ok(self):
        return not self.ctx.max_slope_ok

    def prepare_find_new_masses(self):
        params=copy.deepcopy(self.get_cp_resources_cp())
        params['wallclock']=self.inputs.benchmark_emass_dt_walltime_s.value 
        
        newcalc=configure_cp_builder_restart(
                   self.get_cp_code(),
                   self.ctx.last_nve[-1],
                   resources=params,
                   copy_mu_mucut=True,
                   cmdline=self.ctx.cmdline_cp,
                   print= lambda x : self.report('[prepare_find_new_masses] [builder] {}'.format(x))
            )
        submitted=self.submit(newcalc)
        self.report('[prepare_find_new_masses] submitted pk={}'.format(submitted.pk))
        self.to_context(last_nve=append_(submitted))
        self.ctx.compare_pw=[]
        self.ctx.min_pk=submitted.pk-1
        self.ctx.dt_benchmark=[submitted] # so compare_forces_pw can start from there
        
    def find_new_ion_masses(self):
        self.ctx.find_new_ion_masses=True
        res_1 = analyze_forces_ratio(self.ctx.compare_pw, minpk=self.ctx.min_pk)#=self.node.pk)
        #there should be 1 emass and one dt. check it
        if len(res_1.keys()) != 1:
            return 409
        emass_=list(res_1.keys())[0] 
        res_2=res_1[emass_]
        if len(res_2.keys()) != 1:
            return 409
        dt_=list(res_2.keys())[0] 
        params=res_2[dt_]
        dt_node=Float(dt_)
        dt_node.store()
        emass_node=Float(emass_)
        emass_node.store()
        self.ctx.dt=dt_node
        self.ctx.emass=emass_node
        #generate dictionary for ionic mass correction
        new_mass={}
        for p in params.keys():
            new_mass[p]=params[p]['fratios_mean']
        self.ctx.ionic_mass_corr=new_mass
        self.report('[find_new_ion_masses] mass_corrections={}'.format(new_mass))
         

    def analysis_step(self):
        self.report('[analysis_step] beginnig')
        #calculate vibrational spectra to have nice nose frequencies. Simply pick the frequency of the maximum
        vdos_maxs={}
        for calc in self.ctx.dt_benchmark:
            if calc.is_finished_ok:
                vdos_v = self.inputs.default_nose_frequency
                if hasattr(calc.outputs, 'output_trajectory'):
                    if calc.outputs.output_trajectory.get_array('times').shape[0] >= self.inputs.min_traj_steps_vdos.value:
                        vdos_v=get_maximum_frequency_vdos(calc.outputs.output_trajectory)
                        if abs(vdos_v.value) < abs(self.inputs.minimum_nose_frequency.value):
                            vdos_v = self.inputs.minimum_nose_frequency
                vdos_maxs[calc.pk]=vdos_v
        self.ctx.vdos_maxs=vdos_maxs
        self.report('[analysis_step] maximum of vibrational spectra: {}'.format(vdos_maxs))
        #analyze forces ratio
        self.report('[analysis_step] comparing {} pw...'.format(len(self.ctx.compare_pw)))
        res_1 = analyze_forces_ratio(self.ctx.compare_pw, minpk=self.node.pk)
        target_force_ratio=float(self.inputs.target_force_ratio)
        
        fratio_threshold=0.05

        class RatioGoodness(Enum):
            GARBAGE = -2
            TOO_SMALL = -1
            OK = 0
            TOO_BIG = 1

        def ratios_are_ok(emass,dt,res):
            diff=[]
            val=[]
            for atom in dict_keys(res,level=2):
                val.append(res[emass][dt][atom]['fratios_mean'] - res[emass][dt][atom]['fratios_std'])
                diff.append(res[emass][dt][atom]['fratios_mean'] - target_force_ratio)
            diff=np.array(diff)
            diff_min=float(diff.min())
            val=np.array(val)
            #if values are too near zero, this is garbage
            if min(val) < 0:
                return RatioGoodness.GARBAGE, min(val)
            if abs(diff_min) < fratio_threshold: # ratios are ok
                return RatioGoodness.OK, abs(diff_min)
            elif diff_min < - fratio_threshold: # ratios are too small
                return RatioGoodness.TOO_SMALL, abs(diff_min)
            elif diff_min > fratio_threshold: #ratios are too big
                return RatioGoodness.TOO_BIG, abs(diff_min)
                

        candidates=[]
        too_small_biggest_emass=0.0
        too_big_smallest_emass=float('inf')
        have_garbage=False
        for emass in res_1.keys():
            for dt in res_1[emass].keys():
                goodness, off =ratios_are_ok(emass,dt,res_1)
                if goodness == RatioGoodness.OK:
                    candidates.append((dt,emass,off))
                elif goodness == RatioGoodness.TOO_BIG:
                    if emass > too_small_biggest_emass:
                        too_small_biggest_emass = emass
                elif goodness == RatioGoodness.TOO_SMALL:
                    if emass < too_big_smallest_emass:
                        too_big_smallest_emass = emass
                elif goodness == RatioGoodness.GARBAGE:
                    self.report('[analysis_step] dt,emass={},{} is garbage ({})'.format(dt,emass,off))
                    have_garbage=True
                else:
                    raise RuntimeError('Bug: missing case')


        def try_to_fix(test,fix=lambda d,e: (d*3.0/4.0, e) ):
            have_something_to_do=False
            for calc in self.ctx.dt_benchmark:
                emass, dt, pk = get_emass_dt_pk(calc)
                if test(emass,dt,calc):
                    #do something with the parameters to try a fix
                    dt,emass=fix(dt,emass)
                    startfrom=get_parent_calculation(calc.get_incoming().all_nodes(),minpk=self.node.pk)
                    if len(startfrom) != 1:                    
                        raise RuntimeError('Bug: wrong logic') 
                    params=copy.deepcopy(self.get_cp_resources_cp())
                    params['wallclock']=self.inputs.benchmark_emass_dt_walltime_s.value 
                    newcalc=configure_cp_builder_restart(            
                               self.get_cp_code(),                  
                               startfrom[0],                        
                               resources=params,
                               dt=dt, mu=emass,
                               print= lambda x : self.report('[analysis_step.try_to_fix] [builder] {}'.format(x))
                            )                                            
                    sub=self.submit(newcalc)
                    self.to_context(dt_benchmark=append_(sub))
                    have_something_to_do=True
                    self.report('[analysis_step.try_to_fix]: pk={} trying new dt,emass={},{}'.format(sub.pk,dt,emass))
            return have_something_to_do

        if len(candidates) == 0 :
            self.report('[analysis_step] no good candidates found. Try to fix')
            self.report('[analysis_step] results are: {}'.format(str(res_1)))
            self.ctx.emass_dt_not_ok = True
            #generate new set of parameters and run new simulations.
            #No candidates are available for 4 possible reasons: 1) emass too big, 2) emass to low, 3) emasses both too big and too low,  4) no successful simulations.
            #decide what is the case.
            if len(res_1) == 0 : # 4) no succesful simulation case: decrease all the timesteps and run again
                raise RuntimeError('Not implemented') 
            else: # if all non candidate simulations have a positive difference between the ratio and the desidered ratio, it means tha I had a too low emass. In all other cases            #loop over all simulations parameters
                increase_emass= too_small_biggest_emass > 0.0
                decrease_emass= too_big_smallest_emass < float('inf')
                self.report('[analysis_step] too_small_biggest_emass,too_big_smallest_emass={},{}'.format(too_small_biggest_emass,too_big_smallest_emass))
            if decrease_emass and not increase_emass: # 1)
                #1. I have to use smaller emass, or the smaller emasses did not have a small enought timestep
                #find if there is a broken simulation with emass lower than the current minimum
                self.report('[analysis_step] try to decrease emass')
		#TODO is this bugged? will the workchain resubmit always the same calculation over and over, as well as new ones?
                if try_to_fix(test=lambda emass_, dt_, calc_:  emass_ < too_big_smallest_emass and not calc_.is_finished_ok):
                    return # do it!
                # pick something lower and run it again!
                new_emass = too_big_smallest_emass*3.0/4.0
                dt_big=max(list(res_1[too_big_smallest_emass].keys()))
                new_dt=dt_big*(3.0/4.0)**0.5
                #get a calculation from wich we can start. pick the first one
                startfrom=get_parent_calc_from_emass_dt(res_1,too_big_smallest_emass,dt_big,minpk=self.node.pk)
                params=copy.deepcopy(self.get_cp_resources_cp())
                params['wallclock']=self.inputs.benchmark_emass_dt_walltime_s.value 
                newcalc=configure_cp_builder_restart(           
                                self.get_cp_code(),                  
                                startfrom[0],                        
                                resources=params,
                                dt=new_dt, mu=new_emass,
                                print= lambda x : self.report('[analysis_step] [builder] {}'.format(x))
                             )                            
                sub=self.submit(newcalc)           
                self.to_context(dt_benchmark=append_(sub))  
                self.report('[analysis_step] pk={} new emass,dt={},{}'.format(sub.pk,new_emass,new_dt))
                return #do it!
            elif increase_emass and decrease_emass: # case 3)
                #check if there are failed simulations with emass in between the biggest too small and the lowest too big
                self.report('[analysis_step] try to find emass in the middle')
                if try_to_fix(test=lambda emass_,dt_,calc_ : emass_ > too_small_biggest_emass and emass_ < too_big_smallest_emass and not calc_.is_finished_ok):
                    return # do it!
                # pick something in the middle and run it again!
                new_emass = (too_small_biggest_emass + too_big_smallest_emass)/2.0
                dt_small=max(list(res_1[too_small_biggest_emass].keys()))
                dt_big=max(list(res_1[too_big_smallest_emass].keys()))
                new_dt=(dt_small+dt_big)/2.0
                #get a calculation from wich we can start. pick the first one
                startfrom=get_parent_calc_from_emass_dt(res_1,too_small_biggest_emass,dt_small,minpk=self.node.pk)
                params=copy.deepcopy(self.get_cp_resources_cp())
                params['wallclock']=self.inputs.benchmark_emass_dt_walltime_s.value 
                newcalc=configure_cp_builder_restart(           
                                self.get_cp_code(),                  
                                startfrom[0],                        
                                resources=params,
                                dt=new_dt, mu=new_emass,
                                print= lambda x : self.report('[analysis_step] [builder] {}'.format(x))
                             )                               
                sub=self.submit(newcalc)        
                self.to_context(dt_benchmark=append_(sub))  
                self.report('[analysis_step] pk={} new emass,dt={},{}'.format(sub.pk,new_emass,new_dt))
                return #do it!
            elif increase_emass: #case 2) emass is too low, I can increase it!
                #check that simulations with higher emass are not failed
                self.report('[analysis_step] try to increase emass')
                if try_to_fix(test=lambda emass_, dt_, calc_:  emass_ > too_small_biggest_emass and not calc_.is_finished_ok):
                    return # do it!
                # pick something lower and run it again!
                new_emass = too_small_biggest_emass*4.0/3.0
                dt_small=max(list(res_1[too_small_biggest_emass].keys()))
                new_dt=dt_small*(4.0/3.0)**0.5
                #get a calculation from wich we can start. pick the first one
                startfrom=get_parent_calc_from_emass_dt(res_1,too_small_biggest_emass,dt_small, minpk=self.node.pk)
                params=copy.deepcopy(self.get_cp_resources_cp())
                params['wallclock']=self.inputs.benchmark_emass_dt_walltime_s.value 
                newcalc=configure_cp_builder_restart(           
                                self.get_cp_code(),                  
                                startfrom[0],                        
                                resources=params,
                                dt=new_dt, mu=new_emass,
                                print= lambda x : self.report('[analysis_step] [builder] {}'.format(x))
                             )       
                sub=self.submit(newcalc)                                
                self.to_context(dt_benchmark=append_(sub))  
                self.report('[analysis_step] pk={} new emass,dt={},{}'.format(sub.pk,new_emass,new_dt))
                return #do it!
            elif have_garbage:
                #try to decrease the emass (or the timestep)?
                return 405  
            else:
                raise RuntimeError('Bug: wrong logic')
        else:  # we have cadidate parameters ( len(candidates!=0) )
            self.report('[analysis_step] good candidates found. Going on.')
            self.ctx.emass_dt_not_ok = False # to exit the loop in the workchain
            #pick the one with the highest dt and highest emass run it
            #(check force variances?)
            best=(0.0,0.0,float('inf'))
            for candidate in candidates:
                if candidate[2]<best[2]:
                    best=candidate
            self.ctx.dt_emass_off=best
            self.ctx.force_ratios=res_1 # save the results
            self.report('[analysis_step] force_ratios={}'.format(res_1))
            self.report('[analysis_step] best candidate dt,emass,off={}'.format(best))
            dt_=Float(best[0])
            dt_.store()
            emass_=Float(best[1])
            emass_.store()
            self.ctx.dt=dt_
            self.ctx.emass=emass_
            #generate dictionary for ionic mass correction
            params=res_1[best[1]][best[0]]
            new_mass={}
            for p in params.keys():
                new_mass[p]=params[p]['fratios_mean']
            self.ctx.ionic_mass_corr=new_mass
            self.report('[analysis_step] mass_corrections={}'.format(new_mass))
        return

    def test_nose_step(self, force_ratios, dt_emass_off, tempw,pressure,vdos_maxs, cp_code, cp_code_res ):
        #setup fake environment
        set_fake_something(self,'ctx','current_cp_code',cp_code)
        set_fake_something(self,'ctx','current_cp_code_cp_resources',cp_code_res)
        set_fake_something(self,'ctx','dt_emass_off',dt_emass_off)
        set_fake_something(self,'ctx','force_ratios',force_ratios)
        set_fake_something(self,'inputs', 'tempw', tempw)
        set_fake_something(self,'inputs', 'pressure', pressure)
        set_fake_something(self,'report',None,lambda s,a: print(a))
        set_fake_something(self,'to_context',None,fake_function) 
        set_fake_something(self,'submit',None,fake_function) 
        global append_
        append_ = lambda a: a
        print('starting fake ctx state')
        print('starting tests:')
        res=self.nose()
        print('fake ctx state (remember that to_context does nothing):')
        print (self.ctx.__dict__)
        print ('return value from analysis_step: ',res)

    def setup_check1(self):
        if not self.inputs.skip_emass_dt_test.value:
            #find simulation to restart. take biggest pk of selected parameters
            dt,emass,off=self.ctx.dt_emass_off
            self.ctx.check1=get_calc_from_emass_dt (self.ctx.force_ratios,emass,dt)
            if 'vdos_maxs' in self.ctx:
                self.ctx.fnosep=abs(float(self.ctx.vdos_maxs[self.ctx.check1.pk]))
                self.report('[setup_check1] using calculated maximum in vdos as nose freq. {} THz'.format(self.ctx.fnosep))
            else:
                self.report('[setup_check1] using default nose freq. {} THz'.format(self.ctx.fnosep))
                
            if not self.ctx.check1.is_finished_ok:
                raise RuntimeError('Bug: wrong logic')
        else:
            #we started only with a cg calculation
            self.ctx.check1=self.ctx.initial_cg
            #set dt_emass_off
            self.ctx.dt_emass_off=(self.inputs.dt.value, self.inputs.emass.value,0)
    
    def nose_setup(self):
        if 'last_nve' in self.ctx:
            self.ctx.last_nve.append(self.ctx.check1)
        else:
            self.ctx.last_nve=[self.ctx.check1] 
        self.ctx.idx_thermo_cycle=0

    def nose_prepare(self):
        #next final_cg will start from scratch, setting the correct number of plane waves
        self.ctx.cg_scratch=True
        self.ctx.nose_start=len(self.ctx.last_nve)
        self.ctx.run_nose_ps = self.inputs.thermobarostat_points[     int(self.ctx.idx_thermo_cycle)]['thermostat_time_ps']
        self.ctx.tempw_current = self.inputs.thermobarostat_points[   int(self.ctx.idx_thermo_cycle)]['temperature_K']
        if 'pressure_KBar' in self.inputs.thermobarostat_points[int(self.ctx.idx_thermo_cycle)]:
            self.ctx.pressure_current = self.inputs.thermobarostat_points[int(self.ctx.idx_thermo_cycle)]['pressure_KBar']
            self.ctx.do_pr = True
        else:
            self.ctx.do_pr = False
            self.ctx.pressure_current = float('nan')
        self.ctx.run_nve_ps = self.inputs.thermobarostat_points[      int(self.ctx.idx_thermo_cycle)]['equilibration_time_ps']

    def fix_last_nve(self,report):
        if not self.ctx.last_nve[-1].is_finished_ok:
            report('[fix_last_nve] last_nve with pk={} is failed with {}'.format(self.ctx.last_nve[-1].pk,self.ctx.last_nve[-1].get_attribute_many(['exit_status', 'exit_message'])))
            if self.ctx.retry_count==0:
                #try to resend calc and hope for the best
                report('[fix_last_nve] resubmitting it')
                resend=self.ctx.last_nve[-1].get_builder_restart()
                self.ctx.retry_count=self.ctx.retry_count+1
                del(self.ctx.last_nve[-1])
                sub=self.submit(resend)
                self.to_context(last_nve=append_(sub))
                return 1
            else:
                report('[fix_last_nve] not teached how to deal with this, sorry')
                return 408
        else:
            self.ctx.retry_count=0
            return 0


    def nose(self):
        self.report('[nose] beginning.')
        res=self.fix_last_nve(report=lambda x : self.report('[nose] {}'.format(x)))
        if res != 0:
            if res>400:
                return res
            return
        nose_ps_done=get_total_time(self.ctx.last_nve[self.ctx.nose_start:])
        # run the thermostat
        ibrav = self.ctx.last_nve[-1].inputs.parameters.get_dict()['SYSTEM'].get('ibrav',0)
        nose_pr_param={
            'CONTROL': {
                'calculation': 'vc-cp' if self.ctx.do_pr else 'cp',
            },
            'IONS': { 
                'ion_temperature': 'nose',
                'tempw': float(self.ctx.tempw_current),
                'fnosep': self.ctx.fnosep,
                'nhpcl' : 3,
            },
        }
        if self.ctx.do_pr:
            nose_pr_param['CELL'] = {
                'cell_dynamics': 'pr',
                'press': float(self.ctx.pressure_current),
                'cell_dofree': 'volume' if ibrav == 1 else 'all',
            }
        
        dt,emass,off=self.ctx.dt_emass_off
        newcalc=configure_cp_builder_restart(                
                self.get_cp_code(),                  
                self.ctx.last_nve[-1],                        
                resources=self.get_cp_resources_cp(),
                dt=dt, mu=emass,
                additional_parameters=nose_pr_param,
                cmdline=self.ctx.cmdline_cp,
                ttot_ps=abs(self.ctx.run_nose_ps-nose_ps_done),
                stepwalltime_s= self.ctx.stepwalltime_nose_s if 'stepwalltime_nose_s' in self.ctx else ( self.ctx.stepwalltime_s*3 if 'stepwalltime_s' in self.ctx else None  ),
                   print= lambda x : self.report('[nose] [builder] {}'.format(x))
             )                                       
        sub=self.submit(newcalc)
        self.to_context(last_nve=append_(sub)) 
        self.report(f'[nose] sent to context pk,dt,emass,tempw,fnosep,press={sub.pk},{dt},{emass},{float(self.ctx.tempw_current)},{self.ctx.fnosep},{float(self.ctx.pressure_current) if self.ctx.do_pr else "fixed_cell"}')
        return        
   
    def check_nose(self):
        nose_number=len(self.ctx.last_nve)-self.ctx.nose_start
        elapsed_simulation_time=get_total_time(self.ctx.last_nve[self.ctx.nose_start:])
        #if necessary, get timestep walltime
        if nose_number>0 and not 'stepwalltime_nose_s' in self.ctx: 
            try:
                time,nsteps=main_loop_line(self.ctx.last_nve[-1])
                r=time/float(nsteps)
                self.ctx.stepwalltime_nose_s=r
                self.report('[check_nose] nose wall time: {}s per step'.format(r))
            except IndexError:
                self.report('[check_nose] cannot read nose wall time')
 
        if elapsed_simulation_time < float(self.ctx.run_nose_ps):
            self.report('[check_nose] finished nose steps: {}'.format(nose_number))
            return True 
        else:
            self.report('[check_nose] nose finished. Total time {} ps. Number of nose submitted: {}'.format(elapsed_simulation_time,nose_number))
            return False
        
    def check_nve_nose(self):
        if 'max_slope_ok' in self.ctx:
            if not self.ctx.max_slope_ok:
                return True
        elapsed_simulation_time=get_total_time(self.ctx.last_nve[self.ctx.first_prod_nve_idx:])
        nose_number=len(self.ctx.last_nve)-self.ctx.first_prod_nve_idx
        #if necessary, get timestep walltime
        if nose_number>0 and not 'stepwalltime_s' in self.ctx: 
            try:
                time,nsteps=main_loop_line(self.ctx.last_nve[-1])
                r=time/float(nsteps)
                self.ctx.stepwalltime_s=r
                self.report('[check_nve_nose] nve wall time: {}s per step'.format(r))
            except IndexError:
                self.report('[check_nve_nose] cannot read nve wall time')
        if elapsed_simulation_time < float(self.ctx.run_nve_ps):
            self.report('[check_nve_nose] finished nve steps: {}'.format(nose_number))
            return True 
        else:
            self.report('[check_nve_nose] equilibration nve finished. Total time {} ps. Number of nve submitted: {}'.format(elapsed_simulation_time,nose_number))
            return False
        
        #if not self.ctx.last_nve[-1].is_finished_ok:
        #    return 402
        #return

    
     
    def final_cg(self):
        if not 'cg_scratch' in self.ctx:
            self.ctx.cg_scratch=False 

        if 'max_slope_ok' in self.ctx:
            if self.ctx.max_slope_ok:
                cg_reset_emass_dt=False
            else:
                cg_reset_emass_dt=True
        else:
            cg_reset_emass_dt=False
            
        
        final_cg=configure_cp_builder_restart(
                   self.get_cp_code(),
                   self.ctx.last_nve[-1],
                   resources=self.get_cp_resources_cg(),
                   copy_mu_mucut=False if cg_reset_emass_dt else True ,
                   dt= self.ctx.dt_emass_off[0] if cg_reset_emass_dt else None,
                   mu=self.ctx.dt_emass_off[1] if cg_reset_emass_dt else None,
                   mucut=self.ctx.max_slope_emass_cut if cg_reset_emass_dt else None,
                   cg=True,
                   tstress=False,
                   from_scratch=True, #if (self.ctx.cg_scratch or cg_reset_emass_dt) else False,
                   nstep=1,
                   remove_parameters_namelist=['CELL'],
                   additional_parameters={'IONS':{'ion_temperature': 'not_controlled'} },
                   cmdline=['-ntg', '1', '-nb', '1'],
                   print= lambda x : self.report('[final_cg] [builder] {}'.format(x))
              )
        self.ctx.after_last_cg_idx=len(self.ctx.last_nve)+1
        sub=self.submit(final_cg)
        self.to_context(last_nve=append_(sub))
        if True: #self.ctx.cg_scratch or cg_reset_emass_dt:
            #reset time per timestep
            if 'stepwalltime_s' in self.ctx:
                del(self.ctx.stepwalltime_s)
            if 'stepwalltime_nose_s' in self.ctx:
                del(self.ctx.stepwalltime_nose_s) 
        self.ctx.cg_scratch=False
        self.report('[final_cg] pk={} cg to context (1 step).'.format(sub.pk))
        return
 
    def check_final_cg(self):
        self.ctx.first_prod_nve_idx=len(self.ctx.last_nve)
        if not self.ctx.last_nve[-1].is_finished_ok:
            return 403
        return

    def equil_not_ok(self):
        if self.inputs.skip_thermobarostat.value:
            return False
        if not 'tempw_current' in self.ctx:
            return True
        # join all previous trajectories
        joined_traj=merge_many_traj(self.ctx.last_nve[self.ctx.first_prod_nve_idx:])
        t=joined_traj.get_array('ionic_temperature')
        p=joined_traj.get_array('pressure')
        tm=t.mean()
        tstd=t.std()
        pm=p.mean()
        pstd=p.std()
        self.report('[equil_not_ok] T={} std(T)={}'.format(tm,tstd))
        self.report('[equil_not_ok] p={} std(p)={}'.format(pm,pstd))
        if self.inputs.temperature_tolerance.value > 0:
            tstd=self.inputs.temperature_tolerance.value
        if self.inputs.pressure_tolerance.value > 0:
            pstd=self.inputs.pressure_tolerance/10.0
        if abs(tm-float(self.ctx.tempw_current))<tstd and ( (not self.ctx.do_pr) or ( abs(pm-float(self.ctx.pressure_current)/10.0) < pstd ) ):
            #we completed this step
            self.ctx.idx_thermo_cycle = self.ctx.idx_thermo_cycle + 1
            self.report('[equil_not_ok] thermo cycle {}/{} completed'.format(self.ctx.idx_thermo_cycle,len(self.inputs.thermobarostat_points)))
        else:
            self.report('[equil_not_ok] repeating thermo cycle {}/{}'.format(self.ctx.idx_thermo_cycle+1,len(self.inputs.thermobarostat_points)))
       
        if self.ctx.idx_thermo_cycle >= len(self.inputs.thermobarostat_points):
            self.report('[equil_not_ok] equilibrated')
            return False
        else:
            return True

    def setup_check2(self):
        if not 'last_nve' in self.ctx:
            self.ctx.last_nve=[self.ctx.check1]
        self.ctx.check2=self.ctx.last_nve[-1]
        self.ctx.run_nve_ps=self.inputs.nve_required_picoseconds.value
        

    def run_nve(self):
        res=self.fix_last_nve(report=lambda x : self.report('[run_nve] {}'.format(x)))
        if res != 0:
            if res>400:
                return res
            return
        if not 'find_new_ion_masses' in self.ctx:
            self.ctx.find_new_ion_masses=False
        s=None
        if self.ctx.find_new_ion_masses:
            self.ctx.find_new_ion_masses=False 
            if self.inputs.adjust_ionic_mass.value:
                self.report('[run_nve] adjusting ionic masses according to force ratio')
                s=set_mass(Dict(dict=self.ctx.ionic_mass_corr),self.ctx.check1.inputs.structure)
            else:
                self.report('[run_nve] keeping previous ionic masses')

        nve_ps_done=get_total_time(self.ctx.last_nve[self.ctx.first_prod_nve_idx:])
        nve=configure_cp_builder_restart(
                   self.get_cp_code(),
                   self.ctx.last_nve[-1],
                   resources=self.get_cp_resources_cp(),
                   copy_mu_mucut=True,
                   cmdline=self.ctx.cmdline_cp,
                   ttot_ps=abs(self.ctx.run_nve_ps-nve_ps_done),
                   stepwalltime_s= self.ctx.stepwalltime_s if 'stepwalltime_s' in self.ctx else None,
                   print= lambda x : self.report('[run_nve] [builder] {}'.format(x)),
                   structure=s
            )
        sub=self.submit(nve)
        self.to_context(last_nve=append_(sub)) 
        self.report('[run_nve] pk={} nve to context'.format(sub.pk))
        return

    def benchmark_parallelization_options(self):
        if self.inputs.skip_parallel_test.value:
            return
        s=None
        if not self.inputs.skip_emass_dt_test.value:
            if self.inputs.adjust_ionic_mass.value:
                self.report('[benchmark_parallelization_options] adjusting ionic masses according to force ratio')
                s=set_mass(Dict(dict=self.ctx.ionic_mass_corr),self.ctx.check1.inputs.structure)
            else:
                self.report('[benchmark_parallelization_options] keeping previous ionic masses' )
        resources=copy.deepcopy(self.get_cp_resources_cp())
        resources['wallclock']=self.inputs.benchmark_parallel_walltime_s.value
        configlist=possible_ntg_nb(resources['resources']['num_machines'],
                                   resources['resources']['num_mpiprocs_per_machine'])
        if len(configlist)<9:
            configlist_cut=configlist
        else:
            configlist_cut=configlist[:9]
        for nb,ntg in configlist_cut:
            nve=configure_cp_builder_restart(               
                       self.get_cp_code(),                  
                       self.ctx.check1,
                       resources=self.get_cp_resources_cp(),
                       copy_mu_mucut=True,
                       cmdline=['-ntg', str(ntg), '-nb', str(nb)],
                       structure = s,
                       nstep=self.inputs.nstep_parallel_test.value,
                       print= lambda x : self.report('[benchmark_parallelization_options] [builder] {}'.format(x))
                )
            sub=self.submit(nve)
            self.to_context(parallel_benchmark=append_(sub)) 
            self.report('[benchmark_parallelization_options] pk={} nve to context: -nb {} -ntg {}'.format(sub.pk,nb,ntg))
        return

    def benchmark_analysis(self):
        if self.inputs.skip_parallel_test.value:
            return
        steptime=float('inf')
        cmdline=['-ntg',str(1),'-nb',str(1)]
        for cp in self.ctx.parallel_benchmark:
            try:
                time,nsteps=main_loop_line(cp)
                r=time/float(nsteps)
                self.report('[benchmark_analysis] {}: {}s per step'.format(cp.inputs.settings['cmdline'], r))
                if r<steptime:
                    steptime=r
                    cmdline=cp.inputs.settings['cmdline']
                    self.ctx.check1=cp
            except Exception as e:
                self.report(e)
        self.ctx.cmdline_cp=cmdline
        self.report('[benchmark_analysis] BEST {}: {}s per step'.format(cmdline, steptime))
        if steptime==float('inf'):
            return
        self.ctx.stepwalltime_s=steptime

    def run_more(self):
        elapsed_simulation_time=get_total_time(self.ctx.last_nve[self.ctx.first_prod_nve_idx:])
        if elapsed_simulation_time < self.inputs.nve_required_picoseconds.value:
            self.ctx.nve_count = self.ctx.nve_count + 1
            self.report('[run_more] nve_count: {}'.format(int(self.ctx.nve_count)))
            return True 
        else:
            self.report('[run_more] simulation finished. Total time {} ps. number if NVE simulations submitted {}'.format(elapsed_simulation_time,self.ctx.nve_count))
            return False
        return
 
    def get_result(self):
        self.report('[get_result] workflow terminated. Preparing outputs.')
        self.out('dt',self.ctx.dt)
        self.out('emass',self.ctx.emass)
        #merge all nve trajectories
        res=merge_many_traj(self.ctx.last_nve[self.ctx.first_prod_nve_idx:])
        self.out('nve_prod_traj', res)
        #concatenate all other trajectories
        res1=merge_many_traj(self.ctx.last_nve[:self.ctx.first_prod_nve_idx],unique=False)
        #cmdline
        if res1 is not None:
            res1=concatenate_traj2(res1,res)
            self.out('full_traj',res1)
        cmdline=List(self.ctx.last_nve[-1].inputs.settings['cmdline'])
        cmdline.store()
        self.out('cmdline_cp',cmdline)
        kinds=List(self.ctx.last_nve[-1].inputs.structure.get_attribute('kinds'))
        kinds.store()
        self.out('kinds',kinds)        
        return


class Empty:
    pass
class FakeClass(CpWorkChain):
    pass

def set_fake_something(self,key1,key2,value):
    if not key1 in FakeClass.__dict__:
        setattr(FakeClass,key1,Empty())
    if key2 is not None:
        setattr(getattr(FakeClass,key1),key2,value)
    else:
        setattr(FakeClass,key1,value)
    self.__class__ = FakeClass

