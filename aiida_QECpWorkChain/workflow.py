import copy
from enum import Enum

import aiida.orm
from aiida.orm import Int, Float, Str, List, Dict, ArrayData, Bool
from aiida.engine import WorkChain, calcfunction, ToContext, append_, while_, if_, return_
from aiida.orm.nodes.data.upf import get_pseudos_from_structure
from aiida.plugins.factories import DataFactory
import numpy as np


####
# utilities for manipulating nested dictionary

def dict_keys(d,level=0):
    #print (d,level)
    if level==0:
        return list(d.keys())
    elif level>0: return dict_keys(d[list(d.keys())[0]],level-1)
    else: raise ValueError('level cannot be negative ({})'.format(level))

def get_element(d,keylist):
    #print(keylist)
    if keylist:
        return get_element(d[keylist.pop()],keylist)
    else:
        return d
    
def inverse_permutation(p):
    inv=[None] * len(p)
    for i in range(len(p)):
        inv[p[i]]=i
    return inv

def apply_perm(a,p,reverse=False):
    if reverse:
        return list(reversed([ a[i] for i in p ]))
    else:
        return [ a[i] for i in p ]
        
def permutation_nested_dict(d,level=0, maxlevels=0,keylist=[],permutation=[]):
    if permutation == []:
        permutation=list(range(maxlevels+1))
    #print (level,maxlevels,keylist,permutation)
    if level==maxlevels:
        return { what:  get_element(
                                d,
                                apply_perm(
                                    keylist+[what],
                                    inverse_permutation(list(reversed(permutation))),
                                    reverse=True
                                )
                        )  for what in dict_keys(d,permutation[0]) 
               }
    else:
        return { 
            what: permutation_nested_dict(
                            d,
                            level+1,
                            maxlevels,
                            keylist=keylist+[what],
                            permutation=permutation
                  ) 
            for what in dict_keys(d,permutation[maxlevels-level])
        }
    
def test_permutation_nested_dict(orig,permuted,permutation,keylist=[]):
    #print('keylist=',keylist)
    element=get_element(orig,keylist.copy())
    if isinstance(element,dict):
        for key in element.keys():
            #print ('new keylist',keylist+[key])
            test_permutation_nested_dict(orig,permuted,permutation,keylist=[key]+keylist)
    else:
        print ('testing',keylist,'==', apply_perm(list(reversed(keylist)),permutation))
        print(element,'==',get_element(permuted,apply_perm(list(reversed(keylist)),permutation)))
        assert( element == get_element(permuted,apply_perm(list(reversed(keylist)),permutation)))




####



def validate_structure(structure_or_trajectory):
    return

def validate_pseudo_family(pseudo_family_str):
    return

def validate_cp_list(cp_code):
    return

def validate_pw_list(pw_code):
    return

def validate_ecutwfc(ecutwfc):
    if ecutwfc <= 0.0:
        return 'ecutwfc must be a positive number'
    return

def validate_tempw(tempw):
    if tempw <= 0.0:
        return 'tempw must be a positive number'
    return



def get_node(node):
    if isinstance(node, aiida.orm.nodes.Node):
        return node
    else:
        return aiida.orm.load_node(node)

def get_pseudo_from_inputs(submitted):
    return {x[9:] : getattr(submitted.inputs,x)  for x in dir(submitted.inputs) if x[:9]=='pseudos__'  }



def configure_cp_builder_cg(code,
                            pseudo_family,
                            aiida_structure,
                            ecutwfc,
                            tempw,
                            resources, #={ 'num_machines': 1, 'num_mpiprocs_per_machine': 20},
                            additional_parameters={},
                            nstep=50,
                            ion_velocities=None,
                            dt=3.0
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
    if ion_velocities is not None:
        parameters['ATOMIC_VELOCITIES']=ion_velocities
    if nstep>22:
        parameters['AUTOPILOT'] = [
        {'onstep' : 7, 'what' : 'dt', 'newvalue' : 6.0 },
        {'onstep' : 14, 'what' : 'dt', 'newvalue' : 18.0},
        {'onstep' : 21, 'what' : 'dt', 'newvalue' : 60.0},
        {'onstep' : nstep-1, 'what' : 'dt', 'newvalue' : dt},
        ]
    elif nstep>10:
        parameters['AUTOPILOT'] = [
        {'onstep' : 7, 'what' : 'dt', 'newvalue' : 15.0 },
        {'onstep' : nstep-1, 'what' : 'dt', 'newvalue' : dt},
        ]
        
    
    for key in additional_parameters.keys():
        for subkey in additional_parameters[key].keys():
            parameters.setdefault(key,{})[subkey]=additional_parameters[key][subkey]
    builder.parameters = Dict(dict=parameters)
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
                                    cmdline=None
                                ):
    '''
    rescaling of atomic and wfc velocities is performed, if needed, using the dt found in the CONTROL namelist.
    If dt is changed with autopilot this can be wrong.
    max_seconds is changed according to 0.9*wallclock, where wallclock is the time requested to the scheduler.
    It performs a restart. In general all parameters but the one that are needed to perform a CP dynamics
    are copied from the parent calculation. tstress and tprnfor are setted to True.
    additional_parameters is setted at the end, so it can override every other parameter setted anywhere before
    or during this function call.
    remove_parameters_namelist are removed at the beginning
    cmdline can be, for example, ['-ntg', '2'] or a longer list. if not specified it is copied from the old calculation
    '''
    start_from = get_node(start_from)
    builder=code.get_builder()
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
    #note that the structure will not be used (he has the restart)
    builder.structure = start_from.inputs.structure
    builder.parent_folder = start_from.outputs.remote_folder
    #settings_dict = {
    #    'cmdline': ['-n', '16'],
    #}
    #builder.settings = Dict(dict=settings_dict)
    builder.settings=start_from.inputs.settings.clone()
    if cmdline is not None:
        builder.settings['cmdline']=cmdline
    
    builder.pseudos = get_pseudo_from_inputs(start_from)
    parameters = copy.deepcopy(start_from.inputs.parameters.get_dict())
    for itemtodel in remove_parameters_namelist:
        parameters.pop(itemtodel,None)
    parameters['CONTROL']['calculation'] = 'cp'
    parameters['CONTROL']['restart_mode'] = 'restart'
    parameters['CONTROL']['tstress'] = True
    parameters['CONTROL']['tprnfor'] = True
    parameters['CONTROL']['max_seconds'] = int(wallclock*0.9)
    parameters['IONS']['ion_velocities'] = 'default'
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
        if abs(parameters['CONTROL']['dt'] - dt) > 1e-5 and dtchange:
            parameters['IONS']['ion_velocities'] = 'change_step'
            parameters['IONS']['tolp'] = parameters['CONTROL']['dt']
            parameters['ELECTRONS']['electron_velocities'] = 'change_step'
        parameters['CONTROL']['dt'] = dt
    parameters['CONTROL']['nstep'] = nstep
    if remove_autopilot:
        try:
            del parameters['AUTOPILOT']
            print ('removed AUTOPILOT input')
        except:
            print ('no AUTOPILOT input to remove')
    for key in additional_parameters.keys():
        for subkey in additional_parameters[key].keys():
            parameters.setdefault(key,{})[subkey]=additional_parameters[key][subkey]
    builder.parameters = Dict(dict=parameters)
    builder.metadata.options.resources = resources_
    builder.metadata.options.max_wallclock_seconds = wallclock
    builder.metadata.options.queue_name = queue
    return builder



#stuff to analyze the trajectory and traverse the aiida graph

def get_children_nodetype(node,nodetype):
    '''
        Find in a recursive way all nodetype nodes that are in the output tree of node.
        The recursion is stopped when the node has no inputs or it is a nodetype node,
        otherwise it will go to every parent node.
    '''
    if isinstance(node, list):
        res=[]
        for n in node:
            res=res+get_children_nodetype(n,nodetype)
        return res
    if not isinstance(node, nodetype):
        incoming=node.get_outgoing().all_nodes()
        res=[]
        for i in incoming:
            res = res + get_children_nodetype(i,nodetype)
        return res
    else:
        return [node]
    
def get_children_calculation(node):
    return get_children_nodetype(node,aiida.orm.nodes.process.calculation.calcjob.CalcJobNode)

def get_children_calcfunction(node):
    return get_children_nodetype(node,aiida.orm.nodes.process.calculation.calcfunction.CalcFunctionNode)

def get_children_trajectory(node):
    return get_children_nodetype(node,aiida.orm.nodes.data.TrajectoryData)


def get_parent_nodetype(node,nodetype):
    '''
        Find in a recursive way all nodetype nodes that are in the input tree of node.
        The recursion is stopped when the node has no inputs or it is a nodetype node,
        otherwise it will go to every parent node.
    '''
    if isinstance(node, list):
        res=[]
        for n in node:
            res=res+get_parent_nodetype(n,nodetype)
        return res
    if not isinstance(node, nodetype):
        incoming=node.get_incoming().all_nodes()
        res=[]
        for i in incoming:
            res = res + get_parent_nodetype(i,nodetype)
        return res
    else:
        return [node]
    
def get_parent_calculation(node):
    return get_parent_nodetype(node,aiida.orm.nodes.process.calculation.calcjob.CalcJobNode)

def get_parent_calcfunction(node):
    return get_parent_nodetype(node,aiida.orm.nodes.process.calculation.calcfunction.CalcFunctionNode)

def get_parent_trajectory(node):
    return get_parent_nodetype(node,aiida.orm.nodes.data.TrajectoryData)

def get_atomic_types_and_masks(type_array):
    names=set(type_array)
    masks=[]
    outnames=[]
    for name in set(names):
        masks.append(np.array(type_array)==name)
        outnames.append(name)
    return zip(outnames,masks)


#compare forces between reference pw calculation and cp one
def compare_forces(pwcalc):
    #get cp trajectory (I am assuming that the graph is as I made it)
    parent_traj=get_parent_trajectory(pwcalc.get_incoming().all_nodes())
    #get calcfunctions (one of them has the index of the step)
    parent_calcfunctions=get_parent_calcfunction(pwcalc.get_incoming().all_nodes())
    if len(parent_traj) != 1:
        raise RuntimeError('wrong number of direct parent trajectories in the graph ({})'.format(len(parent_traj)))
    #get parent cp calculation
    parent_cp=get_parent_calculation(parent_traj)
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

def compare_forces_many(submitted):
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
            list_ratios, emass, pk, dt = compare_forces(sub)
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


def analyze_forces_ratio(pwcalcjobs,fthreshold=0.1,corrfactor=1.0,ax=None):
    '''
    Given a list of pw calcjobs, analyzes the force ratios between them and their cp ancestors.
    Produces a series of histogram plots and mean and standard deviation of the ratio in an output dictionary.
    '''
    atom_forces=compare_forces_many(pwcalcjobs)
    res={}
    for emass in atom_forces.keys():
        for dt in atom_forces[emass].keys():
            if ax is not None:
                ax.set_title('emass={}, dt={}, PK={}'.format(emass,dt,atom_forces[emass][dt]['PK']))
            for element in atom_forces[emass][dt]['fratios'].keys():
                mask=atom_forces[emass][dt]['forces'][element]>fthreshold
                res.setdefault(emass,{}).setdefault(dt,{})[element]={
                    'forces_std': atom_forces[emass][dt]['forces'][element][mask].std(),
                    'fratios_mean': atom_forces[emass][dt]['fratios'][element][mask].mean()*corrfactor,
                    'fratios_std': atom_forces[emass][dt]['fratios'][element][mask].std()*corrfactor,
                    'PK': atom_forces[emass][dt]['PK']
                }
                if ax is not None:
                    ax.hist(
                    atom_forces[emass][dt]['fratios'][element][
                        mask
                    ]*corrfactor,
                    bins=100,alpha=0.5,label='{0:s} mean={1:.4f} std={2:.4f}'.format(element,res[emass][dt][element]['fratios_mean'],res[emass][dt][element]['fratios_std'])
                )
            if ax is not None:
                ax.legend()
    if ax is not None:
        return res, ax
    else:
        return res

def check_econt(cpcalcjob):
    pass    

#various calcfunctions

@calcfunction
def extract_structure_from_trajectory(traj, step_index=lambda: Int(-1)):
    if abs(int(step_index)) < traj.numsteps:
        return traj.get_step_structure(int(step_index))
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

def get_structures_from_trajectory(traj,every=1):
    structures=[]
    for i in range(0,traj.numsteps,every):
        structures.append(extract_structure_from_trajectory(traj,Int(i)))
    return structures



def generate_pw_from_trajectory(pwcode, start_from,
                                    skip=1, numcalc=0,
                                    resources=None
                                ):
    start_from = get_node(start_from)
    pseudos = get_pseudo_from_inputs(start_from)
    traj=start_from.outputs.output_trajectory
    if numcalc==0 and skip>0:
        pass
    elif numcalc>0 and skip==1:
        skip=int(traj.numsteps/numcalc)
        if skip==0:
            skip=1
    else:
        raise KeyError('cannot specify both skip and numcalc')
    structures=get_structures_from_trajectory(traj,skip)
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
        KpointsData = DataFactory('array.kpoints')
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
                'ecutwfc': start_from.inputs.parameters.get_dict()['SYSTEM']['ecutwfc'],
            },
            'ELECTRONS' : {
            },
        }
        builder.settings = Dict(dict={'gamma_only': True})
        builder.parameters = Dict(dict=parameters)
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

def get_parent_calc_from_emass_dt(res,emass,dt):
    startfrom=get_parent_calculation(get_calc_from_emass_dt(res,emass,dt).get_incoming().all_nodes())
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
    tstart=[]
    tstop=[]
    for cp in cps:
        tstart.append(cp.outputs.output_trajectory.get_array('times')[0])
        tstop.append (cp.outputs.output_trajectory.get_array('times')[1])
    return max(tstop)-min(tstart)
    

#TODO
@calcfunction
def set_mass(new_mass,structure,multiply=True):
    new_structure=structure.clone()
    new_kinds=copy.deepcopy(structure.get_attribute('kinds'))
    for kind in new_kinds:
        if multiply:
            kind['mass']=new_mass[kind['name']]*kind['mass']
        else:
            kind['mass']=new_mass[kind['name']]
    new_structure.set_attribute('kinds',new_kinds)
    return new_structure



#TODO
def volatility(ek,t):
    return sum(abs(ek[1:]-ek[:-1])/(t[1:]-t[:-1]))


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

#TODO
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
        spec.input('structure', required=True, valid_type=(aiida.orm.nodes.data.StructureData,aiida.orm.nodes.data.TrajectoryData),validator=validate_structure)
        spec.input('pseudo_family', required=True, valid_type=(Str),validator=validate_pseudo_family)
        spec.input('ecutwfc', required=True, valid_type=(Float),validator=validate_ecutwfc)
        spec.input('tempw', required=True, valid_type=(Float),validator=validate_tempw)
        spec.input('pressure', required=False, valid_type=(Float))
        spec.input('pw_code', required=True, valid_type=(aiida.orm.nodes.data.code.Code),validator=validate_pw_list)
        spec.input('cp_code',required=True, valid_type=(aiida.orm.nodes.data.code.Code), validator=validate_cp_list)
        spec.input('pw_resources_list',valid_type=(List),required=True)
        spec.input('cp_resources_cp_list',valid_type=(List),required=True)
        spec.input('cp_resources_cg_list',valid_type=(List),required=True)
        spec.input('target_force_ratio', valid_type=(Float), default=lambda: Float(0.9), validator= lambda a: 'target_force_ratio must be between 0 and 1' if a>=1.0 or a<=0.0 else None )
        spec.input('additional_parameters_cp', valid_type=(Dict),default=lambda: Dict(dict={}))
        spec.input('dt_start_stop_step', valid_type=(List), default=lambda: List(list=[2.0,4.0,20.0]))
        spec.input('emass_start_stop_step_mul', valid_type=(List), default=lambda: List(list=[1.0,1.0,8.0,25.0]))
        spec.input('number_of_pw_per_trajectory', valid_type=(Int), default=lambda: Int(100))
        spec.input('skip_emass_dt_test',valid_type=(Bool), default=lambda: Bool(False))
        spec.input('skip_thermobarostat',valid_type=(Bool),  default=lambda: Bool(False))
        spec.input('nve_required_picoseconds',valid_type=(Float), default=lambda: Float(50.0))
        spec.input('nstep_initial_cg',valid_type=(Int), default=lambda: Int(50))
        spec.input('initial_atomic_velocities',valid_type=(ArrayData),required=False)
        spec.input('dt',valid_type=(Float),required=False)
        spec.input('emass',valid_type=(Float),required=False)
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
            cls.benchmark_parallelization_options,
            cls.benchmark_analysis,
            #thermobarostatation
            while_(cls.equil_not_ok)(
                cls.nose,
                cls.check_nose, # this sets ctx.check2
                cls.final_cg,# append to ctx.last_nve
                cls.check_final_cg, 
                cls.run_nve # append to ctx.last_nve
            ),
            cls.setup_check2, #everything will start from ctx.check2
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

        spec.output('result')

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
        #extract structure, if necessary
        if isinstance(self.inputs.structure,aiida.orm.nodes.data.TrajectoryData ):
            self.ctx.start_structure = extract_structure_from_trajectory(self.inputs.structure)
        else:
            self.ctx.start_structure = self.inputs.structure
        self.ctx.start_structure = collapse_kinds(self.ctx.start_structure)
        #set default codes 
        self.set_pw_code(0)
        self.set_cp_code(0)
        self.ctx.emass_dt_not_ok=True
        self.ctx.nve_count=0 
        self.ctx.cmdline_cp=['-ntg',1,'-nb',1]
        if self.input.skip_emass_dt_test.value:
            if not 'dt' in self.inputs and not 'emass' in self.inputs:
                self.report('[setup] ERROR: if you want to skip the emass/dt benchmark, you must provide values for dt and emass!')
                return 406 
        self.report('setup completed')

    def find_emass_dt(self):
        return self.inputs.skip_emass_dt_test.value

    def small_equilibration_cg(self):
        #This is always the first step
        #build cg input
        builder = configure_cp_builder_cg(
                                self.get_cp_code(),
                                self.inputs.pseudo_family,
                                self.ctx.start_structure,
                                self.inputs.ecutwfc,
                                self.inputs.tempw,
                                self.get_cp_resources_cg (),
                                additional_parameters=self.inputs.additional_parameters_cp.get_dict(),
                                ion_velocities = self.inputs.initial_atomic_velocities if 'initial_atomic_velocities' in self.inputs else None 
                               )
        
        node = self.submit(builder)
        self.to_context(initial_cg=node)
        
        self.report('[small_equilibration] cg sent to context')
        return

    def emass_benchmark(self):
        #check that initial cg is ok

        if not self.ctx.initial_cg.is_finished_ok:
            return 401

        #...

        #prepare multiple
        params=self.get_cp_resources_cp()
        start,stop,step,fac=self.inputs.emass_start_stop_step_mul
        for mu in fac*np.arange(start,stop,step)**2:
            calc=configure_cp_builder_restart(
                                               self.get_cp_code(),
                                               self.ctx.initial_cg,
                                               mu=mu,
                                               resources=params
                                             )
            self.to_context(emass_benchmark=append_(self.submit(calc)))
            self.report('[emass_benchmark] emass={} sent to context'.format(mu))
           
        #apply first correction of atomic mass
        return

    def dt_benchmark(self):
        params=self.get_cp_resources_cp()
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
                                                   dt=dt
                                                   )
                    self.to_context(dt_benchmark=append_(self.submit(newcalc)))
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

    def analysis_step(self):

        self.report('[analysis_step] beginnig')
 
        #calculate vibrational spectra to have nice nose frequencies. Simply pick the frequency of the maximum
        vdos_maxs={}
        for calc in self.ctx.dt_benchmark:
            if calc.is_finished_ok:
                vdos_maxs[calc.pk]=get_maximum_frequency_vdos(calc.outputs.output_trajectory)
        self.ctx.vdos_maxs=vdos_maxs
        self.report('[analysis_step] maximum of vibrational spectra: {}'.format(vdos_maxs))
        #analyze forces ratio
        self.report('[analysis_step] comparing {} pw...'.format(len(self.ctx.compare_pw)))
        res_1 = analyze_forces_ratio(self.ctx.compare_pw)
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
            val=np.array(val)
            #if values are too near zero, this is garbage
            if min(val) < 0:
                return RatioGoodness.GARBAGE, min(val)
            if abs(diff.min()) < fratio_threshold: # ratios are ok
                return RatioGoodness.OK, abs(diff.min())
            elif diff.min() < - fratio_threshold: # ratios are too small
                return RatioGoodness.TOO_SMALL, abs(diff.min())
            elif diff.min() > fratio_threshold: #ratios are too big
                return RatioGoodness.TOO_BIG, abs(diff.min())
                

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
                    startfrom=get_parent_calculation(calc.get_incoming().all_nodes())
                    if len(startfrom) != 1:                    
                        raise RuntimeError('Bug: wrong logic') 
                    newcalc=configure_cp_builder_restart(            
                               self.get_cp_code(),                  
                               startfrom[0],                        
                               resources=self.get_cp_resources_cp(),
                               dt=dt, mu=emass                      
                            )                                            
                    self.to_context(dt_benchmark=append_(self.submit(newcalc)))
                    have_something_to_do=True
                    self.report('[analysis_step.try_to_fix]: trying new dt,emass={},{}'.format(dt,emass))
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
                if try_to_fix(test=lambda emass_, dt_, calc_:  emass_ < too_big_smallest_emass and not calc_.is_finished_ok):
                    return # do it!
                # pick something lower and run it again!
                new_emass = too_big_smallest_emass*3.0/4.0
                dt_big=max(list(res_1[too_big_smallest_emass].keys()))
                new_dt=dt_big*(3.0/4.0)**0.5
                #get a calculation from wich we can start. pick the first one
                startfrom=get_parent_calc_from_emass_dt(res_1,too_big_smallest_emass,dt_big)
                newcalc=configure_cp_builder_restart(           
                                self.get_cp_code(),                  
                                startfrom[0],                        
                                resources=self.get_cp_resources_cp(),
                                dt=new_dt, mu=new_emass                      
                             )                                       
                self.to_context(dt_benchmark=append_(self.submit(newcalc)))  
                self.report('[analysis_step] new emass,dt={},{}'.format(new_emass,new_dt))
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
                startfrom=get_parent_calc_from_emass_dt(res_1,too_small_biggest_emass,dt_small)
                newcalc=configure_cp_builder_restart(           
                                self.get_cp_code(),                  
                                startfrom[0],                        
                                resources=self.get_cp_resources_cp(),
                                dt=new_dt, mu=new_emass                      
                             )                                       
                self.to_context(dt_benchmark=append_(self.submit(newcalc)))  
                self.report('[analysis_step] new emass,dt={},{}'.format(new_emass,new_dt))
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
                startfrom=get_parent_calc_from_emass_dt(res_1,too_small_biggest_emass,dt_small)
                newcalc=configure_cp_builder_restart(           
                                self.get_cp_code(),                  
                                startfrom[0],                        
                                resources=self.get_cp_resources_cp(),
                                dt=new_dt, mu=new_emass                      
                             )                                       
                self.to_context(dt_benchmark=append_(self.submit(newcalc)))  
                self.report('[analysis_step] new emass,dt={},{}'.format(new_emass,new_dt))
                return #do it!
            elif have_garbage:
                #try to decrease the emass (or the timestep)?
                return 405  
            else:
                raise RuntimeError('Bug: wrong logic')
        else:  # we have candidate parameters ( len(candidates!=0) )
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
            if not self.ctx.check1.is_finished_ok:
                raise RuntimeError('Bug: wrong logic')
        else:
            #we started only with a cg calculation
            self.ctx.check1=self.ctx.initial_cg
            #set dt_emass_off
            self.ctx.dt_emass_off=(self.inputs.dt.value, self.inputs.emass.value,0)
       
     

    def nose(self):
        self.report('[nose] beginning.')
        # run the thermostat
        nose_pr_param={
            'CONTROL': {
                'calculation': 'vc-cp',
            },
            'IONS': { 
                'ion_temperature': 'nose',
                'tempw': float(self.inputs.tempw),
                'fnosep': abs(float(self.ctx.vdos_maxs[self.ctx.check1.pk])),
                'nhpcl' : 3,
            },
            'CELL': {
                'cell_dynamics': 'pr',
                'press': float(self.inputs.pressure),
                'cell_dofree': 'volume',
            },
        }
        dt,emass,off=self.ctx.dt_emass_off
        newcalc=configure_cp_builder_restart(                
                self.get_cp_code(),                  
                self.ctx.check1,                        
                resources=self.get_cp_resources_cp(),
                dt=dt, mu=emass,
                additional_parameters=nose_pr_param,
                cmdline=self.ctx.cmdline_cp
             )                                       
        self.to_context(final_nose=self.submit(newcalc))       
        self.report('[nose] sent to context dt,emass,tempw,fnosep,press={},{},{},{},{}'.format(dt,emass,float(self.inputs.tempw),float(self.ctx.vdos_maxs[self.ctx.check1.pk]),float(self.inputs.pressure)))
        return        
   
    def check_nose(self):
        if not self.ctx.final_nose.is_finished_ok:
            return 402
        else:
            self.ctx.check2=self.ctx.final_nose
        return

    
     
    def final_cg(self):
        final_cg=configure_cp_builder_restart(
                   self.get_cp_code(),
                   self.ctx.check2,
                   resources=self.get_cp_resources_cg(),
                   copy_mu_mucut=True,
                   cg=True,
                   nstep=1,
                   remove_parameters_namelist=['CELL'],
                   additional_parameters={'IONS':{'ion_temperature': 'not_controlled'} }            )
        self.to_context(last_nve=append_(self.submit(final_cg))) 
        self.report('[final_cg] cg to context (1 step).')
        return
 
    def check_final_cg(self):
        if not self.ctx.last_nve[-1].is_finished_ok:
            return 403
        return

    def equil_not_ok(self):
        if self.inputs.skip_thermobarostat.value:
            return False
        try:
            calc_to_check=self.ctx.last_nve[-1]
        except:
            return True 
        t=calc_to_check.outputs.output_trajectory.get_array('ionic_temperature')
        tm=t.mean()
        tstd=t.std()
        if abs(tm-float(self.inputs.tempw))<tstd:
            return False
        else:
            return True

    def setup_check2(self):
        if self.inputs.skip_thermobarostat.value:
            self.ctx.check2=self.ctx.check1
            self.ctx.first_prod_nve_idx=0
        else:
            self.ctx.first_prod_nve_idx=len(self.ctx.last_nve)
            self.ctx.check2=self.ctx.last_nve[-1]

    def run_nve(self):
        nve=configure_cp_builder_restart(
                   self.get_cp_code(),
                   self.ctx.last_nve[-1],
                   resources=self.get_cp_resources_cp(),
                   copy_mu_mucut=True,
                   cmdline=self.ctx.cmdline_cp
            )
        self.to_context(last_nve=append_(self.submit(nve))) 
        self.report('[run_nve] nve to context')
        return

    def benchmark_parallelization_options(self):
        resources=self.get_cp_resources_cp()
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
                       cmdline=['-ntg', ntg, '-nb', nb]
                )
            self.to_context(parallel_benchmark=append_(self.submit(nve))) 
            self.report('[benchmark_parallelization_options] nve to context: -nb {} -ntg {}'.format(nb,ntg))
        return

    def benchmark_analysis(self):
        steptime=float('inf')
        cmdline=['-ntg',1,'-nb',1]
        for cp in self.ctx.parallel_benchmark:
            try:
                time,nsteps=main_loop_line(cp)
                r=time/float(nsteps)
                self.report('[benchmark_analysis] {}: {}s per step'.format(cp.inputs.parameters['cmdline'], r))
                if r<steptime:
                    steptime=r
                    cmdline=cp.inputs.parameters['cmdline']
            except Exception as e:
                self.report(e)
        self.ctx.cmdline_cp=cmdline
        self.report('[benchmark_analysis] BEST {}: {}s per step'.format(cmdline, steptime))

    def run_more(self):
        elapsed_simulation_time=get_total_time(self.ctx.last_nve[self.ctx.first_prod_nve_idx:])
        if elapsed_simulation_time < self.inputs.nve_required_picoseconds.value:
            self.ctx.nve_count = self.ctx.nve_count + 1
            self.report('[run_more] nve_count: {}'.format(int(self.ctx.nve_count)))
            return True 
        else:
            self.report('[run_more] simulation finished. Total time {} ps'.format(elapsed_simulation_time))
            return False
        return
 
    def get_result(self):
        self.out('result', self.ctx.last_nve[self.ctx.first_prod_nve_idx])
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

