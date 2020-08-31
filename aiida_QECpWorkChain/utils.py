import copy
from enum import Enum

import aiida.orm
from aiida.orm import Int, Float, Str, List, Dict, ArrayData, Bool
from aiida.engine import WorkChain, calcfunction, ToContext, append_, while_, if_, return_
from aiida.orm.nodes.data.upf import get_pseudos_from_structure
from aiida.plugins.factories import DataFactory
import numpy as np
import qe_tools.constants as qeunits
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
def get_node(node):
    if isinstance(node, aiida.orm.nodes.Node):
        return node
    else:
        return aiida.orm.load_node(node)

def get_pseudo_from_inputs(submitted):
    return {x[9:] : getattr(submitted.inputs,x)  for x in dir(submitted.inputs) if x[:9]=='pseudos__'  }

###

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


def get_parent_nodetype(node, nodetype, minpk=0):
    '''
        Find in a recursive way all nodetype nodes that are in the input tree of node.
        The recursion is stopped when the node has no inputs or it is a nodetype node,
        otherwise it will go to every parent node. Recursion is stopped if pk is less than minpk
    '''
    if isinstance(node, list):
        res=[]
        for n in node:
            res=res+get_parent_nodetype(n,nodetype,minpk)
        return res
    if not isinstance(node, nodetype):
        incoming=node.get_incoming().all_nodes()
        res=[]
        for i in incoming:
            if i.pk < minpk:
                continue
            res = res + get_parent_nodetype(i,nodetype,minpk)
        return res
    else:
        return [node]
    
def get_parent_calculation(node, minpk=0):
    return get_parent_nodetype(node,aiida.orm.nodes.process.calculation.calcjob.CalcJobNode,minpk)

def get_parent_calcfunction(node, minpk=0):
    return get_parent_nodetype(node,aiida.orm.nodes.process.calculation.calcfunction.CalcFunctionNode,minpk)

def get_parent_trajectory(node, minpk=0):
    return get_parent_nodetype(node,aiida.orm.nodes.data.TrajectoryData,minpk)

def get_atomic_types_and_masks(type_array):
    names=set(type_array)
    masks=[]
    outnames=[]
    for name in set(names):
        masks.append(np.array(type_array)==name)
        outnames.append(name)
    return zip(outnames,masks)



##tools to move repository's files around in the file system (useful when you run out of space)

import shutil
from pathlib import Path

def move_traj_on_scratch(t,new_path='/scratch/rbertoss/', truncate_first=4,dry_run=True,print=print):
    '''
    Given a trajectory t as input, it moves the files in the repository in a new folder new_path.
    Removes first truncate_first elements from the original path, then appends this string to new_path:
    this is the new position of the file. A symbolic link is created in place of the old one.
    If there is a file in the new position, it stops for this particula file. If dry_run, it prints
    only the equivalent unix command to perform the same task, without doing anything.
    '''
    tsize=0
    for n in t.list_object_names():
        path=t._repository._get_base_folder().get_abs_path(n)
        p=Path(path)
        size=p.stat().st_size
        trunc_p=Path(*p.parts[truncate_first:])
        newd=Path(new_path,*trunc_p.parts[:-1])
        print('mkdir -p {}'.format(newd))
        if not dry_run:
            newd.mkdir(parents=True, exist_ok=True)
        newf=Path(*newd.parts,p.parts[-1])
        if newf.exists():
            print('{} exists: not doing nothing'.format(newf))
            continue
        print('mv {} {}'.format(p,newf))
        if not dry_run:
            shutil.move(p, newf)
        print('ln -s {} {}'.format(newf,p))
        if not dry_run:
            p.symlink_to(newf)
        tsize=tsize+size
        print('{}: {}kB'.format(trunc_p,size//1024))
    print('traj size={}MB'.format(tsize//(1024*1024)))
    return tsize


def move_all_outputs(wf,dry_run=True, print=print):
    ''' Moves all trajectories but the output ones in a different folder.
    It uses the default folder settings of move_traj_on_scratch.
    It returns the total size of the files
    '''
    tsize=0
    ts=get_children_trajectory(wf.called)
    outputs_pk_traj=[wf.outputs.__getattr__(n).pk if isinstance(wf.outputs.__getattr__(n),aiida.orm.nodes.TrajectoryData) else None for n in wf.outputs]
    not_moved=[]
    for t in ts:
        if t.pk in outputs_pk_traj:
            print('trajectory {} is in the workchain outputs. Not moving'.format(t.pk))
            not_moved.append(t)
            continue
        if t.pk < wf.pk:
            print('trajectory {} is too old. Not moving'.format(t.pk))
            continue
        tsize=tsize+move_traj_on_scratch(t,dry_run=dry_run,print=print)
    print('total_size={}MB'.format(tsize//(1024*1024)))
    print('not moved: {}'.format(not_moved))
    return tsize


