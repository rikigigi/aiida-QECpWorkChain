res='''
#traj=cp.outputs.output_trajectory
def merge_trajectories(t,unique=True):
    """
    Merge trajectories taking only one timestep for each timestep id.
    In case of multiple timesteps with the same id if unique is True,
    take the one of the last trajectory in the list.
    """
    arraynames=t[0].get_arraynames()
    arrays={}
    steps={}
    symbols=t[0].symbols
    for a in arraynames:
        arrays[a]=[]
    for traj in t:
        #traj=c.outputs.output_trajectory
        if not symbols == traj.symbols:
            raise RuntimeError('Wrong symbols: trajectories are not compatible')
        if unique:
            for idx,step in enumerate(traj.get_array('steps')):
                steps.setdefault(step,[]).append((traj,idx))
        else:
            for a in arraynames:
                arrays[a].append(traj.get_array(a))
    if unique:
        sortedkeys=list(steps.keys())
        sortedkeys.sort()
        for stepid in sortedkeys:
            print(stepid)
            for arrkey in arraynames:
                #pick only the last occurence in the trajectory list of each timestep 'stepid'
                arrays[arrkey].append(steps[stepid][-1][0].get_array(arrkey)[steps[stepid][-1][1]])
    res=aiida.orm.nodes.data.array.trajectory.TrajectoryData()
    res.set_attribute('symbols',symbols)
    for a in arraynames:
        if unique:
            res.set_array(a,np.array(arrays[a]))
        else:
            res.set_array(a,np.concatenate(arrays[a]))
    return res
'''
case='''
def merge_many_traj(cplist,unique=True):
    res=None
    trajs=[]
    for cp in cplist:
        if 'output_trajectory' in cp.outputs:
            trajs.append(cp.outputs.output_trajectory)
    return merge_many_trajs(trajs,unique)

def merge_many_trajs(trajs,unique):
    if len(trajs)==0:
        return None
    elif len(trajs)==1:
        return trajs[0]
'''
for i in range(2,40):
    case+='''
    elif len(trajs)=={}:
        if unique:
            return merge_traj{}(*trajs)
        else:
            return concatenate_traj{}(*trajs)
'''.format(i,i,i)
    args=''
    for j in range(2,i+1):
        args+=',t{}'.format(j)
    argsf='t1'+args
    argsl='[t1'+args+']'
    res+='''
@calcfunction
def merge_traj{}({}):
    return merge_trajectories({})

@calcfunction
def concatenate_traj{}({}):
    return merge_trajectories({},unique=False)
'''.format(i,argsf,argsl,i,argsf,argsl)

case+='''
    idx=int(len(trajs)//2)
    return merge_many_trajs([merge_many_trajs(trajs[:idx],unique), merge_many_trajs(trajs[idx:],unique)], unique)
'''
if __name__ == "__main__":
    print(res)
    print(case)
else:
    from aiida.engine import calcfunction
    import aiida.orm
    import numpy as np
    exec(res)
    exec(case)
