from qe_tools.constants import bohr_to_ang, hartree_to_ev, timeau_to_sec

def w3a(o,arr):
    o.write('{} {} {}\n'.format(arr[0],arr[1],arr[2]))
def wn3a(o,arr):
    for i in range(arr.shape[0]):
        w3a(o,arr[i])

def write_cp_trajectory(out_prefix,time,pos,vel,cel,steps=None):
    pos_out=open(out_prefix+'.pos','w')
    vel_out=open(out_prefix+'.vel','w')
    cel_out=open(out_prefix+'.cel','w')
    for i in range(pos.shape[0]):
        header='{} {}\n'.format(steps[i] if steps is not None else i, time[i])
        for p in (pos_out, vel_out, cel_out):
            p.write(header)
        wn3a(pos_out,pos[i]/bohr_to_ang)
        wn3a(vel_out,vel[i]/(bohr_to_ang / (timeau_to_sec * 10**12)))
        wn3a(cel_out,cel[i]/bohr_to_ang)
    pos_out.close()
    vel_out.close()
    cel_out.close()
