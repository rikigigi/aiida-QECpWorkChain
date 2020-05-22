from aiida.engine import CalcJob
from aiida.common import exceptions
from aiida.parsers.parser import Parser
from aiida.common.datastructures import CalcInfo, CodeInfo
from aiida.engine import CalcJob

import aiida.orm
from aiida.orm import Int, Float, Str, List, Dict, ArrayData, Bool, TrajectoryData, CalcJobNode
import json
import numpy as np
import six

class DeepMdTrainCalculation(CalcJob):
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input('traj', required=False, valid_type=(TrajectoryData), help='If the training is started from scratch, this will create the input file starting from this raw trajectories. The following arrays names must be present: forces, cells, positions and scf_total_energy (the potential energy that the network has to learn). If the training is restarted from a previous calculation, this input is not needed.' )
        spec.input('param', required=True, valid_type=(Dict), help='Input dictionary, as documented in the deepmd package. The parameters that set the names of the various input file are setted by this plugin.')
        spec.input('nline_per_set', default=lambda : Int(2000), valid_type=(Int))
        spec.input('metadata.options.parser_name', valid_type=six.string_types, default='DeepMdTrainParser')
        spec.input('metadata.options.input_filename', valid_type=str, default='aiida.in')
        spec.input('metadata.options.output_filename', valid_type=str, default='aiida.out')
        spec.input('restart_calculation_folder', valid_type=aiida.orm.RemoteData, required=False, help='If provided the training procedure will restart using the data that is present in this folder (you can find the folder in the outputs of the completed calculation, as well as the calculated network weights)')
        spec.output('lcurve',valid_type=ArrayData)
        spec.output('param',valid_type=Dict)
        spec.exit_code(400,'ERROR_NO_TRAINING_DATA','You must provide the training set with a restart or with a trajectory data')
        spec.exit_code(300, 'ERROR_NO_RETRIEVED_FOLDER',
            message='The retrieved folder data node could not be accessed.')
        spec.exit_code(310, 'ERROR_READING_OUTPUT_FILE',
            message='The output file could not be read from the retrieved folder.')
        spec.exit_code(320, 'ERROR_INVALID_OUTPUT',
            message='The output file contains invalid output.')

    def prepare_for_submission(self,folder):
        #create box.npy, coord.npy, force.npy, energy.npy and stress.npy (if present)
        mother_folder=None
        write_dict=self.inputs.param.get_dict()
        if not 'systems' in write_dict:
            write_dict['systems']=['raw']
        if not 'set_prefix' in write_dict:
            write_dict['set_prefix']='set'
        if not 'save_ckpt' in write_dict:
            write_dict['save_ckpt']='model.ckpt'
        if not 'disp_file' in write_dict:
            write_dict['disp_file']='lcurve.out'
        if 'restart_calculation_folder' in self.inputs:
            mother_folder=self.inputs.restart_calculation_folder

        symlink=[]
        copy_list=[]
        cmdline_params=[ self.options.input_filename ]
        if 'traj' in self.inputs:
            force=self.inputs.traj.get_array('forces')
            box=self.inputs.traj.get_array('cells')
            coord=self.inputs.traj.get_array('positions')
            energy=self.inputs.traj.get_array('scf_total_energy')
            #TODO: stress and scf_total_energy generic names for traj arrays
            nstep=force.shape[0]
            nfolders=int(nstep/int(self.inputs.nline_per_set))
            for i in range(nfolders):
                startidx=i*int(self.inputs.nline_per_set)
                stopidx=(i+1)*int(self.inputs.nline_per_set) if i+1<nfolders else nstep
                size=stopidx-startidx
                subfoldername='{}/{}.{}'.format(write_dict['systems'][0],write_dict['set_prefix'],str(i).zfill(3))
                subfolder=folder.get_subfolder(subfoldername,create=True)
                def save_sliced_arr(fname,arr):
                    with subfolder.open(fname,mode='wb',encoding=None) as handle:
                        np.save(handle,arr[startidx:stopidx].reshape(size,-1) if len(arr.shape)>1 else arr[startidx:stopidx])
                save_sliced_arr('box.npy',box)
                save_sliced_arr('coord.npy',coord)
                save_sliced_arr('energy.npy',energy)
                save_sliced_arr('force.npy',force)
            #write types of atoms
            def create_id(traj):
                l=[]; sp={}; nidx=0
                for s in traj.symbols:
                    idx=sp.setdefault(s,nidx)
                    if idx==nidx: nidx = nidx +1
                    l.append(idx)
                return l

            with folder.open(write_dict['systems'][0]+'/type.raw','w') as handle:
                np.savetxt(handle, create_id(self.inputs.traj))
        elif mother_folder is not None:
            #make symlink
            for system in write_dict['systems']:
                symlink.append((mother_folder.computer.uuid, mother_folder.get_remote_path()+'/'+system, system))
            #copy checkpoint data
            copy_list.append((mother_folder.computer.uuid, mother_folder.get_remote_path()+'/'+write_dict['save_ckpt']+'*', '.'))
            write_dict['load_ckpt']=write_dict['save_ckpt']
            write_dict['restart']=True
            #restart commandline option (the only one that is working for real?)
            cmdline_params.append('--restart={}'.format(write_dict['load_ckpt']))
        else:
            #error: where is the data?? 
            return self.exit_codes.ERROR_NO_TRAINING_DATA
        with folder.open(self.options.input_filename,'w') as handle:
            json.dump(write_dict,handle)
        codeinfo = CodeInfo()
        codeinfo.code_uuid = self.inputs.code.uuid
        codeinfo.stdout_name = self.options.output_filename
        codeinfo.cmdline_params =  cmdline_params

        calcinfo = CalcInfo()
        calcinfo.codes_info = [codeinfo]
        calcinfo.local_copy_list = []
        calcinfo.remote_copy_list = copy_list
        calcinfo.remote_symlink_list=symlink

        #or self.metadata.options.output_filename?
        #calcinfo.retrieve_list = [
        #            (self.options.output_filename,'.',1),
        #            (write_dict['save_ckpt']+'*','.',1),
        #            (write_dict['disp_file'],'.',1)
        #          ]
        calcinfo.retrieve_list = [
                    self.options.input_filename,
                    self.options.output_filename,
                    write_dict['save_ckpt']+'*',
                    write_dict['disp_file']
                  ]


        return calcinfo

class DeepMdTrainParser(Parser):
    def parse(self, **kwargs):
        try:
            output_folder = self.retrieved
        except exceptions.NotExistent:
            return self.exit_codes.ERROR_NO_RETRIEVED_FOLDER
        try:
            with output_folder.open(self.node.get_attribute('input_filename'),'r') as handle:
                params=json.load(handle)
            with output_folder.open(params['disp_file'],'r') as handle:
                headers = self.get_headers(handle)
                result = self.parse_lcurve(handle)
        except (OSError, IOError):
            return self.exit_codes.ERROR_READING_OUTPUT_FILE
	

        if result is None:
            return self.exit_codes.ERROR_INVALID_OUTPUT

        arraydata=ArrayData()
        if headers is None or len(headers) != result.shape[1]:
            arraydata.set_array('lcurve.out',result)
        else:
            for i,h in enumerate(headers):
                arraydata.set_array(h,result[:,i])
        self.out('lcurve',arraydata) 
        self.out('param',Dict(dict=params))

    def get_headers(self,filelike):
        filelike.seek(0,0)
        line=filelike.readline()
        line=line.split()
        if line[0]=='#':
            return line[1:]
        else:
            filelike.seek(0,0)
            return None

    def parse_lcurve(self,filelike):
        result = None
        try:
            result = np.loadtxt(filelike, comments='#')
        except:
            pass

        return result


