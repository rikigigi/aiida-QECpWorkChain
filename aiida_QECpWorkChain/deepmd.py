from aiida.engine import CalcJob
from aiida.common import exceptions
from aiida.parsers.parser import Parser
from aiida.common.datastructures import CalcInfo, CodeInfo
from aiida.engine import CalcJob

import aiida.orm
from aiida.orm import Int, Float, Str, List, Dict, ArrayData, Bool, TrajectoryData
import json
import numpy as np
import six

class DeepMdTrainCalculation(CalcJob):
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input('traj', required=True, valid_type=(TrajectoryData) )
        spec.input('param', required=True, valid_type=(Dict))
        spec.input('nline_per_set', default=lambda : Int(2000), valid_type=(Int))
        spec.input('metadata.options.parser_name', valid_type=six.string_types, default='DeepMdTrainParser')

    def prepare_for_submission(self,folder):
        #create box.npy, coord.npy, force.npy, energy.npy and stress.npy (if present)
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
            subfoldername='set.{}'.format(str(i).zfill(3))
            subfolder=folder.get_subfolder(subfoldername,create=True)
            def save_sliced_arr(fname,arr):
                with subfolder.open(fname,'b') as handle:
                    np.save(handle,arr[startidx:stopidx].reshape(size,-1) if len(arr.shape)>1 else arr[startidx:stopidx])
            save_sliced_arr('box.npy',box)
            save_sliced_arr('coord.npy',coord)
            save_sliced_arr('energy.npy',energy)
            save_sliced_arr('force.npy',force)
        write_dict=self.inputs.get_dict()
        write_dict['systems']=['.']
        write_dict['set_prefix']='set'
        write_dict['save_ckpt']='model.ckpt'
        write_dict['disp_file']='lcurve.out'
        with folder.open(self.options.input_filename,'w') as handle:
            json.dump(write_dict,handle)
        codeinfo = CodeInfo()
        codeinfo.code_uuid = self.inputs.code.uuid
        codeinfo.stdout_name = self.options.output_filename
        codeinfo.cmdline_params = [ self.options.input_filename ]

        calcinfo = CalcInfo()
        calcinfo.codes_info = [codeinfo]
        calcinfo.local_copy_list = []
        calcinfo.remote_copy_list = []
        #or self.metadata.options.output_filename?
        calcinfo.retrieve_list = [
                    (self.options.output_filename,'.',1),
                    (write_dict['save_ckpt']+'*','.',1),
                    (write_dict['disp_file'],'.',1)
                  ]

        return calcinfo

class DeepMdTrainParser(Parser):
    def parse(self, **kwargs):
        try:
            output_folder = self.retrieved
        except exceptions.NotExistent:
            return self.exit_codes.ERROR_NO_RETRIEVED_FOLDER
        try:
            with output_folder.open('lcurve.out','r') as handle:
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
                arraydata.set_array(h,result[:i])
        self.out('lcurve',arraydata) 

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


