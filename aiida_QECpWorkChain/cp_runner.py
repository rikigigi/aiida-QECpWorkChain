from aiida.engine import WorkChain
from aiida.engine.processes.workchains.workchain import WorkChainSpec
from aiida.engine import (while_, append_, )
from aiida.orm import (Int, Float, Dict)
from aiida.common import (AttributeDict, )
from aiida_quantumespresso.workflows.protocols.utils import (recursive_merge, )
from aiida_quantumespresso.utils.mapping import (wrap_bare_dict_inputs, )

from aiida_quantumespresso.calculations.cp import CpCalculation

import copy

      
class CpBaseWorkChain(WorkChain):
    @classmethod
    def define(cls, spec: WorkChainSpec):
        super().define(spec)
        spec.expose_inputs(CpCalculation, namespace='cp')
        spec.input('nstep_cg', valid_type=(Int), 
                   default = lambda: Int(1),
                   required = False,
                   validator = lambda x, _ : None if x >= 1 else f'the number of step must be positive (got {x})')
        spec.input('nstep', valid_type=(Int), 
                   default = lambda: Int(1),
                   required = False,
                   validator = lambda x, _ : None if x >= 1 else f'the number of step must be positive (got {x})')
        spec.output('trajectory')

        spec.outline(
            cls.setup,
            cls.run_cg,
            while_(cls.run_more)(
                cls.run_cp,
                cls.check_cp
            ),
            cls.prepare_result
        )

    def setup(self):
        #get input for the CP calculation from workchain's inputs
        self.ctx.inputs: AttributeDict = self.exposed_inputs(CpCalculation, 'cp')

        #convert to regular dictionary the input parameters
        self.ctx.inputs.parameters = self.ctx.inputs.parameters.get_dict()

        #set some default values
        self.ctx.inputs.parameters.setdefault('CONTROL', {})
        self.ctx.inputs.parameters.setdefault('ELECTRONS', {})
        self.ctx.inputs.parameters.setdefault('SYSTEM', {})
        self.ctx.inputs.parameters.setdefault('IONS', {})
        self.ctx.inputs.parameters['CONTROL'].setdefault('calculation','cp')
        self.ctx.inputs.parameters['CONTROL'].setdefault('dt',2.0)
        self.ctx.inputs.parameters['CONTROL']['nstep'] = self.inputs.nstep
        self.ctx.inputs.parameters['CONTROL'].setdefault('restart_mode','from_scratch')
        self.ctx.inputs.parameters['CONTROL'].setdefault('tstress',False)
        self.ctx.inputs.parameters['CONTROL'].setdefault('tprnfor',True)
        self.ctx.inputs.parameters['ELECTRONS'].setdefault('emass',25)
        self.ctx.inputs.parameters['ELECTRONS'].setdefault('electron_dynamics','verlet')
        self.ctx.inputs.parameters['ELECTRONS'].setdefault('orthogonalization','ortho')
        self.ctx.inputs.parameters['IONS'].setdefault('ion_dynamics','verlet')

        self.ctx.first_step = -1
        self.ctx.last_step = -1




    def run_cg(self):
        inputs = copy.deepcopy(self.ctx.inputs)

        inputs.parameters = recursive_merge(self.ctx.inputs.parameters, {
                    'CONTROL' : {'nstep' : self.inputs.nstep_cg},
                    'ELECTRONS' : {'electron_dynamics':'cg'}
                })
        
        inputs.parameters = Dict(inputs.parameters)
        job = self.submit(CpCalculation, **inputs)
        self.report(f'launching initialization CG calculation {job.pk}')
        self.to_context(calculations=append_(job))

        return
        

    def run_more(self):
        return int(self.inputs.nstep) - (self.ctx.last_step - self.ctx.first_step) > 0

    def run_cp(self):
        inputs = copy.deepcopy(self.ctx.inputs)

        inputs.parameters = recursive_merge(self.ctx.inputs.parameters, {
                    'CONTROL' : {'restart_mode' : 'restart',
                                 'nstep' : int(self.inputs.nstep) - (self.ctx.last_step - self.ctx.first_step)},
                    'ELECTRONS' : {'electron_dynamics':'verlet'}
                })
        inputs.parent_folder = self.ctx.calculations[-1].outputs.remote_folder
        
        #inputs = wrap_bare_dict_inputs(CpCalculation, inputs)
        inputs.parameters = Dict(inputs.parameters)
        job = self.submit(CpCalculation, **inputs)
        self.report(f'launching CP calculation {job.pk}')
        self.to_context(calculations=append_(job))

        return

    def check_cp(self):
        calc = self.ctx.calculations[-1]
        if calc.is_finished_ok:
            if 'output_trajectory' in calc.outputs:
                t=calc.outputs.output_trajectory
                if 'steps' in t.get_arraynames():
                    if self.ctx.first_step < 0:
                        self.ctx.first_step = t.get_array('steps')[0]
                    self.ctx.last_step = t.get_array('steps')[-1]
                    return
        return 400

    def prepare_result(self):
        self.outputs.trajectory = [ x.outputs.output_trajectory for x in self.ctx.calculations if 'output_trajectory' in x.outputs]
        
