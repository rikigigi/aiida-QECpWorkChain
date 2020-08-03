#TODO
class CpParamConvergence(WorkChain):
    @classmethod
    def define(cls,spec):
        super().define(spec)
        spec.input('structure',required=True, valid_type=(aiida.orm.nodes.data.StructureData))
        spec.input('pseudo_family',required=True, valid_type=(Str))
        spec.input('cp_code',required=True, valid_type=(aiida.orm.nodes.data.code.Code))
        spec.input('resources',required=True, valid_type=(Dict))
        spec.input('additional_parameters',valid_type=(Dict))
        spec.input('start_parameters',valid_type=(Dict))
        spec.input('end_parameters',valid_type=(Dict))
        spec.input('step_parameters',valid_type=(Dict))
        spec.input('ntest_stability',valid_type=(Int), default= lambda : Int(10))
        spec.output('parameters')

        spec.outline(
            cls.setup,
            cls.stability_test,
            cls.ekin_conv_thr,
            cls.ecut,
            cls.ecut_rho, # this will call small_boxes if necessary
            cls.results
        )

    def setup(self):
        self.ctx.param=self.inputs.end_parameters.get_dict()
        if 'nr1b' in self.ctx.param['SYSTEM']:
            self.ctx.nrb=True
        else:
            self.ctx.nrb=False
    
    def run_calc(self):
        calc=configure_cp_builder_cg(
            self.inputs.cp_code,
            self.inputs.pseudo_family,
            self.inputs.structure,
            42.0,
            self.inputs.resources,
            self.inputs.additional_parameters,
            dt=6.0
         )
        return calc
    
    def stability_test(self):
        for i in range(self.inputs.ntest_stability.value):
            calc=self.run_calc()
            self.to_context(tests=append_(calc))

    def ecut(self):
        pass

    def ecut_rho(self):
        pass

    def small_boxes(self):
        pass

    def ekin_conv_thr(self):
        pass
    
    def results(self):
        pass

