from TALENT.model.methods.base import Method


class TabAutoPNPNetMethod(Method):
    def __init__(self, args, is_regression):
        super().__init__(args, is_regression)
        assert (args.cat_policy == 'tabr_ohe')

    def construct_model(self, model_config=None):
        if model_config is None:
            model_config = self.args.config['model']
        from TALENT.model.models.tabautopnpnet import TabAutoPNPNet
        self.model = TabAutoPNPNet(
            categorical_input_size=len(self.categories or ()),
            continuous_input_size=self.d_in,
            output_size=self.d_out,
            **model_config
        ).to(self.args.device)
        if self.args.use_float:
            self.model.float()
        else:
            self.model.double()
