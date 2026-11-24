from ..lib.periodic_tab_dl.autopnp.AutoPNPBlock import AutoPNPBlock
from ..lib.periodic_tab_dl.base.BaseTabNet import BaseTabNet


class TabAutoPNPNet(BaseTabNet):
    def __init__(
            self,
            continuous_input_size: int,
            categorical_input_size: int,
            num_fourier_features: int,
            max_poly_terms: int,
            hidden_size: int,
            output_size: int,
            use_residual: bool = True,
            poly_type: str = "chebyshev"
    ):
        """
        TabAutoPNPNet extends AutoPNPNet to handle both continuous and categorical tabular data. It applies Fourier
        and Chebyshev transformations to continuous features and processes categorical features through an MLP.

        :param continuous_input_size: Number of continuous input features.
        :param categorical_input_size: Number of one-hot encoded categorical features.
        :param num_fourier_features: Number of Fourier features generated per continuous feature.
        :param max_poly_terms: Number of Chebyshev polynomial terms for continuous feature transformation.
        :param hidden_size: Size of hidden layers for processing categorical features.
        :param output_size: Desired output size; >1 indicates multi-output, 1 for single-output tasks.
        :param use_residual: If True, applies a residual connection to the continuous feature transformations.
        """

        # Initialize AutoPNPBlock for continuous features with Fourier and Chebyshev transformations
        autopnp_layer = AutoPNPBlock(
            input_size=continuous_input_size,
            num_fourier_features=num_fourier_features,
            max_poly_terms=max_poly_terms,
            poly_type=poly_type
        ) if continuous_input_size else None

        # Initialize BaseTabNet with the AutoPNP layer for continuous features and MLP for categorical features
        super(TabAutoPNPNet, self).__init__(
            processing_layer=autopnp_layer,
            categorical_input_size=categorical_input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            use_residual=use_residual
        )
