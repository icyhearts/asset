hidden_size = 2048
eps = 1e-6
RMSNorm(hidden_size,
        eps,
        var_hidden_size=None,
        cast_x_before_out_mul=False,
        fp32_residual=False
        )
