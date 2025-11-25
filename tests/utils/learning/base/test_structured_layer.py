import torch

from utils.data.column import Column
from utils.data.unit import Unit
from utils.learning.base.structured_layer import StructuredLayer


def make_layer(
    input_cols,
    output_cols,
    input_mean=0.0,
    input_std=1.0,
    output_mean=0.0,
    output_std=1.0,
    output_max=5.0,
):
    input_mean_dict = {str(col): input_mean for col in input_cols}
    input_std_dict = {str(col): input_std for col in input_cols}
    output_mean_dict = {str(col): output_mean for col in output_cols}
    output_std_dict = {str(col): output_std for col in output_cols}
    output_max_dict = {str(col): output_max for col in output_cols}

    return StructuredLayer(
        input_cols=input_cols,
        input_stats=(input_mean_dict, input_std_dict),
        output_cols=output_cols,
        output_stats=(output_mean_dict, output_std_dict, output_max_dict),
        backbone_dim=8,
        backbone_depth=1,
        head_dim=4,
        head_depth=1,
    )


def test_normalize_input_applies_stats_and_unsqueezes():
    unit = Unit("metre", "m")
    col = Column("altitude", "alt_norm", "alt_norm", unit)
    layer = make_layer([col], [col], input_mean=1.0, input_std=2.0, output_max=5.0)

    x_dict = {col: torch.tensor([3.0, 5.0], dtype=torch.float32)}

    norm_list = layer.normalize_input(x_dict)

    assert norm_list[0].shape == (2, 1)
    assert torch.allclose(norm_list[0].squeeze(1), torch.tensor([1.0, 2.0]))


def test_denormalize_output_clamps_using_max_ratio():
    unit = Unit("metre", "m")
    col = Column("altitude", "alt_clamp", "alt_clamp", unit)
    layer = make_layer([col], [col], output_mean=0.0, output_std=1.0, output_max=2.0)

    out_norm_dict = {col: torch.tensor([[5.0], [-5.0]], dtype=torch.float32)}

    out = layer.denormalize_output(out_norm_dict)

    expected = torch.tensor([2.4, -2.4])
    assert torch.allclose(out[col], expected)


def test_forward_produces_predictions_for_all_output_columns():
    unit = Unit("metre", "m")
    col_alt = Column("altitude", "alt_fw", "alt_fw", unit)
    col_spd = Column("speed", "spd_fw", "spd_fw", unit)
    layer = make_layer([col_alt, col_spd], [col_alt, col_spd], output_max=10.0)

    batch = torch.tensor(
        [
            [10.0, 20.0],
            [0.5, -1.0],
            [3.3, 0.0],
        ],
        dtype=torch.float32,
    )
    x_dict = {col_alt: batch[:, 0], col_spd: batch[:, 1]}

    out = layer(x_dict)

    assert set(out.keys()) == {col_alt, col_spd}
    assert out[col_alt].shape == (3,)
    assert out[col_spd].shape == (3,)
    assert torch.isfinite(out[col_alt]).all()
    assert torch.isfinite(out[col_spd]).all()
