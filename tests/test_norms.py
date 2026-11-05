import torch.nn as nn
import torch

def test_norms(rms_norm):
    '''
    Test the RMSNorm module.
    Args:
        rms_norm (nn.Module): The RMSNorm module.
    '''
    # Structural Test
    test_initialization(rms_norm)

    # Functional Tests
    test_shapes(rms_norm)
    test_values(rms_norm)

    # Behavioural Tests
    test_no_mean_subtraction(rms_norm)
    test_learnable_scale(rms_norm)
    test_mixed_precision(rms_norm)


def test_initialization(rms_norm):
    '''
    Test that the learnable scale exists and starts as ones.
    '''
    print("Testing initialization ...")
    d_model = 16
    model = rms_norm(d_model)

    assert hasattr(model, 'weight'), "RMSNorm is missing its weight"
    assert isinstance(model.weight, nn.Parameter), "weight should be an nn.Parameter so that it is learnable"
    assert model.weight.shape == (d_model,), f"weight shape: expected {(d_model,)} but got {tuple(model.weight.shape)}"
    assert torch.allclose(model.weight, torch.ones(d_model)), "weight should be initialized to ones"

    # RMSNorm has no bias
    assert not hasattr(model, 'bias') or model.bias is None, "RMSNorm should not have a bias"

    print("Test Passed: RMSNorm is initialized correctly")


def test_shapes(rms_norm):
    '''
    Test that the output shape matches the input shape.
    '''
    print("Testing shapes ...")
    d_model = 16
    model = rms_norm(d_model)

    for shape in [(4, d_model), (4, 8, d_model), (2, 3, 8, d_model)]:
        x = torch.randn(*shape)
        assert model(x).shape == x.shape, f"Output shape: expected {shape} but got {tuple(model(x).shape)}"

    print("Test Passed: Output shape matches input shape")


def test_values(rms_norm):
    '''
    Test the normalization against a directly computed reference.
    '''
    print("Testing values ...")
    d_model = 8
    model = rms_norm(d_model)
    x = torch.randn(4, 6, d_model) * 3.0

    expected = x / torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + model.eps)
    assert torch.allclose(model(x), expected, rtol=1e-4, atol=1e-5), \
        "RMSNorm values do not match x / sqrt(mean(x^2) + eps)"

    # Each normalized vector should have unit root mean square
    rms = model(x).pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), rtol=1e-3, atol=1e-3), \
        "Normalized vectors should have a root mean square of 1"

    print("Test Passed: RMSNorm values are correct")


def test_no_mean_subtraction(rms_norm):
    '''
    Test that the mean is not subtracted, which is what separates RMSNorm from LayerNorm.
    '''
    print("Testing that the mean is not subtracted ...")
    d_model = 8
    model = rms_norm(d_model)

    # A vector with a large mean offset. LayerNorm would centre it and return a
    # zero-mean output. RMSNorm should leave the offset in place.
    x = torch.ones(1, 1, d_model) * 5.0
    out = model(x)

    assert not torch.allclose(out.mean(dim=-1), torch.zeros(1, 1), atol=1e-3), \
        "Output is zero-mean, which means the mean was subtracted. RMSNorm should not centre the input"
    assert torch.allclose(out, torch.ones_like(out), rtol=1e-3, atol=1e-3), \
        "A constant input should normalize to a constant output of ones"

    print("Test Passed: The mean is not subtracted")


def test_learnable_scale(rms_norm):
    '''
    Test that the learnable scale is applied per feature.
    '''
    print("Testing the learnable scale ...")
    d_model = 4
    model = rms_norm(d_model)

    with torch.no_grad():
        model.weight.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))

    x = torch.randn(2, 5, d_model)
    base = rms_norm(d_model)
    expected = base(x) * torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert torch.allclose(model(x), expected, rtol=1e-4, atol=1e-5), \
        "The learnable scale should multiply each feature independently"

    # And it should receive gradients
    model(x).sum().backward()
    assert model.weight.grad is not None, "weight did not receive a gradient"

    print("Test Passed: The learnable scale is applied correctly")


def test_mixed_precision(rms_norm):
    '''
    Test that the normalization survives float16 inputs.

    We train under mixed precision, so x reaches this layer as float16. Computing
    mean(x^2) in float16 overflows above |x| = 256, and the failure is silent: the
    sum becomes inf, the square root becomes inf, and x / inf is zero. The layer
    then outputs zeros with no NaN to catch. Computing the statistic in float32
    avoids this.
    '''
    print("Testing mixed precision ...")
    d_model = 256
    model = rms_norm(d_model)

    for scale in [1.0, 50.0, 500.0, 5000.0]:
        x = (torch.randn(4, 8, d_model) * scale).half()
        out = model(x).float()

        assert not torch.isnan(out).any(), f"Output contains NaN at |x| ~ {scale:g}"
        assert not torch.isinf(out).any(), f"Output contains Inf at |x| ~ {scale:g}"

        rms = out.pow(2).mean().sqrt().item()
        assert rms > 0.5, (
            f"Output root mean square collapsed to {rms:.4f} at |x| ~ {scale:g}. "
            "mean(x^2) overflowed in float16. Compute the statistic in float32 and "
            "cast the result back."
        )
        assert abs(rms - 1.0) < 0.15, f"Output root mean square is {rms:.4f} at |x| ~ {scale:g}, expected about 1.0"

    print("Test Passed: RMSNorm is stable under mixed precision")


def main():
    """
    Main function to run the RMSNorm tests using the testing framework.
    """
    from hw3lib.model import RMSNorm
    from tests.testing_framework import TestingFramework

    framework = TestingFramework(
        test_categories={
            'RMSNorm': [
                {
                    'func': lambda: test_norms(RMSNorm),
                    'description': 'Test the RMSNorm module'
                }
            ]
        }
    )

    framework.run_tests()
    framework.summarize_results()

if __name__ == '__main__':
    main()
