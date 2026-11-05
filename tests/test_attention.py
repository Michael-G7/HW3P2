import torch.nn as nn
import torch

def test_attention(attention):
    '''
    Test the MultiHeadAttention module.
    Args:
        attention (nn.Module): The multi-head attention module.
    '''
    # Structural Test
    test_initialization(attention)

    # Functional Tests
    test_shapes(attention)

    # Behavioural Tests
    test_equivalence_to_torch(attention)
    test_masking(attention)
    test_rope_applied_to_qk_only(attention)
    test_qk_norm(attention)


def test_initialization(attention):
    '''
    Test the projections and the head split.
    '''
    print("Testing initialization ...")
    embed_dim, num_heads = 32, 4
    model = attention(embed_dim, num_heads)

    for name in ('q_proj', 'k_proj', 'v_proj', 'out_proj'):
        assert hasattr(model, name), f"MultiHeadAttention is missing {name}"
        proj = getattr(model, name)
        assert isinstance(proj, nn.Linear), f"{name} should be an nn.Linear"
        assert proj.in_features == embed_dim,  f"{name} in_features: expected {embed_dim} but got {proj.in_features}"
        assert proj.out_features == embed_dim, f"{name} out_features: expected {embed_dim} but got {proj.out_features}"

    assert model.embed_dim == embed_dim, f"embed_dim: expected {embed_dim} but got {model.embed_dim}"
    assert model.num_heads == num_heads, f"num_heads: expected {num_heads} but got {model.num_heads}"
    assert model.head_dim == embed_dim // num_heads, \
        f"head_dim: expected {embed_dim // num_heads} but got {model.head_dim}"

    # embed_dim must divide evenly into heads
    try:
        attention(embed_dim, 5)
        raise AssertionError("An embed_dim not divisible by num_heads should raise a ValueError")
    except ValueError:
        pass

    print("Test Passed: The projections are set up correctly")


def test_shapes(attention):
    '''
    Test the output and attention weight shapes, including when the query and key
    lengths differ as they do in cross-attention.
    '''
    print("Testing shapes ...")
    N, L, S, E, H = 3, 7, 11, 32, 4
    model = attention(E, H)

    q, k, v = torch.randn(N, L, E), torch.randn(N, S, E), torch.randn(N, S, E)
    out, weights = model(q, k, v)

    assert out.shape == (N, L, E), f"Output shape: expected {(N, L, E)} but got {tuple(out.shape)}"
    assert weights.shape == (N, L, S), \
        f"Attention weight shape: expected {(N, L, S)}, averaged over heads, but got {tuple(weights.shape)}"

    # Attention weights are a distribution over the keys
    assert torch.allclose(weights.sum(dim=-1), torch.ones(N, L), rtol=1e-4, atol=1e-4), \
        "Attention weights should sum to 1 over the key dimension"

    print("Test Passed: Shapes are correct")


def test_equivalence_to_torch(attention):
    '''
    Test against nn.MultiheadAttention.

    With rotary embeddings and QK-Norm disabled, this module computes exactly what
    PyTorch's does, so we can copy the weights across and compare directly. Note
    that nn packs the three input projections into one in_proj_weight, so the first
    third is the query projection, the second the key, and the third the value.
    '''
    print("Testing equivalence to nn.MultiheadAttention ...")
    N, L, S, E, H = 3, 7, 11, 32, 4
    model = attention(E, H, dropout=0.0)
    ref   = nn.MultiheadAttention(E, H, dropout=0.0, batch_first=True)

    with torch.no_grad():
        W, b = ref.in_proj_weight, ref.in_proj_bias
        model.q_proj.weight.copy_(W[:E]);        model.q_proj.bias.copy_(b[:E])
        model.k_proj.weight.copy_(W[E:2*E]);     model.k_proj.bias.copy_(b[E:2*E])
        model.v_proj.weight.copy_(W[2*E:]);      model.v_proj.bias.copy_(b[2*E:])
        model.out_proj.weight.copy_(ref.out_proj.weight)
        model.out_proj.bias.copy_(ref.out_proj.bias)
    model.eval(); ref.eval()

    q, k, v = torch.randn(N, L, E), torch.randn(N, S, E), torch.randn(N, S, E)
    key_padding_mask = torch.zeros(N, S, dtype=torch.bool); key_padding_mask[:, -3:] = True
    attn_mask = torch.zeros(L, S, dtype=torch.bool); attn_mask[:, -1] = True

    cases = [
        ("no mask",          {}),
        ("key padding mask", {"key_padding_mask": key_padding_mask}),
        ("attention mask",   {"attn_mask": attn_mask}),
        ("both masks",       {"key_padding_mask": key_padding_mask, "attn_mask": attn_mask}),
    ]
    for name, kwargs in cases:
        with torch.no_grad():
            out, weights = model(q, k, v, **kwargs)
            ref_out, ref_weights = ref(q, k, v, need_weights=True, average_attn_weights=True, **kwargs)
        assert torch.allclose(out, ref_out, rtol=1e-4, atol=1e-5), \
            f"Output does not match nn.MultiheadAttention with {name}. Check the scaling by sqrt(head_dim)."
        assert torch.allclose(weights, ref_weights, rtol=1e-4, atol=1e-5), \
            f"Attention weights do not match nn.MultiheadAttention with {name}"

    print("Test Passed: Matches nn.MultiheadAttention exactly")


def test_masking(attention):
    '''
    Test that masked positions receive no attention.
    '''
    print("Testing masking ...")
    N, L, S, E, H = 2, 6, 8, 16, 2
    model = attention(E, H, dropout=0.0).eval()
    q, k, v = torch.randn(N, L, E), torch.randn(N, S, E), torch.randn(N, S, E)

    to_pad = 3
    key_padding_mask = torch.zeros(N, S, dtype=torch.bool)
    key_padding_mask[:, -to_pad:] = True
    with torch.no_grad():
        _, weights = model(q, k, v, key_padding_mask=key_padding_mask)
    assert torch.all(weights[:, :, -to_pad:] == 0), "Padded key positions should receive zero attention"

    causal = torch.triu(torch.ones(L, L, dtype=torch.bool), diagonal=1)
    with torch.no_grad():
        _, weights = model(q, q, q, attn_mask=causal)
    assert torch.all(weights.triu(diagonal=1) == 0), "Future positions should receive zero attention"

    print("Test Passed: Masking is applied correctly")


def test_rope_applied_to_qk_only(attention):
    '''
    Test that rotary embeddings reach the queries and keys but not the values.
    '''
    print("Testing that rotary embeddings reach only the queries and keys ...")
    from hw3lib.model import RotaryPositionalEncoding
    N, L, E, H = 2, 9, 32, 4
    rope = RotaryPositionalEncoding(E // H, 64)

    model = attention(E, H, dropout=0.0, rope=rope).eval()
    assert model.rope is not None, "The rope argument was not stored"

    x = torch.randn(N, L, E)
    with torch.no_grad():
        with_rope, _ = model(x, x, x)
        model.rope = None
        without_rope, _ = model(x, x, x)
    assert not torch.allclose(with_rope, without_rope, atol=1e-4), \
        "Enabling rotary embeddings did not change the output, so they are not being applied"

    # Rotation is norm preserving, so a value path left unrotated keeps the output
    # magnitude in the same range. A rotation wrongly applied to v would not.
    ratio = with_rope.norm() / without_rope.norm()
    assert 0.5 < ratio < 2.0, \
        f"Output magnitude changed by {ratio:.2f}x, which suggests the rotation was applied to the values as well"

    print("Test Passed: Rotary embeddings reach only the queries and keys")


def test_qk_norm(attention):
    '''
    Test the optional QK-Norm.
    '''
    print("Testing QK-Norm ...")
    from hw3lib.model import RMSNorm
    E, H = 32, 4

    off = attention(E, H, qk_norm=False)
    assert getattr(off, 'q_norm', None) is None, "q_norm should be None when qk_norm is False"

    on = attention(E, H, qk_norm=True)
    assert isinstance(on.q_norm, RMSNorm), "q_norm should be an RMSNorm when qk_norm is True"
    assert isinstance(on.k_norm, RMSNorm), "k_norm should be an RMSNorm when qk_norm is True"
    assert on.q_norm.weight.shape == (E // H,), \
        f"QK-Norm normalizes over head_dim, so its weight should have shape {(E // H,)} but got {tuple(on.q_norm.weight.shape)}"

    x = torch.randn(2, 6, E)
    on.eval()
    with torch.no_grad():
        out, weights = on(x, x, x)
    assert out.shape == x.shape, "QK-Norm changed the output shape"
    assert not torch.isnan(out).any(), "QK-Norm produced NaN"

    print("Test Passed: QK-Norm is applied correctly")


def main():
    """
    Main function to run the multi-head attention tests using the testing framework.
    """
    from hw3lib.model import MultiHeadAttention
    from tests.testing_framework import TestingFramework

    framework = TestingFramework(
        test_categories={
            'MultiHeadAttention': [
                {
                    'func': lambda: test_attention(MultiHeadAttention),
                    'description': 'Test the multi-head attention module'
                }
            ]
        }
    )

    framework.run_tests()
    framework.summarize_results()

if __name__ == '__main__':
    main()
