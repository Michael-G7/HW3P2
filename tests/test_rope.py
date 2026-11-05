import torch.nn as nn
import torch

def test_rope(rope):
    '''
    Test the RotaryPositionalEncoding module.
    Args:
        rope (nn.Module): The rotary positional encoding module.
    '''
    # Structural Test
    test_initialization(rope)
    test_rotate_half(rope)

    # Functional Tests
    test_shapes(rope)

    # Behavioural Tests
    test_norm_preserved(rope)
    test_relative_position(rope)
    test_offset(rope)
    test_max_length(rope)


def test_initialization(rope):
    '''
    Test the rotation tables.
    '''
    print("Testing initialization ...")
    head_dim, max_len = 16, 32
    model = rope(head_dim, max_len)

    for name in ('cos', 'sin'):
        assert hasattr(model, name), f"RotaryPositionalEncoding is missing its {name} table"
        table = getattr(model, name)
        assert name in dict(model.named_buffers()), f"{name} should be registered as a buffer, not a plain attribute"
        assert table.shape == (1, 1, max_len, head_dim), \
            f"{name} shape: expected {(1, 1, max_len, head_dim)} but got {tuple(table.shape)}"

    # Position 0 has angle 0 everywhere, so cos is all ones and sin is all zeros
    assert torch.allclose(model.cos[0, 0, 0], torch.ones(head_dim), atol=1e-5), \
        "cos at position 0 should be all ones"
    assert torch.allclose(model.sin[0, 0, 0], torch.zeros(head_dim), atol=1e-5), \
        "sin at position 0 should be all zeros"

    # Each angle is shared by dimension i and dimension i + head_dim // 2
    half = head_dim // 2
    assert torch.allclose(model.cos[0, 0, :, :half], model.cos[0, 0, :, half:], atol=1e-6), \
        "cos should repeat across the two halves, since dimension i pairs with i + head_dim // 2"

    # An odd head_dim cannot be split into pairs
    try:
        rope(head_dim + 1, max_len)
        raise AssertionError("An odd head_dim should raise a ValueError")
    except ValueError:
        pass

    print("Test Passed: The rotation tables are correct")


def test_rotate_half(rope):
    '''
    Test the half rotation helper.
    '''
    print("Testing rotate_half ...")
    model = rope(4, 8)

    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    expected = torch.tensor([[-3.0, -4.0, 1.0, 2.0]])
    got = model.rotate_half(x)
    assert torch.allclose(got, expected), \
        f"rotate_half([1, 2, 3, 4]) should be [-3, -4, 1, 2] but got {got.tolist()}"

    # Applying it four times returns the original, as with any quarter turn
    y = torch.randn(2, 3, 4)
    assert torch.allclose(model.rotate_half(model.rotate_half(model.rotate_half(model.rotate_half(y)))), y, atol=1e-6), \
        "Applying rotate_half four times should return the original tensor"

    print("Test Passed: rotate_half is correct")


def test_shapes(rope):
    '''
    Test that the queries and keys keep their shape.
    '''
    print("Testing shapes ...")
    head_dim, max_len = 16, 64
    model = rope(head_dim, max_len)

    q = torch.randn(2, 4, 10, head_dim)
    k = torch.randn(2, 4, 10, head_dim)
    q_out, k_out = model(q, k)

    assert q_out.shape == q.shape, f"Query shape: expected {tuple(q.shape)} but got {tuple(q_out.shape)}"
    assert k_out.shape == k.shape, f"Key shape: expected {tuple(k.shape)} but got {tuple(k_out.shape)}"

    print("Test Passed: Shapes are preserved")


def test_norm_preserved(rope):
    '''
    Test that the rotation preserves vector length, which any rotation must.
    '''
    print("Testing that the rotation preserves norms ...")
    head_dim = 32
    model = rope(head_dim, 64)
    q = torch.randn(2, 4, 20, head_dim)
    q_out, _ = model(q, q)

    assert torch.allclose(q.norm(dim=-1), q_out.norm(dim=-1), rtol=1e-4, atol=1e-4), \
        "The rotation changed the vector norms, so it is not a rotation"

    # Position 0 is the identity rotation
    assert torch.allclose(q_out[:, :, 0], q[:, :, 0], atol=1e-5), \
        "Position 0 should be left unchanged"

    print("Test Passed: The rotation preserves norms")


def test_relative_position(rope):
    '''
    Test the property that RoPE exists for: the attention score between two
    positions depends only on the distance between them, not on where they sit.
    '''
    print("Testing the relative position property ...")
    head_dim = 32
    model = rope(head_dim, 128)

    a = torch.randn(1, 1, 1, head_dim)
    b = torch.randn(1, 1, 1, head_dim)

    def score(m, n):
        qa, _ = model(a, a, offset=m)
        kb, _ = model(b, b, offset=n)
        return (qa * kb).sum().item()

    # Same distance, different absolute positions, so the scores should agree
    same_distance = [score(m, m - 2) for m in (5, 12, 40, 99)]
    spread = max(same_distance) - min(same_distance)
    assert spread < 1e-3, \
        f"Scores for the same relative distance differ by {spread:.2e}, so they still depend on absolute position"

    # Different distances should give different scores
    assert abs(score(10, 8) - score(10, 4)) > 1e-3, \
        "Different relative distances gave the same score, so position is not being encoded"

    print("Test Passed: Scores depend only on relative position")


def test_offset(rope):
    '''
    Test that the offset selects the right slice of the tables.
    '''
    print("Testing the position offset ...")
    head_dim = 16
    model = rope(head_dim, 64)

    q = torch.randn(1, 1, 8, head_dim)
    full, _ = model(q, q, offset=0)

    # Rotating positions 3 onwards should match rotating the whole thing and slicing
    tail, _ = model(q[:, :, 3:], q[:, :, 3:], offset=3)
    assert torch.allclose(tail, full[:, :, 3:], atol=1e-5), \
        "Rotating with an offset should match the corresponding slice of the unoffset result"

    print("Test Passed: The position offset is applied correctly")


def test_max_length(rope):
    '''
    Test that exceeding the maximum length raises.
    '''
    print("Testing the maximum length check ...")
    head_dim, max_len = 16, 10
    model = rope(head_dim, max_len)

    q = torch.randn(1, 1, max_len + 1, head_dim)
    try:
        model(q, q)
        raise AssertionError("A sequence longer than max_len should raise a ValueError")
    except ValueError:
        pass

    # The offset counts towards the limit too
    q = torch.randn(1, 1, max_len, head_dim)
    try:
        model(q, q, offset=1)
        raise AssertionError("An offset that pushes past max_len should raise a ValueError")
    except ValueError:
        pass

    print("Test Passed: The maximum length is enforced")


def main():
    """
    Main function to run the rotary positional encoding tests using the testing framework.
    """
    from hw3lib.model import RotaryPositionalEncoding
    from tests.testing_framework import TestingFramework

    framework = TestingFramework(
        test_categories={
            'RotaryPositionalEncoding': [
                {
                    'func': lambda: test_rope(RotaryPositionalEncoding),
                    'description': 'Test the rotary positional encoding'
                }
            ]
        }
    )

    framework.run_tests()
    framework.summarize_results()

if __name__ == '__main__':
    main()
