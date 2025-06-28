"""
Test functions for hierarchical BERT components
"""

import torch
from layers import split_768_to_384, split_384_to_192, create_192d_transformer, process_all_192d_branches


def test_dimension_splitting():
    """Test that our splitting functions work correctly"""

    # Create dummy input: batch_size=2, seq_len=10, hidden_size=768
    dummy_input = torch.randn(2, 10, 768)
    print(f"Input shape: {dummy_input.shape}")

    # Test 768 -> 2x384 split
    branch_A, branch_B = split_768_to_384(dummy_input)
    print(f"Branch A shape: {branch_A.shape}")
    print(f"Branch B shape: {branch_B.shape}")

    # Test 384 -> 2x192 splits
    sub_A1, sub_A2 = split_384_to_192(branch_A)
    sub_B1, sub_B2 = split_384_to_192(branch_B)

    print(f"Sub-branch shapes: {sub_A1.shape}, {sub_A2.shape}, {sub_B1.shape}, {sub_B2.shape}")

    return sub_A1, sub_A2, sub_B1, sub_B2


def test_192d_transformer():
    """Test that the 192D transformer layer works"""

    # Create dummy 192D input: batch_size=2, seq_len=10, hidden_size=192
    dummy_192d = torch.randn(2, 10, 192)
    print(f"192D input shape: {dummy_192d.shape}")

    # Create transformer layer
    transformer_192d = create_192d_transformer()
    print(f"Transformer created: {type(transformer_192d)}")

    # Process through transformer
    with torch.no_grad():  # No gradients needed for testing
        output = transformer_192d(dummy_192d)
        processed_output = output[0]  # BertLayer returns tuple, first element is hidden states

    print(f"Output shape: {processed_output.shape}")
    print(f"Input and output same shape: {dummy_192d.shape == processed_output.shape}")

    return processed_output


def test_process_all_branches():
    """Test processing all 4x192D branches independently"""

    # Create dummy input and split it
    dummy_input = torch.randn(2, 10, 768)
    branch_A, branch_B = split_768_to_384(dummy_input)
    sub_A1, sub_A2 = split_384_to_192(branch_A)
    sub_B1, sub_B2 = split_384_to_192(branch_B)

    print(f"Input branches: {sub_A1.shape}, {sub_A2.shape}, {sub_B1.shape}, {sub_B2.shape}")

    # Process all branches
    with torch.no_grad():
        processed_A1, processed_A2, processed_B1, processed_B2 = process_all_192d_branches(
            sub_A1, sub_A2, sub_B1, sub_B2
        )

    print(
        f"Processed shapes: {processed_A1.shape}, {processed_A2.shape}, {processed_B1.shape}, {processed_B2.shape}"
    )

    # Check all outputs have correct shape
    expected_shape = torch.Size([2, 10, 192])
    shapes_correct = all(
        tensor.shape == expected_shape
        for tensor in [processed_A1, processed_A2, processed_B1, processed_B2]
    )

    print(f"All shapes correct: {shapes_correct}")

    return processed_A1, processed_A2, processed_B1, processed_B2


if __name__ == "__main__":
    test_dimension_splitting()
    print("\n" + "="*50 + "\n")
    test_192d_transformer()
    print("\n" + "="*50 + "\n")
    test_process_all_branches()