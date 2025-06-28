"""
Test functions for minimal hierarchical BERT model
"""

import torch
from minimal_model import MinimalHierarchicalBERT


def test_embeddings():
    """Test that embeddings work correctly"""

    # Create dummy input: batch_size=2, seq_len=10
    input_ids = torch.randint(0, 1000, (2, 10))  # Random token IDs
    print(f"Input IDs shape: {input_ids.shape}")
    print(f"Sample input IDs: {input_ids[0]}")

    # Create model
    model = MinimalHierarchicalBERT()
    print(f"Model created: {type(model)}")

    # Forward pass (embeddings only for now)
    with torch.no_grad():
        embeddings = model(input_ids)

    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Expected shape: torch.Size([2, 10, 768])")
    print(f"Shapes match: {embeddings.shape == torch.Size([2, 10, 768])}")

    return embeddings


def test_full_model():
    """Test complete model: embeddings + hierarchical processing"""

    # Create dummy input
    input_ids = torch.randint(0, 1000, (2, 10))
    print(f"Input IDs shape: {input_ids.shape}")

    # Create model with 2 hierarchical layers
    model = MinimalHierarchicalBERT(num_layers=2)
    print(f"Model created with {len(model.hierarchical_layers)} layers")

    # Forward pass through full model
    with torch.no_grad():
        output = model(input_ids)

    print(f"Final output shape: {output.shape}")
    print(f"Expected shape: torch.Size([2, 10, 768])")
    print(f"Shapes match: {output.shape == torch.Size([2, 10, 768])}")

    return output

if __name__ == "__main__":
    test_embeddings()
    print("\n" + "="*50 + "\n")
    test_full_model()