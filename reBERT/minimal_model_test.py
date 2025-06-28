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
        outputs = model(input_ids)
        embeddings = outputs["last_hidden_state"]

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
        outputs = model(input_ids)
        output = outputs['last_hidden_state']

    print(f"Final output shape: {output.shape}")
    print(f"Expected shape: torch.Size([2, 10, 768])")
    print(f"Shapes match: {output.shape == torch.Size([2, 10, 768])}")

    return output


def test_gradient_flow():
    """Test that gradients flow through hierarchical layers AND QA head"""

    # Create model and dummy input
    model = MinimalHierarchicalBERT(num_layers=2)
    input_ids = torch.randint(0, 1000, (2, 10))

    outputs = model(input_ids)

    # Test gradients for BOTH hidden states AND QA head
    hidden_states = outputs["last_hidden_state"]
    start_logits = outputs["start_logits"]
    end_logits = outputs["end_logits"]

    # Create targets for both
    hidden_target = torch.randn_like(hidden_states)
    start_target = torch.randint(0, 10, (2,))  # Random start positions
    end_target = torch.randint(0, 10, (2,))  # Random end positions

    # Combined loss: hidden states + QA predictions
    hidden_loss = torch.nn.functional.mse_loss(hidden_states, hidden_target)
    start_loss = torch.nn.functional.cross_entropy(start_logits, start_target)
    end_loss = torch.nn.functional.cross_entropy(end_logits, end_target)

    total_loss = hidden_loss + start_loss + end_loss

    # Backward pass
    total_loss.backward()

    # Check gradients exist for all parameters
    missing_grads = []
    for name, param in model.named_parameters():
        if param.grad is None:
            missing_grads.append(name)
            print(f"❌ No gradient for: {name}")
        else:
            print(f"✅ Gradient exists for: {name} (norm: {param.grad.norm():.4f})")

    print(f"Total loss: {total_loss.item():.4f}")
    print(f"Missing gradients: {len(missing_grads)}")

    if len(missing_grads) == 0:
        print("🎉 All parameters have gradients!")


def test_save_load():
    """Test saving and loading model state"""

    # Create model and get initial output
    model1 = MinimalHierarchicalBERT(num_layers=2)
    model1.eval()
    input_ids = torch.randint(0, 1000, (2, 10))

    with torch.no_grad():
        outputs1 = model1(input_ids)
        output1 = outputs1['last_hidden_state']

    # Save model
    torch.save(model1.state_dict(), "temp_model.pth")
    print("✅ Model saved")

    # Create new model and load state
    model2 = MinimalHierarchicalBERT(num_layers=2)
    model2.load_state_dict(torch.load("temp_model.pth"))
    model2.eval()

    with torch.no_grad():
        outputs2 = model2(input_ids)
        output2 = outputs2['last_hidden_state']

    # Check outputs match
    matches = torch.allclose(output1, output2, atol=1e-6)
    print(f"✅ Model loaded, outputs match: {matches}")

    # Cleanup
    import os
    os.remove('temp_model.pth')
    print("✅ Temp file cleaned up")

def test_input_validation():
    """Test model with various input sizes and edge cases"""

    model = MinimalHierarchicalBERT(num_layers=2)
    model.eval()

    test_cases = [
        (1, 5),  # Small batch, short sequence
        (4, 100),  # Normal case
        (1, 512),  # Max length
        (8, 256),  # Large batch
    ]

    for batch_size, seq_len in test_cases:
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        try:
            with torch.no_grad():
                outputs = model(input_ids)
                output = outputs['last_hidden_state']
            print(f"✅ Success: batch={batch_size}, seq={seq_len}, output={output.shape}")
        except Exception as e:
            print(f"❌ Failed: batch={batch_size}, seq={seq_len}, error={e}")

def test_training_setup():
    """Test that model can train (loss decreases, weights update)"""

    model = MinimalHierarchicalBERT(num_layers=2)
    model.train()  # Training mode

    # Dummy training data
    batch_size, seq_len = 4, 32
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    # Random regression targets (same shape as output)
    targets = torch.randn(batch_size, seq_len, 768)

    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    # Store initial weights for comparison
    initial_weight = model.token_embeddings.weight.clone()

    print("Starting training test...")
    losses = []

    for step in range(20):
        optimizer.zero_grad()

        # Forward pass
        outputs = model(input_ids)
        output = outputs['last_hidden_state']
        loss = criterion(output, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if step % 5 == 0:
            print(f"Step {step}: Loss = {loss.item():.4f}")

    # Check if loss decreased
    improved = losses[-1] < losses[0]
    print(f"✅ Loss decreased: {improved} ({losses[0]:.4f} → {losses[-1]:.4f})")

    # Check if weights changed
    weight_changed = not torch.equal(initial_weight, model.token_embeddings.weight)
    print(f"✅ Weights updated: {weight_changed}")


def test_qa_head():
    """Test QA head produces correct output format"""

    model = MinimalHierarchicalBERT(num_layers=2)
    model.eval()

    # Dummy input
    batch_size, seq_len = 2, 20
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))

    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids)

    # Check output structure
    print(f"Output keys: {list(outputs.keys())}")
    print(f"Hidden states shape: {outputs['last_hidden_state'].shape}")
    print(f"Start logits shape: {outputs['start_logits'].shape}")
    print(f"End logits shape: {outputs['end_logits'].shape}")

    # Check expected shapes
    expected_hidden = torch.Size([batch_size, seq_len, 768])
    expected_logits = torch.Size([batch_size, seq_len])

    print(f"✅ Hidden shape correct: {outputs['last_hidden_state'].shape == expected_hidden}")
    print(f"✅ Start logits correct: {outputs['start_logits'].shape == expected_logits}")
    print(f"✅ End logits correct: {outputs['end_logits'].shape == expected_logits}")

    # Sample logit values (should be reasonable floats)
    print(f"Sample start logits: {outputs['start_logits'][0, :5]}")
    print(f"Sample end logits: {outputs['end_logits'][0, :5]}")

if __name__ == "__main__":
    test_embeddings()
    print("\n" + "="*50 + "\n")
    test_full_model()
    print("\n" + "="*50 + "\n")
    test_gradient_flow()
    print("\n" + "="*50 + "\n")
    test_save_load()
    print("\n" + "="*50 + "\n")
    test_input_validation()
    print("\n" + "="*50 + "\n")
    test_training_setup()
    print("\n" + "="*50 + "\n")
    test_qa_head()
