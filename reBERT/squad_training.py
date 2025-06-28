"""
Basic SQuAD training for hierarchical BERT
"""
import string
import re
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

from minimal_model import MinimalHierarchicalBERT
from rich.console import Console
from rich.panel import Panel

console = Console()


def process_squad_example(example, tokenizer, max_length=128):
    """Convert SQuAD example to model inputs with REAL answer positions"""

    question = example["question"]
    context = example["context"]
    answer_text = example["answers"]["text"][0] if example["answers"]["text"] else ""
    answer_start_char = (
        example["answers"]["answer_start"][0] if example["answers"]["answer_start"] else 0
    )

    # Tokenize with offset mapping to track character-to-token alignment
    inputs = tokenizer(
        question,
        context,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_offsets_mapping=True,  # This is key!
        return_tensors="pt",
    )

    # Get offset mapping (maps each token to character positions)
    offset_mapping = inputs["offset_mapping"][0]  # Remove batch dimension

    # Find answer span in tokens
    answer_start_token = 0
    answer_end_token = 0

    if answer_text:  # Only if we have an answer
        answer_end_char = answer_start_char + len(answer_text)

        # Find which tokens contain the answer span
        for i, (start_char, end_char) in enumerate(offset_mapping):
            # Skip special tokens ([CLS], [SEP]) and padding
            if start_char == 0 and end_char == 0:
                continue

            # Check if this token overlaps with answer start
            if start_char <= answer_start_char < end_char:
                answer_start_token = i

            # Check if this token overlaps with answer end
            if start_char < answer_end_char <= end_char:
                answer_end_token = i
                break

        # Ensure end >= start
        if answer_end_token < answer_start_token:
            answer_end_token = answer_start_token

    # Remove offset_mapping from inputs (model doesn't need it)
    inputs.pop("offset_mapping")

    return {
        "input_ids": inputs["input_ids"].squeeze(0),
        "attention_mask": inputs["attention_mask"].squeeze(0),
        "start_positions": torch.tensor(answer_start_token, dtype=torch.long),
        "end_positions": torch.tensor(answer_end_token, dtype=torch.long),
        # For debugging
        "answer_text": answer_text,
        "found_positions": (answer_start_token, answer_end_token),
    }

def test_data_processing():
    """Test our data processing function"""

    console.print(Panel("🔧 Testing Data Processing", style="bold blue"))

    # Load one example
    dataset = load_dataset("squad", split="train[:1000]")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    example = dataset[0]
    processed = process_squad_example(example, tokenizer)

    console.print(f"✅ Input IDs shape: {processed['input_ids'].shape}")
    console.print(f"✅ Start position: {processed['start_positions'].item()}")
    console.print(f"✅ End position: {processed['end_positions'].item()}")

    return processed


def test_improved_processing():
    """Test improved answer position detection"""

    console.print(Panel("🎯 Testing Improved Answer Detection", style="bold cyan"))

    # Load examples
    dataset = load_dataset("squad", split="train[:20]")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    for i, example in enumerate(dataset):
        console.print(f"\n[bold]Example {i+1}:[/bold]")
        console.print(f"Question: [italic]{example['question']}[/italic]")
        console.print(f"Answer: [green]{example['answers']['text'][0]}[/green]")

        processed = process_squad_example(example, tokenizer)

        start_pos = processed["start_positions"].item()
        end_pos = processed["end_positions"].item()

        console.print(f"Token positions: [{start_pos}, {end_pos}]")

        # Show what tokens we found
        tokens = tokenizer.convert_ids_to_tokens(processed["input_ids"])
        predicted_tokens = tokens[start_pos : end_pos + 1]

        console.print(f"Found tokens: [yellow]{' '.join(predicted_tokens)}[/yellow]")

        # Check if it makes sense
        predicted_text = tokenizer.decode(processed["input_ids"][start_pos : end_pos + 1])
        console.print(f"Decoded span: [dim]{predicted_text}[/dim]")



def simple_training_test():
    """Test basic training loop with SQuAD data"""

    console.print(Panel("🏋️ Testing Training Loop", style="bold green"))

    # Load small dataset
    dataset = load_dataset("squad", split="train[:1000]")  # Just 10 examples
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Process all examples
    processed_data = []
    for example in dataset:
        processed = process_squad_example(example, tokenizer)
        processed_data.append(processed)

    console.print(f"📚 Processed {len(processed_data)} examples")

    # Create model and optimizer
    model = MinimalHierarchicalBERT(num_layers=2)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

    console.print("🚀 Starting training...")

    # Simple training loop
    for epoch in range(25):
        total_loss = 0

        for i, data in enumerate(processed_data):
            optimizer.zero_grad()

            # Forward pass
            outputs = model(
                input_ids=data["input_ids"].unsqueeze(0),  # Add batch dim
                attention_mask=data["attention_mask"].unsqueeze(0),
            )

            # Calculate loss
            start_logits = outputs["start_logits"]
            end_logits = outputs["end_logits"]

            start_loss = nn.CrossEntropyLoss()(start_logits, data["start_positions"].unsqueeze(0))
            end_loss = nn.CrossEntropyLoss()(end_logits, data["end_positions"].unsqueeze(0))
            loss = start_loss + end_loss

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(processed_data)
        console.print(f"📈 Epoch {epoch + 1}: Average Loss = {avg_loss:.4f}")

    console.print("✅ Training test complete!")
    return model


def normalize_answer(s):
    """Normalize answer text for fair comparison"""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_exact_match(prediction, ground_truth):
    """Check if prediction exactly matches ground truth"""
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def compute_f1(prediction, ground_truth):
    """Compute F1 score between prediction and ground truth"""
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()

    # If either is empty, return appropriate score
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)

    common_tokens = Counter(pred_tokens) & Counter(truth_tokens)
    num_common = sum(common_tokens.values())

    if num_common == 0:
        return 0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1


def evaluate_model(model, dataset, tokenizer, max_examples=20):
    """Evaluate model on SQuAD data with proper metrics"""

    console.print(Panel("📊 Model Evaluation", style="bold purple"))

    model.eval()
    exact_matches = []
    f1_scores = []

    # Debug: Check dataset structure
    console.print(f"🔍 Dataset type: {type(dataset)}")
    console.print(f"🔍 Dataset length: {len(dataset)}")
    console.print(f"🔍 First example type: {type(dataset[0])}")
    console.print(
        f"🔍 First example keys: {list(dataset[0].keys()) if hasattr(dataset[0], 'keys') else 'No keys'}"
    )

    console.print(f"🧪 Evaluating on {min(max_examples, len(dataset))} examples...")

    with torch.no_grad():
        for i in range(min(max_examples, len(dataset))):
            example = dataset[i]

            # Debug: Print example structure
            if i == 0:
                console.print(f"🔍 Example structure: {type(example)}")
                if hasattr(example, "keys"):
                    console.print(f"🔍 Example keys: {list(example.keys())}")

            # Process example
            try:
                processed = process_squad_example(example, tokenizer)

                # Get model prediction
                outputs = model(
                    input_ids=processed["input_ids"].unsqueeze(0),
                    attention_mask=processed["attention_mask"].unsqueeze(0),
                )

                # Find predicted answer span
                start_logits = outputs["start_logits"][0]  # Remove batch dim
                end_logits = outputs["end_logits"][0]

                pred_start = torch.argmax(start_logits).item()
                pred_end = torch.argmax(end_logits).item()

                # Ensure valid span
                if pred_end < pred_start:
                    pred_end = pred_start

                # Extract predicted text
                pred_tokens = processed["input_ids"][pred_start : pred_end + 1]
                predicted_answer = tokenizer.decode(pred_tokens, skip_special_tokens=True)

                # Ground truth answer
                ground_truth = example["answers"]["text"][0] if example["answers"]["text"] else ""

                # Calculate metrics
                em = compute_exact_match(predicted_answer, ground_truth)
                f1 = compute_f1(predicted_answer, ground_truth)

                exact_matches.append(em)
                f1_scores.append(f1)

                # Show first few examples
                if i < 3:
                    console.print(f"\n[bold]Example {i+1}:[/bold]")
                    console.print(f"Question: [italic]{example['question'][:80]}...[/italic]")
                    console.print(f"Ground Truth: [green]{ground_truth}[/green]")
                    console.print(f"Predicted: [yellow]{predicted_answer}[/yellow]")
                    console.print(f"EM: [cyan]{em}[/cyan], F1: [cyan]{f1:.3f}[/cyan]")

            except Exception as e:
                console.print(f"❌ Error processing example {i}: {e}")
                continue

    if exact_matches:  # Only calculate if we have results
        exact_match_score = sum(exact_matches) / len(exact_matches) * 100
        f1_score = sum(f1_scores) / len(f1_scores) * 100

        console.print(f"\n🏆 [bold]Final Results:[/bold]")
        console.print(f"   📍 Exact Match: [bold green]{exact_match_score:.1f}%[/bold green]")
        console.print(f"   🎯 F1 Score: [bold blue]{f1_score:.1f}%[/bold blue]")
    else:
        console.print("❌ No successful evaluations")

    return {
        'exact_match': exact_match_score if exact_matches else 0,
        'f1': f1_score if exact_matches else 0,
        'individual_em': exact_matches,
        'individual_f1': f1_scores
    }

if __name__ == "__main__":
    # Skip small tests when doing serious training
    # test_data_processing()
    # console.print("\n" + "=" * 60 + "\n")

    trained_model = simple_training_test()  # This is the main training
    console.print("\n" + "=" * 60 + "\n")

    # test_improved_processing()  # Skip this too
    # console.print("\n" + "="*60 + "\n")

    # Keep the evaluation to see results
    dataset = load_dataset("squad", split="validation[:20]")
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    results = evaluate_model(trained_model, dataset, tokenizer, max_examples=20)