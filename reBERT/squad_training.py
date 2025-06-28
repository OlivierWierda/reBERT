"""
Basic SQuAD training for hierarchical BERT
"""
import string
import re
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

from minimal_model import MinimalHierarchicalBERT
from rich.console import Console
from rich.panel import Panel

console = Console()


# Configuration
MAX_SEQUENCE_LENGTH = 512  # Increase from 128 to reduce answer truncation

def process_squad_example(example, tokenizer, max_length=MAX_SEQUENCE_LENGTH):
    """Convert SQuAD example to model inputs with REAL answer positions"""

    question = example["question"]
    context = example["context"]
    answer_text = example["answers"]["text"][0] if example["answers"]["text"] else ""
    answer_start_char = (
        example["answers"]["answer_start"][0] if example["answers"]["answer_start"] else 0
    )
    # Check if answer is truncated
    input_text = question + " " + context
    if answer_start_char >= len(input_text):
        console.print(
            f"⚠️  Answer truncated: char {answer_start_char} >= text len {len(input_text)}"
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

        # Debug failing cases
        if answer_start_token == 0 and answer_end_token == 0 and answer_text:
            console.print(f"⚠️  Failed to find: '{answer_text}' at char {answer_start_char}")
            console.print(f"    Context: '{context[:100]}...'")

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
    processed = process_squad_example(example, tokenizer, max_length=MAX_SEQUENCE_LENGTH)

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

        processed = process_squad_example(example, tokenizer, max_length=MAX_SEQUENCE_LENGTH)

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



def simple_training_test(num_examples=1000, num_epochs=25, learning_rate=5e-5, num_layers=2):
    """Test basic training loop with SQuAD data"""

    console.print(Panel(f"🏋️ Training Loop: {num_examples} examples, {num_epochs} epochs", style="bold green"))

    # Load dataset
    dataset = load_dataset("squad", split=f"train[:{num_examples}]")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Process all examples
    processed_data = []
    for example in dataset:
        processed = process_squad_example(example, tokenizer, max_length=MAX_SEQUENCE_LENGTH)
        processed_data.append(processed)

    console.print(f"📚 Processed {len(processed_data)} examples")

    console.print(f"🔍 First 3 training examples:")
    for i in range(min(3, len(processed_data))):
        console.print(
            f"   Ex {i+1}: start={processed_data[i]['start_positions'].item()}, end={processed_data[i]['end_positions'].item()}, answer='{processed_data[i]['answer_text']}'"
        )

    # Create batched dataset
    input_ids = torch.stack([data["input_ids"] for data in processed_data])
    attention_masks = torch.stack([data["attention_mask"] for data in processed_data])
    start_positions = torch.stack([data["start_positions"] for data in processed_data])
    end_positions = torch.stack([data["end_positions"] for data in processed_data])

    dataset = TensorDataset(input_ids, attention_masks, start_positions, end_positions)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    console.print(f"🚀 Created DataLoader: batch_size=8, {len(dataloader)} batches per epoch")

    # Create model and optimizer
    model = MinimalHierarchicalBERT(num_layers=num_layers)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    console.print(f"🚀 Starting training: {num_epochs} epochs, LR={learning_rate}")

    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0

        for batch in dataloader:
            input_ids_batch, attention_mask_batch, start_pos_batch, end_pos_batch = batch

            optimizer.zero_grad()

            # Forward pass
            outputs = model(
                input_ids=input_ids_batch,
                attention_mask=attention_mask_batch,
            )

            # Calculate loss
            start_logits = outputs["start_logits"]
            end_logits = outputs["end_logits"]

            start_loss = nn.CrossEntropyLoss()(start_logits, start_pos_batch)
            end_loss = nn.CrossEntropyLoss()(end_logits, end_pos_batch)
            loss = start_loss + end_loss

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        console.print(f"📈 Epoch {epoch + 1}: Average Loss = {avg_loss:.4f}")

    console.print("✅ Training test complete!")

    # Save the trained model
    model_filename = f"trained_hierarchical_bert_{num_examples}ex_{num_epochs}ep.pth"
    console.print(f"💾 Saving trained model as '{model_filename}'...")
    torch.save(model.state_dict(), model_filename)
    console.print("✅ Model saved!")

    return model


def run_training_experiment(
    train_examples=1000, train_epochs=25, eval_examples=20, learning_rate=5e-5
):
    """Run complete training and evaluation experiment"""

    console.print(
        Panel(
            f"🧪 Training Experiment: {train_examples} train, {train_epochs} epochs, eval on {eval_examples}",
            style="bold cyan",
        )
    )

    # Training
    trained_model = simple_training_test(
        num_examples=train_examples, num_epochs=train_epochs, learning_rate=learning_rate
    )

    console.print("\n" + "=" * 60 + "\n")

    # Evaluation
    dataset = load_dataset("squad", split=f"validation[:{eval_examples}]")
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    results = evaluate_model(trained_model, dataset, tokenizer, max_examples=eval_examples)

    return trained_model, results


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

    console.print(f"🧪 Evaluating on {min(max_examples, len(dataset))} examples...")

    with torch.no_grad():
        for i in range(min(max_examples, len(dataset))):
            example = dataset[i]

            # Process example
            processed = process_squad_example(example, tokenizer, max_length=MAX_SEQUENCE_LENGTH)

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

            # DEBUG: Show what's happening
            if i < 3:
                console.print(f"\n🔍 [bold]Debug Example {i+1}:[/bold]")
                console.print(f"   Predicted start: {pred_start}, end: {pred_end}")

                # Show top predictions
                start_top3 = torch.topk(start_logits, 3)
                end_top3 = torch.topk(end_logits, 3)
                console.print(
                    f"   Start top 3: positions {start_top3.indices.tolist()}, scores {start_top3.values.tolist()}"
                )
                console.print(
                    f"   End top 3: positions {end_top3.indices.tolist()}, scores {end_top3.values.tolist()}"
                )

                # Show tokens around prediction
                all_tokens = tokenizer.convert_ids_to_tokens(processed["input_ids"])
                console.print(f"   Sequence length: {len(all_tokens)}")
                console.print(
                    f"   Token at start pos: '{all_tokens[pred_start]}' (pos {pred_start})"
                )
                console.print(f"   Token at end pos: '{all_tokens[pred_end]}' (pos {pred_end})")

            # Ensure valid span
            if pred_end < pred_start:
                pred_end = pred_start

            # Extract predicted text - be more careful
            if pred_start < len(processed["input_ids"]) and pred_end < len(processed["input_ids"]):
                pred_tokens = processed["input_ids"][pred_start : pred_end + 1]
                predicted_answer = tokenizer.decode(pred_tokens, skip_special_tokens=True).strip()
            else:
                predicted_answer = ""

            # Ground truth answer
            ground_truth = example["answers"]["text"][0] if example["answers"]["text"] else ""

            # DEBUG: Show the comparison
            if i < 3:
                console.print(f"   Ground truth: '{ground_truth}'")
                console.print(f"   Predicted: '{predicted_answer}'")
                console.print(f"   Prediction empty: {len(predicted_answer) == 0}")

            # Calculate metrics
            em = compute_exact_match(predicted_answer, ground_truth)
            f1 = compute_f1(predicted_answer, ground_truth)

            exact_matches.append(em)
            f1_scores.append(f1)

            # Show first few examples
            if i < 3:
                console.print(f"Question: [italic]{example['question'][:80]}...[/italic]")
                console.print(f"Ground Truth: [green]{ground_truth}[/green]")
                console.print(f"Predicted: [yellow]{predicted_answer}[/yellow]")
                console.print(f"EM: [cyan]{em}[/cyan], F1: [cyan]{f1:.3f}[/cyan]")

    # Overall metrics
    if exact_matches:
        exact_match_score = sum(exact_matches) / len(exact_matches) * 100
        f1_score = sum(f1_scores) / len(f1_scores) * 100

        console.print(f"\n🏆 [bold]Final Results:[/bold]")
        console.print(f"   📍 Exact Match: [bold green]{exact_match_score:.1f}%[/bold green]")
        console.print(f"   🎯 F1 Score: [bold blue]{f1_score:.1f}%[/bold blue]")

        return {
            "exact_match": exact_match_score,
            "f1": f1_score,
            "individual_em": exact_matches,
            "individual_f1": f1_scores,
        }
    else:
        console.print("❌ No successful evaluations")
        return {"exact_match": 0, "f1": 0}


if __name__ == "__main__":
    # Easy configuration - change these parameters to experiment!

    # Quick test (SUPAfast)
    # run_training_experiment(train_examples=25, train_epochs=10, eval_examples=25, learning_rate=5e-5)

    # Quick test (fast)
    # run_training_experiment(train_examples=100, train_epochs=25, eval_examples=25, learning_rate=5e-5)

    # Medium experiment (current)
    run_training_experiment(train_examples=1000, train_epochs=50, eval_examples=100, learning_rate=5e-5)

    # Large experiment (uncomment for serious training)
    # run_training_experiment(train_examples=5000, train_epochs=50, eval_examples=100, learning_rate=3e-5)