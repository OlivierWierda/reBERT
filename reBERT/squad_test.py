"""
Test SQuAD dataset loading and processing with Rich pretty printing
"""

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from minimal_model import MinimalHierarchicalBERT

from rich.console import Console
from rich.pretty import pprint
from rich.panel import Panel
from rich.text import Text

console = Console()


def test_squad_loading():
    """Load SQuAD dataset and examine structure"""

    console.print(Panel("🚀 Loading SQuAD Dataset", style="bold blue"))

    # Load small subset for testing
    dataset = load_dataset("squad", split="train[:100]")  # Just 100 examples

    console.print(f"✅ Dataset size: [bold green]{len(dataset)}[/bold green]")
    console.print("📋 Dataset features:")
    pprint(dataset.features)

    # Look at first example
    example = dataset[0]
    console.print(Panel("📄 First Example", style="bold yellow"))

    console.print("[bold]Context:[/bold]")
    context_text = Text(example['context'][:200] + "...")
    context_text.stylize("dim", 200, len(context_text))
    console.print(context_text)

    console.print(f"\n[bold]Question:[/bold] [italic cyan]{example['question']}[/italic cyan]")

    console.print("\n[bold]Answer:[/bold]")
    pprint(example['answers'])

    return dataset


def test_tokenization():
    """Test tokenizing question+context pairs"""

    console.print(Panel("🔤 Testing Tokenization", style="bold magenta"))

    # Load tokenizer (we'll use BERT tokenizer for now)
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    # Sample data
    context = "The quick brown fox jumps over the lazy dog. This is a test sentence."
    question = "What does the fox jump over?"
    answer = "the lazy dog"

    console.print("[bold]Sample Data:[/bold]")
    console.print(f"🦊 Context: [dim]{context}[/dim]")
    console.print(f"❓ Question: [italic]{question}[/italic]")
    console.print(f"✅ Answer: [bold green]{answer}[/bold green]")

    # Tokenize question + context
    inputs = tokenizer(
        question,
        context,
        truncation=True,
        max_length=128,
        padding='max_length',
        return_tensors='pt'
    )

    console.print("\n[bold]Tokenization Results:[/bold]")
    console.print(f"📐 Input IDs shape: [cyan]{inputs['input_ids'].shape}[/cyan]")
    console.print(f"👁️  Attention mask shape: [cyan]{inputs['attention_mask'].shape}[/cyan]")

    # Show first 20 tokens with nice formatting
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])[:20]
    console.print("\n[bold]First 20 tokens:[/bold]")

    # Create a nice token display
    token_text = Text()
    for i, token in enumerate(tokens):
        if token in ['[CLS]', '[SEP]']:
            token_text.append(f"{token} ", style="bold red")
        elif token == '[PAD]':
            token_text.append(f"{token} ", style="dim")
        else:
            token_text.append(f"{token} ", style="white")

    console.print(token_text)

    return inputs


def test_model_with_squad_format():
    """Test our model with SQuAD-formatted input"""

    console.print(Panel("🤖 Testing Model with SQuAD Format", style="bold green"))

    model = MinimalHierarchicalBERT(num_layers=2)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Sample question and context
    question = "What does the fox jump over?"
    context = "The quick brown fox jumps over the lazy dog."

    # Tokenize
    inputs = tokenizer(
        question,
        context,
        truncation=True,
        max_length=64,  # Shorter for display
        padding="max_length",
        return_tensors="pt",
    )

    # Filter inputs to only what our model expects
    model_inputs = {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}

    # Forward pass
    with torch.no_grad():
        outputs = model(**model_inputs)

    console.print("[bold]Model Output:[/bold]")
    console.print(f"🧠 Hidden states shape: [cyan]{outputs['last_hidden_state'].shape}[/cyan]")
    console.print(f"🎯 Start logits shape: [cyan]{outputs['start_logits'].shape}[/cyan]")
    console.print(f"🎯 End logits shape: [cyan]{outputs['end_logits'].shape}[/cyan]")

    # Show predicted positions (for demo)
    start_scores = torch.softmax(outputs["start_logits"], dim=-1)
    end_scores = torch.softmax(outputs["end_logits"], dim=-1)

    predicted_start = torch.argmax(start_scores).item()
    predicted_end = torch.argmax(end_scores).item()

    console.print(f"\n🤔 Model's prediction (untrained):")
    console.print(f"   📍 Start position: [yellow]{predicted_start}[/yellow]")
    console.print(f"   📍 End position: [yellow]{predicted_end}[/yellow]")

    # Show what tokens those correspond to
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    if predicted_start < len(tokens) and predicted_end < len(tokens):
        console.print(f"   🔤 Predicted span: [dim]{' '.join(tokens[predicted_start:predicted_end+1])}[/dim]")

    return outputs


if __name__ == "__main__":
    dataset = test_squad_loading()
    console.print("\n" + "="*60 + "\n")

    inputs = test_tokenization()
    console.print("\n" + "="*60 + "\n")

    outputs = test_model_with_squad_format()