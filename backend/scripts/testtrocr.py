from transformers import TrOCRProcessor

processor = TrOCRProcessor.from_pretrained(
    "microsoft/trocr-large-handwritten",
    use_fast=False,
)

print("Loaded successfully!")
