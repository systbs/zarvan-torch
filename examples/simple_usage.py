import torch
import os
from zarvan import Zarvan, ZarvanConfig

def main():
    print("--- Testing Independent Zarvan Model ---")

    # 1. Define the model configuration using the independent ZarvanConfig
    # This configuration object holds all the architectural parameters.
    config = ZarvanConfig(
        vocab_size=10000,
        embed_dim=256,
        hidden_dim=1024,
        num_heads=4,
        num_layers=6,
        num_classes=2, # for a binary classification task
        max_len=128
    )

    # 2. Instantiate the model from the configuration
    model = Zarvan(config)
    model.eval() # Set to evaluation mode

    print(f"Model created successfully!")
    
    # CORRECTED: This is the proper way to count trainable parameters in PyTorch.
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params / 1e6:.2f}M")

    # 3. Create some dummy input data
    # A batch of 2 sequences, each with a length of 50 tokens.
    input_ids = torch.randint(0, config.vocab_size, (2, 50)) 

    # 4. Perform a forward pass
    with torch.no_grad():
        # CORRECTED: The forward pass is now a direct call.
        # It no longer accepts keyword arguments like 'input_ids='.
        logits = model(input_ids)

    # CORRECTED: The model's output is now a simple torch.Tensor of logits.
    # There is no 'BaseModelOutput' object, and no 'last_hidden_state'.
    # The 'logits' variable already holds the final output.

    print("\n--- I/O Shapes ---")
    print("Input IDs shape:", input_ids.shape)
    # The concept of 'last_hidden_state' is removed as the output is directly logits.
    print("Logits shape:", logits.shape)
    
    # 5. Save and load the model using the new independent methods
    save_directory = "./saved_model"
    model.save_pretrained(save_directory)
    print(f"\nModel saved to '{save_directory}'")
    
    loaded_model = Zarvan.from_pretrained(save_directory)
    # The from_pretrained method now sets the model to eval() by default.
    
    print("Model loaded successfully!")
    with torch.no_grad():
        loaded_logits = loaded_model(input_ids)
    
    # CORRECTED: The assertion now directly compares the output tensors.
    assert torch.allclose(logits, loaded_logits, atol=1e-5)
    print("Saved and loaded model outputs match. ✅")


if __name__ == "__main__":
    main()