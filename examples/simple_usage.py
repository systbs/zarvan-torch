import torch
from zarvan import Zarvan, ZarvanConfig

def main():
    print("--- Testing Zarvan Model ---")

    # 1. Define the model configuration
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
    print(f"Number of parameters: {model.num_parameters() / 1e6:.2f}M")

    # 3. Create some dummy input data
    # A batch of 2 sequences, each with a length of 50 tokens.
    input_ids = torch.randint(0, config.vocab_size, (2, 50)) 

    # 4. Perform a forward pass
    with torch.no_grad():
        outputs = model(input_ids=input_ids)

    # The output is a special object from Hugging Face.
    # We can access the final hidden states (before the classification head).
    last_hidden_state = outputs.last_hidden_state
    
    # NOTE: In our custom Zarvan, we didn't add the logits to the output object.
    # For a classification task, you would typically use the output_head on the last_hidden_state.
    # Let's run the output_head manually here to get logits.
    logits = model.output_head(last_hidden_state)


    print("\n--- I/O Shapes ---")
    print("Input IDs shape:", input_ids.shape)
    print("Last Hidden State shape:", last_hidden_state.shape)
    print("Logits shape:", logits.shape)
    
    # 5. Save and load the model (demonstrates Hugging Face hub compatibility)
    # save_directory = "./saved_model"
    # model.save_pretrained(save_directory)
    # print(f"\nModel saved to {save_directory}")
    
    # loaded_model = Zarvan.from_pretrained(save_directory)
    # loaded_model.eval()
    # print("Model loaded successfully!")
    # with torch.no_grad():
    #     loaded_outputs = loaded_model(input_ids)
    # assert torch.allclose(outputs.last_hidden_state, loaded_outputs.last_hidden_state, atol=1e-5)
    # print("Saved and loaded model outputs match. ✅")


if __name__ == "__main__":
    main()