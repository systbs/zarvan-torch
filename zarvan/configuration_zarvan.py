# zarvan/configuration_zarvan.py

from transformers import PretrainedConfig

class ZarvanConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a `Zarvan` model.
    It is used to instantiate a Zarvan model according to the specified arguments,
    defining the model architecture.
    """
    model_type = "zarvan"

    def __init__(
        self,
        vocab_size=30522,
        embed_dim=256,
        hidden_dim=1024,
        num_heads=4,
        num_layers=6,
        num_classes=2,
        max_len=512,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.max_len = max_len
        super().__init__(**kwargs)