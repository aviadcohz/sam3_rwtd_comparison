"""
Text embedders for contrastive alignment loss.

Two implementations:
1. QwenTextEmbedder: Uses Qwen base-model mean-pooled embeddings (2048-dim).
   Simple but weak separation in embedding space (cosine sim ~0.8 for all pairs).
2. SentenceTextEmbedder: Uses sentence-transformers (all-mpnet-base-v2, 768-dim).
   10x better semantic separation between similar vs dissimilar textures.
   Requires a trainable projection (2048→768) on the model side.

The embeddings are computed once at startup, stored on CPU,
and looked up during training with zero additional forward-pass cost.
"""

import torch
import torch.nn.functional as F


class QwenTextEmbedder:
    """
    Precomputes and caches Qwen base-model hidden-state embeddings (2048-dim).

    Usage:
        embedder = QwenTextEmbedder()
        embedder.precompute(model.qwen, model.processor, all_texture_labels, device)
        emb = embedder["smooth shell surface"]  # (2048,) tensor on CPU
    """

    def __init__(self):
        self.cache: dict[str, torch.Tensor] = {}
        self.embed_dim: int = 0

    @torch.no_grad()
    def precompute(
        self,
        texture_labels: list[str],
        qwen_model=None,
        processor=None,
        device=None,
    ) -> None:
        unique_labels = sorted(set(texture_labels))
        print(f"  QwenTextEmbedder: encoding {len(unique_labels)} unique texture labels...")

        with qwen_model.disable_adapter():
            qwen_model.eval()
            for label in unique_labels:
                inputs = processor.tokenizer(
                    label, return_tensors="pt", add_special_tokens=True,
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = qwen_model(**inputs, output_hidden_states=True)
                hidden = outputs.hidden_states[-1]
                embedding = hidden.mean(dim=1).squeeze(0)
                self.cache[label] = F.normalize(embedding, dim=-1).cpu()

        self.embed_dim = next(iter(self.cache.values())).shape[-1]
        print(f"  QwenTextEmbedder: cached {len(self.cache)} embeddings (dim={self.embed_dim})")

    def __getitem__(self, label: str) -> torch.Tensor:
        if label not in self.cache:
            raise KeyError(f"Texture label '{label}' not in cache.")
        return self.cache[label]

    def get_pair(
        self, label_a: str, label_b: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self[label_a], self[label_b]


class SentenceTextEmbedder:
    """
    Precomputes and caches sentence-transformer embeddings (768-dim).
    Uses all-mpnet-base-v2 which provides 10x better separation between
    similar vs dissimilar texture descriptions compared to Qwen mean-pool.

    Requires a trainable align_projector (2048→768) on the model side
    to project Qwen SEG hidden states into the sentence-transformer space.

    Usage:
        embedder = SentenceTextEmbedder()
        embedder.precompute(texture_labels)
        emb = embedder["smooth shell surface"]  # (768,) tensor on CPU
    """

    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        self.cache: dict[str, torch.Tensor] = {}
        self.embed_dim: int = 0
        self.model_name = model_name

    @torch.no_grad()
    def precompute(
        self,
        texture_labels: list[str],
        **kwargs,  # accepts unused args for API compatibility
    ) -> None:
        from sentence_transformers import SentenceTransformer

        unique_labels = sorted(set(texture_labels))
        print(f"  SentenceTextEmbedder ({self.model_name}): encoding {len(unique_labels)} labels...")

        st_model = SentenceTransformer(self.model_name)

        for label in unique_labels:
            emb = st_model.encode(label, convert_to_tensor=True)
            self.cache[label] = F.normalize(emb, dim=-1).cpu().float()

        self.embed_dim = next(iter(self.cache.values())).shape[-1]
        print(f"  SentenceTextEmbedder: cached {len(self.cache)} embeddings (dim={self.embed_dim})")

        del st_model
        torch.cuda.empty_cache()

    def __getitem__(self, label: str) -> torch.Tensor:
        if label not in self.cache:
            raise KeyError(f"Texture label '{label}' not in cache.")
        return self.cache[label]

    def get_pair(
        self, label_a: str, label_b: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self[label_a], self[label_b]
