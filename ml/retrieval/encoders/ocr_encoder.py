"""OCR Encoder for Multi-Vector Retrieval Encodes OCR text snippets using sentence-transformers."""

import torch
import torch.nn as nn

try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class OCREncoder(nn.Module):
    """OCR encoder for text embeddings. Uses sentence-transformers to embed OCR text snippets."""

    def __init__(
        self, model_name: str = "all-MiniLM-L6-v2", embed_dim: int = 384, max_texts: int = 10
    ):
        super().__init__()

        self.model_name = model_name
        self.embed_dim = embed_dim
        self.max_texts = max_texts

        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.text_encoder = SentenceTransformer(model_name)
                self.use_sentence_transformers = True
            except Exception:
                self.use_sentence_transformers = False
                self.text_encoder = None
        else:
            self.use_sentence_transformers = False
            self.text_encoder = None

        # Fallback: simple text encoder.
        if not self.use_sentence_transformers:
            # Character-level embedding.
            self.char_embedding = nn.Embedding(128, 64)  # ASCII characters.
            self.text_encoder = nn.Sequential(
                nn.LSTM(64, 128, batch_first=True), nn.Linear(128, embed_dim)
            )

    def forward(
        self,
        text_snippets: list[list[str]],  # [B, N_texts] list of text strings.
        text_confidences: torch.Tensor | None = None,  # [B, N_texts].
    ) -> tuple[torch.Tensor, list[list[str]]]:
        """Encode OCR text snippets."""
        B = len(text_snippets)
        device = next(self.parameters()).device if list(self.parameters()) else torch.device("cpu")

        text_embeddings = []
        valid_texts_list = []

        for b in range(B):
            texts = text_snippets[b]

            # Limit to max_texts.
            texts = texts[: self.max_texts]

            # Encode texts.
            if self.use_sentence_transformers and self.text_encoder is not None:
                # Use sentence-transformers (only if it's a SentenceTransformer)
                if hasattr(self.text_encoder, "encode"):
                    emb_tensor = self.text_encoder.encode(  # type: ignore[union-attr]
                        texts, convert_to_tensor=True, device=device
                    )  # [N_texts, embed_dim].
                    if not isinstance(emb_tensor, torch.Tensor):
                        emb_tensor = torch.as_tensor(emb_tensor, device=device)
                else:
                    # Fallback: treat as nn.Sequential (rare; safe path)
                    emb_parts: list[torch.Tensor] = []
                    for text in texts:
                        # Convert text to character indices.
                        chars = [ord(c) % 128 for c in text[:100]]  # Limit length.
                        if not chars:
                            chars = [0]
                        char_tensor = torch.tensor(chars, device=device).unsqueeze(0)
                        char_emb = self.char_embedding(char_tensor)  # [1, len, 64].
                        if isinstance(self.text_encoder, nn.Sequential):
                            lstm_out, _ = self.text_encoder[0](char_emb)
                            text_emb = self.text_encoder[1](lstm_out[:, -1, :])  # [1, embed_dim].
                        else:
                            # Unexpected type, use simple embedding.
                            text_emb = char_emb.mean(dim=1)  # [1, 64].
                            text_emb = nn.Linear(64, self.embed_dim).to(device)(text_emb)
                        emb_parts.append(text_emb.squeeze(0))
                    emb_tensor = (
                        torch.stack(emb_parts)
                        if emb_parts
                        else torch.zeros(1, self.embed_dim, device=device)
                    )
            else:
                # Fallback: character-level encoding.
                emb_parts = []
                if self.text_encoder is None or not isinstance(self.text_encoder, nn.Sequential):
                    # Create fallback encoder if needed.
                    raise RuntimeError(
                        "Fallback encoder not properly initialized. text_encoder is None or wrong type."
                    )

                for text in texts:
                    # Convert text to character indices.
                    chars = [ord(c) % 128 for c in text[:100]]  # Limit length.
                    if not chars:
                        chars = [0]
                    char_tensor = torch.tensor(chars, device=device).unsqueeze(0)
                    char_emb = self.char_embedding(char_tensor)  # [1, len, 64].
                    lstm_out, _ = self.text_encoder[0](char_emb)
                    text_emb = self.text_encoder[1](lstm_out[:, -1, :])  # [1, embed_dim].
                    emb_parts.append(text_emb.squeeze(0))

                emb_tensor = (
                    torch.stack(emb_parts)
                    if emb_parts
                    else torch.zeros(1, self.embed_dim, device=device)
                )

            # Pad to max_texts.
            n_emb = emb_tensor.shape[0]
            if self.max_texts > n_emb:
                padding = torch.zeros(self.max_texts - n_emb, self.embed_dim, device=device)
                emb_tensor = torch.cat([emb_tensor, padding], dim=0)
                texts = texts + [""] * (self.max_texts - n_emb)

            text_embeddings.append(emb_tensor[: self.max_texts])
            valid_texts_list.append(texts[: self.max_texts])

        stacked = torch.stack(text_embeddings)  # [B, max_texts, embed_dim].

        # L2 normalize.
        stacked = nn.functional.normalize(stacked, p=2, dim=2)

        return stacked, valid_texts_list
