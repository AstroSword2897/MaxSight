# HuggingFace Token Setup (Optional)

## What's the Warning?

When loading CLIP models from HuggingFace Hub, you may see:

```
Warning: You are sending unauthenticated requests to the HF Hub. 
Please set a HF_TOKEN to enable higher rate limits and faster downloads.
```

**This is harmless** - the models will still load and work fine. However, authenticating gives you:
- ✅ Higher rate limits (fewer throttling issues)
- ✅ Faster downloads
- ✅ Access to private models (if needed)

---

## Option 1: Ignore It (Recommended)

**You can safely ignore this warning.** It's just informational. The models will download and work perfectly fine without authentication.

---

## Option 2: Set HF_TOKEN (Optional)

If you want to eliminate the warning and get better rate limits:

### In Colab:

```python
# Set your HuggingFace token as an environment variable
import os
os.environ["HF_TOKEN"] = "your_token_here"

# Then load models (they'll use the token automatically)
```

### Get Your Token:

1. Go to https://huggingface.co/settings/tokens
2. Click "New token"
3. Name it (e.g., "colab-training")
4. Copy the token
5. Set it in your Colab cell:

```python
import os
os.environ["HF_TOKEN"] = "hf_xxxxxxxxxxxxxxxxxxxxx"  # Your token here
```

### In Local Environment:

```bash
# Set in your shell
export HF_TOKEN="your_token_here"

# Or add to ~/.bashrc or ~/.zshrc for persistence
echo 'export HF_TOKEN="your_token_here"' >> ~/.zshrc
```

---

## Code Already Updated

The code in `ml/retrieval/encoders/global_encoder.py` now automatically uses `HF_TOKEN` if it's set. No code changes needed - just set the environment variable if you want.

---

## Summary

- **Warning is harmless** - models work fine without token
- **Setting HF_TOKEN is optional** - only if you want better rate limits
- **Code already supports it** - just set the environment variable
