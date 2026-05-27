#!/usr/bin/env bash
# One-click setup for FedMPT-main.
#
# This script is intentionally "best-effort":
# - It will try to download missing CLIP assets automatically.
# - If network/download is blocked, it will print the exact target paths
#   you should manually place files into.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

echo "[1/4] Installing Python dependencies..."
python -m pip install -r requirements.txt

echo "[2/4] Ensuring CLIP tokenizer BPE vocab exists..."
#
# CLIP tokenizer expects this file:
#   clip/bpe_simple_vocab_16e6.txt.gz  (used by ./clip/simple_tokenizer.py)
# and this repo also has a parallel copy used by ./convclip/simple_tokenizer.py.
#
BPE_URL="https://raw.githubusercontent.com/openai/CLIP/main/clip/bpe_simple_vocab_16e6.txt.gz"
BPE_TARGETS=(
  "$ROOT_DIR/clip/bpe_simple_vocab_16e6.txt.gz"
  "$ROOT_DIR/convclip/bpe_simple_vocab_16e6.txt.gz"
)

download_file() {
  local url="$1"
  local target="$2"
  if command -v curl >/dev/null 2>&1; then
    echo "  - downloading with curl: $url"
    curl -L --fail -o "$target" "$url"
  elif command -v wget >/dev/null 2>&1; then
    echo "  - downloading with wget: $url"
    wget -O "$target" "$url"
  else
    echo "  - neither curl nor wget found; manual download required."
    return 1
  fi
}

for t in "${BPE_TARGETS[@]}"; do
  if [[ -f "$t" ]]; then
    echo "  - found: $(basename "$t")"
    continue
  fi
  echo "  - missing: $t"
  mkdir -p "$(dirname "$t")"
  if ! download_file "$BPE_URL" "$t" 2>/dev/null; then
    echo "    Please manually download:"
    echo "      $BPE_URL"
    echo "    and place it to:"
    echo "      $t"
  fi
done

echo "[3/4] Ensuring CLIP pretrained model weights exist..."
#
# Default backbone is ViT-B/16.
# convclip/clip.py and clip/clip.py download weights into:
#   ~/.cache/clip/ViT-B-16.pt
#
CLIP_CACHE_DIR="${CLIP_CACHE_DIR:-$HOME/.cache/clip}"
WEIGHT_NAME="ViT-B-16.pt"
WEIGHT_PATH="$CLIP_CACHE_DIR/$WEIGHT_NAME"

WEIGHT_URL="https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt"
EXPECTED_SHA256="5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f"

mkdir -p "$CLIP_CACHE_DIR"

if [[ -f "$WEIGHT_PATH" ]]; then
  echo "  - found weight: $WEIGHT_PATH"
  if command -v sha256sum >/dev/null 2>&1; then
    got="$(sha256sum "$WEIGHT_PATH" | awk '{print $1}')"
    if [[ "$got" != "$EXPECTED_SHA256" ]]; then
      echo "    WARNING: SHA256 mismatch (expected $EXPECTED_SHA256, got $got)."
      echo "    Training may fail; re-download from the official URL."
    fi
  fi
else
  echo "  - missing weight: $WEIGHT_PATH"
  echo "    Please download and place the file here:"
  echo "      $WEIGHT_URL"
  echo "      -> $WEIGHT_PATH"
  if ! download_file "$WEIGHT_URL" "$WEIGHT_PATH" 2>/dev/null; then
    echo "    (Auto-download failed; manual download is fine.)"
  fi
fi

echo "[4/4] Setup done."
echo "You can now run:"
echo "  bash run.sh voc 0.001 fedmpt 50 0 2"

