#!/usr/bin/env bash
# Upload the checkpoints to Hugging Face.
#   pip install -U 'huggingface_hub[cli]'
#   huggingface-cli login
set -euo pipefail
cd "$(dirname "$0")/../.."

# tweet
huggingface-cli repo create bert-tweet-sentiment-3class --type model -y
huggingface-cli upload YOUR-HF-USERNAME/bert-tweet-sentiment-3class models/bert_sentiment_model . \
    --commit-message 'Add tweet checkpoint'
huggingface-cli upload YOUR-HF-USERNAME/bert-tweet-sentiment-3class Report/hf/bert-tweet-sentiment-3class/README.md README.md

# movie
huggingface-cli repo create bert-lora-movie-sentiment --type model -y
huggingface-cli upload YOUR-HF-USERNAME/bert-lora-movie-sentiment models/lora-bert-model . \
    --commit-message 'Add movie checkpoint'
huggingface-cli upload YOUR-HF-USERNAME/bert-lora-movie-sentiment Report/hf/bert-lora-movie-sentiment/README.md README.md

# glasses
huggingface-cli repo create resnet18-glasses-detection --type model -y
huggingface-cli upload YOUR-HF-USERNAME/resnet18-glasses-detection models/resnet18_glasses_v2.pt . \
    --commit-message 'Add glasses checkpoint'
huggingface-cli upload YOUR-HF-USERNAME/resnet18-glasses-detection Report/hf/resnet18-glasses-detection/README.md README.md
