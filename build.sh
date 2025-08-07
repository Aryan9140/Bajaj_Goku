#!/usr/bin/env bash
# build.sh

# Fail on error
apt-get update
apt-get install -y tesseract-ocr


# Install Python dependencies
pip install -r requirements.txt

# (Optional) Install tesseract if you're on a platform that allows apt installs (not available on Render free tier)
# sudo apt-get update && sudo apt-get install -y tesseract-ocr

# For custom fonts/languages, download as needed
