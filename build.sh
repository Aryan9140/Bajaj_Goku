#!/usr/bin/env bash
set -e

# Install system dependencies (for OCR)
apt-get update
apt-get install -y tesseract-ocr

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt
