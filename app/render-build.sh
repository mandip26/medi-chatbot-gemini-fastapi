#!/bin/bash
# Install system dependencies for PDF processing
apt-get update
apt-get install -y poppler-utils

# Install Python dependencies
pip install -r requirements.txt
