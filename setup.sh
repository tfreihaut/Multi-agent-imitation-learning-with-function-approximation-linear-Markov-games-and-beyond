#!/bin/bash

PYTHON_VERSION="3.11.9"
VENV_NAME="venv"

# Check for pyenv
if ! command -v pyenv &> /dev/null
then
    echo "❌ pyenv not found. Please install pyenv first."
    exit 1
fi

# Install Python version if not already installed
if ! pyenv versions --bare | grep -q "^${PYTHON_VERSION}$"; then
    echo "⬇️ Installing Python $PYTHON_VERSION..."
    pyenv install "$PYTHON_VERSION"
fi

# Create virtualenv if not exists
if ! pyenv virtualenvs --bare | grep -q "^${VENV_NAME}$"; then
    echo "🐍 Creating virtualenv $VENV_NAME..."
    pyenv virtualenv "$PYTHON_VERSION" "$VENV_NAME"
fi

# Set local version
echo "$VENV_NAME" > .python-version

# Activate virtualenv
echo "✅ Activating virtualenv $VENV_NAME..."
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
pyenv activate "$VENV_NAME"

# Install Python dependencies
echo "📦 Installing Python dependencies..."
~/.pyenv/versions/$VENV_NAME/bin/python -m pip install -r requirements.txt

echo "✅ Setup complete. Virtualenv '$VENV_NAME' is ready."
