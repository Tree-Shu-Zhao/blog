#!/bin/bash

# Exit on error
set -e

# Print versions for debugging
echo "Node version: $(node --version)"
echo "NPM version: $(npm --version)"

# Install Hugo Extended if not present
HUGO_VERSION="0.139.3"
if ! command -v hugo &> /dev/null; then
    echo "Installing Hugo Extended ${HUGO_VERSION}..."
    wget -q -O hugo.tar.gz "https://github.com/gohugoio/hugo/releases/download/v${HUGO_VERSION}/hugo_extended_${HUGO_VERSION}_linux-amd64.tar.gz"
    tar -xzf hugo.tar.gz
    chmod +x hugo
    export PATH="$PWD:$PATH"
fi

# Verify Hugo installation
echo "Hugo version:"
hugo version

# Clean public directory
rm -rf public

# Build the site with the Vercel production URL
if [ -n "$VERCEL_PROJECT_PRODUCTION_URL" ]; then
    echo "Building for production URL: https://${VERCEL_PROJECT_PRODUCTION_URL}"
    hugo --gc --minify --baseURL "https://${VERCEL_PROJECT_PRODUCTION_URL}"
else
    echo "Building with default baseURL from hugo.yaml"
    hugo --gc --minify
fi

# List the generated files
echo "Generated files in public directory:"
ls -la public/

echo "Build completed successfully!"