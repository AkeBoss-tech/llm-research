#!/bin/bash
# Setup script to build Python 3.10+ from source on Amarel
# Run this ONCE on a login node or interactive compute node before submitting jobs

set -e  # Exit on error

PYTHON_VERSION="3.10.13"  # You can change this to 3.11.x or 3.12.x if needed
PYTHON_DIR="$HOME/python/$PYTHON_VERSION"
LIBFFI_DIR="$HOME/libffi/3.4.2"

echo "=========================================="
echo "Building Python $PYTHON_VERSION from source"
echo "Installation directory: $PYTHON_DIR"
echo "=========================================="

# Step 1: Install libffi (required for _ctypes module)
echo ""
echo "Step 1: Installing libffi..."
if [ -d "$LIBFFI_DIR" ]; then
    echo "libffi already installed at $LIBFFI_DIR"
else
    echo "Downloading and building libffi..."
    cd /tmp || cd "$HOME"
    wget -q https://github.com/libffi/libffi/releases/download/v3.4.2/libffi-3.4.2.tar.gz
    tar -zxf libffi-3.4.2.tar.gz
    cd libffi-3.4.2
    ./configure --prefix="$LIBFFI_DIR"
    make -j 4
    make install
    cd ..
    rm -rf libffi-3.4.2 libffi-3.4.2.tar.gz
    echo "libffi installed successfully"
fi

# Step 2: Build Python from source
echo ""
echo "Step 2: Building Python $PYTHON_VERSION from source..."
if [ -d "$PYTHON_DIR" ]; then
    echo "Python $PYTHON_VERSION already installed at $PYTHON_DIR"
    echo "To rebuild, remove the directory first: rm -rf $PYTHON_DIR"
else
    echo "Downloading Python source..."
    cd /tmp || cd "$HOME"
    wget -q "https://www.python.org/ftp/python/$PYTHON_VERSION/Python-$PYTHON_VERSION.tgz"
    tar -zxf "Python-$PYTHON_VERSION.tgz"
    cd "Python-$PYTHON_VERSION"
    
    echo "Configuring Python build..."
    ./configure \
        --prefix="$PYTHON_DIR" \
        CXX=g++ \
        --with-ensurepip=install \
        LDFLAGS="-L$LIBFFI_DIR/lib64" \
        CPPFLAGS="-I$LIBFFI_DIR/include" \
        PKG_CONFIG_PATH="$LIBFFI_DIR/lib64/pkgconfig"
    
    echo "Building Python (this may take 10-20 minutes)..."
    make -j 8
    
    echo "Installing Python..."
    make install
    
    cd ..
    rm -rf "Python-$PYTHON_VERSION" "Python-$PYTHON_VERSION.tgz"
    echo "Python installed successfully"
fi

# Step 3: Update .bashrc with Python paths
echo ""
echo "Step 3: Updating ~/.bashrc with Python paths..."
BASHRC_UPDATE="
# Python $PYTHON_VERSION (custom build)
export PATH=\"$PYTHON_DIR/bin:\$PATH\"
export LD_LIBRARY_PATH=\"$PYTHON_DIR/lib:\$LD_LIBRARY_PATH\"
export MANPATH=\"$PYTHON_DIR/share/man:\$MANPATH\"
"

if ! grep -q "Python $PYTHON_VERSION (custom build)" "$HOME/.bashrc" 2>/dev/null; then
    echo "$BASHRC_UPDATE" >> "$HOME/.bashrc"
    echo "Added Python paths to ~/.bashrc"
else
    echo "Python paths already in ~/.bashrc"
fi

# Step 4: Verify installation
echo ""
echo "Step 4: Verifying installation..."
source "$HOME/.bashrc" 2>/dev/null || true
if [ -f "$PYTHON_DIR/bin/python3" ]; then
    "$PYTHON_DIR/bin/python3" --version
    echo ""
    echo "=========================================="
    echo "SUCCESS! Python $PYTHON_VERSION is installed"
    echo "Location: $PYTHON_DIR"
    echo ""
    echo "To use it, run:"
    echo "  source ~/.bashrc"
    echo "  which python3"
    echo "  python3 --version"
    echo "=========================================="
else
    echo "ERROR: Python installation verification failed"
    exit 1
fi

