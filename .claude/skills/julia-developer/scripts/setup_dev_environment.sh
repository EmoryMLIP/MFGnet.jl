#!/bin/bash
# Julia Development Environment Setup Script
# Sets up Julia, VS Code extensions, and common development tools

set -e

echo "🚀 Setting up Julia development environment..."

# Check if Julia is installed
if ! command -v julia &> /dev/null; then
    echo "❌ Julia is not installed. Please install Julia first:"
    echo "   - macOS: brew install julia"
    echo "   - Linux: https://julialang.org/downloads/"
    echo "   - Or use juliaup: curl -fsSL https://install.julialang.org | sh"
    exit 1
fi

JULIA_VERSION=$(julia --version)
echo "✅ Found Julia: $JULIA_VERSION"

# Install essential Julia packages
echo "📦 Installing essential Julia packages..."
julia -e 'using Pkg
packages = [
    "Revise",           # Auto-reload code changes
    "OhMyREPL",         # Enhanced REPL
    "BenchmarkTools",   # Performance benchmarking
    "ProfileView",      # Profiling visualization
    "Debugger",         # Interactive debugger
    "Infiltrator",      # Lightweight debugging
    "Test",             # Testing framework (stdlib)
    "TestEnv",          # Test environment management
    "LocalRegistry",    # Local package registry
    "PkgTemplates",     # Package generation
]
Pkg.add(packages)
println("✅ Essential packages installed")
'

# Setup Julia startup file for convenience
JULIA_CONFIG_DIR="$HOME/.julia/config"
JULIA_STARTUP="$JULIA_CONFIG_DIR/startup.jl"

mkdir -p "$JULIA_CONFIG_DIR"

if [ ! -f "$JULIA_STARTUP" ] || ! grep -q "Revise" "$JULIA_STARTUP"; then
    echo "📝 Configuring Julia startup file..."
    cat >> "$JULIA_STARTUP" << 'EOF'

# Auto-load Revise for interactive development
try
    using Revise
catch e
    @warn "Revise.jl not available" exception=(e, catch_backtrace())
end

# Enhanced REPL experience
try
    using OhMyREPL
catch e
    @warn "OhMyREPL.jl not available" exception=(e, catch_backtrace())
end
EOF
    echo "✅ Julia startup file configured"
else
    echo "ℹ️  Julia startup file already configured"
fi

# Check for VS Code
if command -v code &> /dev/null; then
    echo "📝 Installing VS Code extensions..."
    code --install-extension julialang.language-julia
    echo "✅ VS Code Julia extension installed"
else
    echo "ℹ️  VS Code not found. Install it manually if needed."
fi

echo ""
echo "✅ Julia development environment setup complete!"
echo ""
echo "Next steps:"
echo "  1. Restart your terminal or run: source ~/.bashrc (or ~/.zshrc)"
echo "  2. Open VS Code and verify Julia extension is active"
echo "  3. Test Julia REPL: julia"
echo "  4. For GPU support, run: julia scripts/gpu_setup_test.jl"
