# MFGnet.jl Devcontainer

This devcontainer provides a reproducible development environment for MFGnet.jl with:
- Julia 1.10 with pre-instantiated project environment
- Claude Code CLI for AI-assisted development
- Network firewall for controlled external access
- Julia language server and debugging support

## Quick Start

### First Time Setup

1. **Open in VS Code**: Open this project in VS Code
2. **Rebuild Container**: Press `Cmd+Shift+P` (Mac) or `Ctrl+Shift+P` (Windows/Linux)
3. **Select**: "Dev Containers: Rebuild and Reopen in Container"
4. **Wait**: First build takes ~5-10 minutes

The container will:
- Install Julia 1.10 and system dependencies
- Install Node.js and Claude Code CLI
- Instantiate and precompile your Julia environment
- Set up network firewall with allowed domains
- Configure VS Code with Julia extension

### Daily Usage

When you reopen the project, VS Code will automatically start the container. If not:
- `Cmd+Shift+P` → "Dev Containers: Reopen in Container"

## Features

### Julia Environment

**Julia Version**: 1.10 (compatible with your Project.toml requirement of 1.2+)

**Package Management**:
```bash
# Activate project environment (automatic in REPL)
julia> using Pkg; Pkg.activate(".")

# Add new packages
julia> Pkg.add("PackageName")

# Update packages
julia> Pkg.update()

# Precompile after changes
julia> Pkg.precompile()
```

**Threading**: Set automatically via `JULIA_NUM_THREADS=auto`

**Depot Location**: Packages persisted in Docker volume at `~/.julia`

### Claude Code Integration

Claude Code is pre-installed and automatically updates on container start.

**Usage**:
```bash
# Start Claude Code
claude

# Check version
claude --version

# Get help
claude --help
```

Your Claude Code configuration is persisted across container rebuilds.

### Network Firewall

The container includes a network firewall that restricts outbound connections to approved domains:

**Allowed Domains**:
- GitHub (API, web, git) - for package downloads and version control
- Julia package servers (pkg.julialang.org, julialang-s3.julialang.org, juliahub.com)
- Anthropic API (api.anthropic.com) - for Claude Code
- VS Code services (marketplace, updates)
- Localhost (all ports) - for local services, databases, LLMs

**Localhost Access**: The firewall allows **unrestricted localhost traffic on any port**:
- Run local databases, APIs, or services
- Use local LLMs (Ollama on localhost:11434, LM Studio on localhost:1234, etc.)
- No restrictions on inter-process communication

**Adding Custom Domains**:

Edit `.devcontainer/init-firewall.sh` around line 71 and add your domain to the list:
```bash
for domain in \
    "api.anthropic.com" \
    # ... existing domains ...
    "your-custom-domain.com"; do
```

Then rebuild the container.

**Disabling Firewall** (not recommended):

1. Remove `"--cap-add=NET_ADMIN"` and `"--cap-add=NET_RAW"` from `.devcontainer/devcontainer.json`
2. Remove firewall script setup from `.devcontainer/Dockerfile`
3. Rebuild container

### Debugging

The container includes debugging support via the Julia VS Code extension.

**Debug Configurations**:
1. **Run active Julia file** - Execute current file
2. **Run active Julia file with args** - Execute with command-line arguments
3. **Debug Julia tests** - Run test/runtests.jl with debugging

**To Debug**:
1. Set breakpoints (click left of line numbers)
2. Press `F5` or select configuration from Run panel
3. Use debug console, variables, and call stack

### VS Code Extensions

Pre-installed extensions:
- **Claude Code** - AI-powered coding assistant
- **Julia Language** - Julia language server, REPL integration, debugging
- **GitLens** - Git integration
- **IntelliCode** - AI-assisted completions

## File Structure

```
.devcontainer/
├── Dockerfile           # Container image definition
├── devcontainer.json    # VS Code devcontainer configuration
├── init-firewall.sh     # Network firewall script
└── README.md           # This file

.vscode/
├── settings.json       # Julia and editor settings
└── launch.json         # Debugging configurations
```

## Customization

### Changing Julia Version

Edit `.devcontainer/Dockerfile` line 1:
```dockerfile
FROM julia:1.11-bullseye  # Change from 1.10
```

### Adding System Packages

Edit `.devcontainer/Dockerfile` and add to `apt-get install`:
```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    # ... existing packages ...
    your-package-name \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
```

### Adding VS Code Extensions

Edit `.devcontainer/devcontainer.json` in the `extensions` array:
```json
"extensions": [
    "anthropic.claude-code",
    "julialang.language-julia",
    "your-publisher.your-extension"
]
```

### Julia Settings

Edit `.vscode/settings.json` to customize Julia behavior, formatting, REPL, etc.

## Troubleshooting

### Container Build Fails

**"Failed to fetch GitHub IP ranges"**:
- Firewall script runs during build and needs internet access
- Check your network connection
- Temporarily disable corporate proxy/firewall

**Solution**: Comment out firewall section in Dockerfile, build, then uncomment.

### Julia Packages Not Found

```bash
# Reinstantiate environment
julia> using Pkg; Pkg.activate("."); Pkg.instantiate()
```

### Claude Code Not Working

```bash
# Manually update Claude Code
npm update -g @anthropic-ai/claude-code

# Check version
claude --version
```

### Firewall Blocking Required Domain

1. Add domain to `.devcontainer/init-firewall.sh`
2. Rebuild container: `Cmd+Shift+P` → "Dev Containers: Rebuild Container"
3. Verify: `curl https://your-domain.com`

### Julia REPL Not Starting

1. Check Julia extension is installed in container (not just locally)
2. Restart Julia language server: `Cmd+Shift+P` → "Julia: Restart Language Server"
3. Check terminal output for errors

## Best Practices

### Package Management
1. Always activate the project environment: `Pkg.activate(".")`
2. Update `Project.toml` when adding dependencies
3. Commit `Manifest.toml` for reproducibility
4. Run `Pkg.precompile()` after adding packages

### Development Workflow
1. Use Julia REPL for interactive development (`Cmd+Shift+P` → "Julia: Start REPL")
2. Run code in REPL with `Shift+Enter` (execute current line/selection)
3. Use debugger for complex issues
4. Test frequently with `Pkg.test()`

### Container Maintenance
1. Rebuild monthly for security updates
2. Clean Docker volumes if container becomes slow
3. Version control `.devcontainer/` and `.vscode/` for team consistency

## Resources

- [Julia Documentation](https://docs.julialang.org/)
- [VS Code Julia Extension](https://www.julia-vscode.org/)
- [Claude Code Documentation](https://docs.claude.com/claude-code)
- [Flux.jl Documentation](https://fluxml.ai/)
- [MFGnet.jl GitHub](https://github.com/your-org/MFGnet.jl)
