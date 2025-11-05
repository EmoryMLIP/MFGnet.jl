---
name: devcontainer-builder
description: This skill should be used when creating or configuring VS Code devcontainers for Python projects with Claude Code integration. It provides templates and tools for building reproducible development environments with controlled network access, debugging capabilities, and package management using uv. Use when setting up new Python projects, configuring AI agent sandboxes, or ensuring consistent development environments across machines.
---

# Devcontainer Builder

## Overview

This skill provides templates and workflows for creating VS Code devcontainers optimized for Python development with Claude Code integration. The devcontainers provide:

1. **Reproducibility and debugging**: Clear Python package ecosystem using uv for speed and consistency, pre-configured VS Code debugging tools (debugpy), and standardized development environment
2. **Optional network restrictions**: Configurable IP rules and firewall to control data and network access for AI agents (when needed)

All containers follow the Claude Code devcontainer pattern documented at https://docs.claude.com/en/docs/claude-code/devcontainer and automatically update Claude Code on rebuild.

## Workflow Decision

**IMPORTANT**: When using this skill, first ask the user whether they need network restrictions:

**Ask the user**: "Do you need network access restrictions (firewall) for this devcontainer? This is useful for AI agent sandboxes or controlling data access, but not needed for most development projects."

**When to recommend the firewall**:
- AI agent sandboxes where you want to control external data access
- Projects with sensitive data that shouldn't leave the container
- Environments where you want to audit/control all network traffic
- Testing environments that should have limited external dependencies

**When to skip the firewall**:
- General development projects
- Projects that need broad internet access (web scraping, API testing, etc.)
- When you're unsure (can always add later)

Based on the user's answer, include or exclude the firewall components in the setup.

## Quick Start

When creating a new devcontainer setup:

1. Ask user about network restrictions (see Workflow Decision above)
2. Copy template files to project root
3. Customize `requirements.txt` for project dependencies
4. If using firewall: Adjust firewall rules in `init-firewall.sh`
5. Build and launch the devcontainer

## Creating a New Devcontainer

### Step 1: Copy Template Files

Copy the devcontainer structure to the project:

```bash
# From project root
cp -r ~/.claude/skills/devcontainer-builder/assets/templates/.devcontainer .
cp -r ~/.claude/skills/devcontainer-builder/assets/templates/.vscode .
cp ~/.claude/skills/devcontainer-builder/assets/templates/requirements.txt .
```

This creates:
- `.devcontainer/Dockerfile` - Container image definition
- `.devcontainer/devcontainer.json` - VS Code devcontainer configuration
- `.devcontainer/init-firewall.sh` - Network firewall script (if using firewall)
- `.vscode/launch.json` - Debugging configurations
- `.vscode/settings.json` - Python and editor settings
- `requirements.txt` - Python package dependencies

**If NOT using firewall**: After copying, remove the firewall components using the helper script:
```bash
python3 ~/.claude/skills/devcontainer-builder/scripts/remove_firewall.py .devcontainer
```

This automatically removes:
- The `init-firewall.sh` script
- Network capabilities (`--cap-add=NET_ADMIN`, `--cap-add=NET_RAW`) from `devcontainer.json`
- Firewall setup section from `Dockerfile`

### Step 2: Customize Python Dependencies

Edit `requirements.txt` to include project-specific packages. The template includes:
- Core scientific computing (numpy, scipy, matplotlib)
- Testing tools (pytest, pytest-cov, debugpy)
- Code quality tools (ruff, black, isort, mypy)
- Jupyter and plotting (jupyter, ipykernel, plotly)

Add additional dependencies below the template packages.

### Step 3: Configure Allowed Domains (Optional)

The firewall restricts network access to:
- GitHub (API, web, git)
- Anthropic API (api.anthropic.com)
- VS Code marketplace
- Sentry, Statsig (telemetry)

To add custom domains, either:

**Option A: Manual edit**
Edit `.devcontainer/init-firewall.sh` and add domains to the `for domain in` list around line 67.

**Option B: Use the helper script**
```bash
python3 ~/.claude/skills/devcontainer-builder/scripts/customize_firewall.py \
    .devcontainer/init-firewall.sh \
    "your-domain.com" "another-domain.org"
```

### Step 4: Build the Devcontainer

1. Open project in VS Code
2. Press `Cmd+Shift+P` (Mac) or `Ctrl+Shift+P` (Windows/Linux)
3. Select "Dev Containers: Rebuild and Reopen in Container"
4. Wait for build to complete (first build takes 5-10 minutes)

The container will:
- Install all system dependencies
- Set up Python 3.11 environment
- Install packages via `uv` (fast)
- Update Claude Code to latest version
- Configure firewall rules
- Set up debugging and testing

## Container Features

### VS Code Extensions

The container comes pre-configured with essential extensions:

**Core development**:
- Claude Code - AI-powered coding assistant
- Python, Pylance - Python language support and IntelliSense
- Jupyter - Notebook support with inline visualization

**Debugging and visualization**:
- debugpy - Python debugging
- Debug Visualizer - MATLAB-like plot viewing during debugging
- Code Runner - Quick script execution

**Code quality**:
- GitLens - Git integration and history
- IntelliCode - AI-assisted code completion
- isort - Import sorting

**Optional** (included if you use them):
- GitHub Copilot and Copilot Chat

All extensions are automatically installed when the container is built.

### Python Environment

- **Python version**: 3.11 (system-wide)
- **Package manager**: uv (installed at `/usr/local/bin/uv`)
- **Installation pattern**: System-wide packages via `uv pip install --system`
- **PYTHONPATH**: Automatically set to `/workspace`

### Debugging Configuration

The container includes pre-configured debugging profiles in `.vscode/launch.json`:

1. **Python: Current File** - Debug the currently open Python file
2. **Python: Current Test File** - Debug current test file with pytest
3. **Python: All Tests** - Debug all tests in the project

To debug:
1. Set breakpoints in code (click left of line number)
2. Press F5 or select debug configuration from Run panel
3. Use debug console, variables panel, and call stack

The debugpy extension is pre-installed and configured to work immediately.

### Image Plotting and Visualization

For MATLAB-like image plotting and debugging capabilities, the template includes:
- **Debug Visualizer** extension (hediet.debug-visualizer) - View plots and data structures while debugging
- Jupyter integration (ipykernel, ipywidgets) - Inline plotting in notebooks
- Interactive plotting libraries (plotly, matplotlib)

**Recommended workflows**:

**During debugging**:
1. Set breakpoints in your code
2. Hover over numpy arrays, matplotlib figures, or data structures
3. The Debug Visualizer automatically shows plots and visualizations in a side panel
4. Works like MATLAB's workspace viewer

**In Jupyter notebooks**:
1. Use `.ipynb` files for interactive development
2. Images and plots display inline automatically
3. Full support for matplotlib, plotly, and other visualization libraries

**In Python scripts**:
1. Use `plt.show()` (matplotlib) or `fig.show()` (plotly) to display plots
2. Plots open in separate windows or browser tabs

### Network Firewall

The firewall configuration (`init-firewall.sh`):
- Runs on container start with NET_ADMIN capabilities
- Blocks all outbound traffic except allowed domains
- **Allows ALL localhost traffic on ANY port** (no restrictions)
- Allows host network communication
- Allows DNS resolution and SSH
- Verifies configuration (blocks example.com, allows github.com)

**Local LLM Support**:
The firewall allows unrestricted localhost traffic, enabling local LLM servers:
- Ollama (typically localhost:11434)
- LM Studio (typically localhost:1234)
- Custom LLM servers on localhost:8080, localhost:8081, or any other port
- Local databases, APIs, and services

AI agents can programmatically access these local services without firewall restrictions.

To manually activate/test firewall:
```bash
sudo /usr/local/bin/init-firewall.sh
```

### Claude Code Integration

The container includes:
- Latest Claude Code CLI (`@anthropic-ai/claude-code`)
- Auto-update on container start via `postStartCommand`
- Persistent config and history via named volumes
- Firewall configured to allow Anthropic API access

The update happens **before** firewall rules are applied, ensuring Claude Code can update even with restricted network access.

## Customization Guide

### Changing Python Version

Edit `.devcontainer/Dockerfile` line 1:
```dockerfile
FROM python:3.12-slim  # Change from 3.11-slim
```

Update symlinks in Dockerfile (lines 45-48) to match new version.

### Adding System Packages

Edit `.devcontainer/Dockerfile` lines 17-42 to add packages to the `apt-get install` command.

Example for adding CUDA support:
```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
  # ... existing packages ...
  nvidia-cuda-toolkit \
  && apt-get clean && rm -rf /var/lib/apt/lists/*
```

### Adding VS Code Extensions

Edit `.devcontainer/devcontainer.json` in the `customizations.vscode.extensions` array:
```json
"extensions": [
  "anthropic.claude-code",
  "ms-python.python",
  "your-publisher.your-extension"
]
```

### Changing Timezone

Edit `.devcontainer/devcontainer.json` build args:
```json
"build": {
  "args": {
    "TZ": "America/New_York"  // Change from America/Los_Angeles
  }
}
```

### Disabling Firewall

To run without firewall restrictions:

1. Remove `"--cap-add=NET_ADMIN", "--cap-add=NET_RAW"` from `.devcontainer/devcontainer.json` runArgs
2. Comment out or remove the firewall script copy/setup section in Dockerfile (lines 136-141)

### Customizing Shell

The default shell is zsh with powerlevel10k theme. To use bash:

Edit `.devcontainer/devcontainer.json`:
```json
"terminal.integrated.defaultProfile.linux": "bash"
```

## Troubleshooting

### Build fails with "Failed to fetch GitHub IP ranges"

The firewall script runs during build. If GitHub is unreachable during build, it fails. This typically happens if:
- You're behind a corporate firewall
- DNS resolution is not working

**Solution**: Temporarily disable firewall setup in Dockerfile, build container, then re-enable.

### Debugger not working

1. Verify debugpy is installed: `pip list | grep debugpy`
2. Check that Python extension is installed in container (not just locally)
3. Ensure `.vscode/launch.json` exists in workspace
4. Try "Developer: Reload Window" in VS Code

### uv command not found

If `uv` is not available:
```bash
# Reinstall uv
curl -LsSf https://astral.sh/uv/install.sh | sh
sudo mv ~/.local/bin/uv /usr/local/bin/uv
```

### Container can't access required domain

If AI agent or development needs access to a domain blocked by firewall:

1. Add domain to allowed list in `init-firewall.sh`
2. Rebuild container
3. Verify with: `curl https://your-domain.com`

### Claude Code outdated despite postStartCommand

Check:
1. `.devcontainer/devcontainer.json` has `"waitFor": "postStartCommand"`
2. Container has network access to npm registry during start
3. Check terminal output for update errors

## Best Practices

### Package Management

1. **Use uv for all installations**: `uv pip install package-name` (faster than pip)
2. **Update requirements.txt**: Keep `requirements.txt` in sync with installed packages
3. **System-wide installation**: Use `uv pip install --system` for reproducibility
4. **Lock dependencies**: Consider using `uv pip freeze > requirements-lock.txt` for exact versions

### Debugging Workflow

1. **Use justMyCode: false**: Template sets this by default to debug into libraries
2. **Leverage integrated terminal**: Debugger uses `integratedTerminal` for better output
3. **Test-driven debugging**: Use "Python: Current Test File" to debug failing tests
4. **Breakpoint conditions**: Right-click breakpoints to add conditions for complex scenarios
5. **Visual debugging**: Hover over variables during debugging to see plots with Debug Visualizer

### Using Local LLMs with AI Agents

The devcontainer firewall allows unrestricted localhost access, making it easy to use local LLMs:

**Setup workflow**:
1. Install local LLM on your host machine (Ollama, LM Studio, etc.)
2. Configure to listen on localhost (e.g., localhost:8080, localhost:11434)
3. AI agents in the container can access via `http://localhost:PORT`

**Common configurations**:
- **Ollama**: Default `localhost:11434`, API compatible with OpenAI format
- **LM Studio**: Default `localhost:1234`, provides OpenAI-compatible API
- **Custom servers**: Use any port, agents access programmatically

**Benefits**:
- No external network access required for AI features
- Keep data private within your local network
- Test AI agents safely in sandboxed environment
- Control costs with local inference

**Example agent configuration**:
```python
import openai
client = openai.OpenAI(
    base_url="http://localhost:11434/v1",  # Ollama
    api_key="not-needed"
)
```

### Firewall Management

1. **Minimal allowed domains**: Only add domains actually needed
2. **Document custom domains**: Add comments in `init-firewall.sh` explaining why domains are allowed
3. **Test restrictions**: Verify firewall blocks unexpected domains
4. **Separate dev/prod**: Use different firewall configs for development vs production containers
5. **Localhost always allowed**: Don't modify localhost rules - needed for debugging, LLMs, and local services

### Container Maintenance

1. **Regular rebuilds**: Rebuild monthly to get security updates
2. **Update Claude Code**: Happens automatically, but verify version with `claude --version`
3. **Clean volumes**: Periodically clean Docker volumes if container becomes slow
4. **Version control**: Commit `.devcontainer/` and `.vscode/` to git for team consistency

## Resources

### assets/templates/

Complete devcontainer template structure ready to copy into projects:
- `.devcontainer/Dockerfile` - Python 3.11 container with uv, Claude Code, debugging tools
- `.devcontainer/devcontainer.json` - VS Code configuration with extensions and settings
- `.devcontainer/init-firewall.sh` - Network firewall with customizable domain whitelist
- `.vscode/launch.json` - Pre-configured debugging profiles for Python and pytest
- `.vscode/settings.json` - Python development settings (formatting, testing, analysis)
- `requirements.txt` - Template Python dependencies for scientific computing

### scripts/

- `customize_firewall.py` - Helper script to add allowed domains to firewall configuration programmatically
- `remove_firewall.py` - Helper script to remove all firewall components from a devcontainer setup
