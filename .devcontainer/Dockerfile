# Use official Julia image as base
FROM julia:1.10-bullseye

# Set timezone
ARG TZ=America/New_York
ENV TZ=${TZ}

# Install system dependencies and Node.js for Claude Code
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Essential build tools
    build-essential \
    ca-certificates \
    curl \
    git \
    wget \
    # Network tools for firewall setup
    iptables \
    iputils-ping \
    dnsutils \
    net-tools \
    iproute2 \
    # Shell and utilities
    zsh \
    vim \
    sudo \
    ssh \
    tzdata \
    # Node.js for Claude Code
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Create non-root user (vscode)
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Install Claude Code globally
RUN npm install -g @anthropic-ai/claude-code

# Install oh-my-zsh for better terminal experience
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended \
    && chsh -s /bin/zsh $USERNAME

# Copy firewall initialization script
COPY .devcontainer/init-firewall.sh /usr/local/bin/init-firewall.sh
RUN chmod +x /usr/local/bin/init-firewall.sh

# Set working directory
WORKDIR /workspace

# Create Julia directory with proper ownership for the vscode user
RUN mkdir -p /home/$USERNAME/.julia && \
    chown -R $USERNAME:$USERNAME /home/$USERNAME/.julia

# Switch to non-root user
USER $USERNAME

# Set Julia depot path to use workspace
ENV JULIA_DEPOT_PATH="/home/$USERNAME/.julia:/usr/local/share/julia"

# Default command
CMD ["/bin/zsh"]
