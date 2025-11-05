#!/usr/bin/env python3
"""
Helper script to remove firewall components from a devcontainer setup.

This script removes the firewall script and updates the devcontainer configuration
to not use network restrictions.
"""

import argparse
import json
import re
import sys
from pathlib import Path


def remove_firewall_from_dockerfile(dockerfile_path: Path) -> bool:
    """
    Remove firewall setup from Dockerfile.

    Args:
        dockerfile_path: Path to the Dockerfile

    Returns:
        True if successful, False otherwise
    """
    if not dockerfile_path.exists():
        print(f"ERROR: Dockerfile not found at {dockerfile_path}")
        return False

    content = dockerfile_path.read_text()

    # Remove the firewall script copy and setup section
    # This is the section that copies and configures init-firewall.sh
    pattern = r'\n# Copy and set up firewall script\n.*?USER node\n'

    new_content = re.sub(pattern, '\n', content, flags=re.DOTALL)

    if new_content == content:
        print("WARNING: Firewall section not found in Dockerfile (may already be removed)")
    else:
        dockerfile_path.write_text(new_content)
        print(f"✓ Removed firewall section from {dockerfile_path}")

    return True


def remove_firewall_from_devcontainer_json(devcontainer_json_path: Path) -> bool:
    """
    Remove NET_ADMIN and NET_RAW capabilities from devcontainer.json.

    Args:
        devcontainer_json_path: Path to devcontainer.json

    Returns:
        True if successful, False otherwise
    """
    if not devcontainer_json_path.exists():
        print(f"ERROR: devcontainer.json not found at {devcontainer_json_path}")
        return False

    with open(devcontainer_json_path, 'r') as f:
        config = json.load(f)

    # Remove runArgs for NET_ADMIN and NET_RAW
    if 'runArgs' in config:
        original_args = config['runArgs'].copy()
        config['runArgs'] = [
            arg for arg in config['runArgs']
            if arg not in ["--cap-add=NET_ADMIN", "--cap-add=NET_RAW"]
        ]

        if config['runArgs'] != original_args:
            # Remove runArgs entirely if empty
            if not config['runArgs']:
                del config['runArgs']

            with open(devcontainer_json_path, 'w') as f:
                json.dump(config, f, indent=2)

            print(f"✓ Removed network capabilities from {devcontainer_json_path}")
        else:
            print("WARNING: Network capabilities not found in devcontainer.json (may already be removed)")
    else:
        print("WARNING: No runArgs found in devcontainer.json")

    return True


def remove_firewall_script(devcontainer_dir: Path) -> bool:
    """
    Remove the init-firewall.sh script.

    Args:
        devcontainer_dir: Path to .devcontainer directory

    Returns:
        True if successful, False otherwise
    """
    firewall_script = devcontainer_dir / "init-firewall.sh"

    if firewall_script.exists():
        firewall_script.unlink()
        print(f"✓ Removed {firewall_script}")
        return True
    else:
        print(f"WARNING: Firewall script not found at {firewall_script} (may already be removed)")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Remove firewall components from a devcontainer setup"
    )
    parser.add_argument(
        "devcontainer_dir",
        type=Path,
        help="Path to the .devcontainer directory"
    )

    args = parser.parse_args()

    devcontainer_dir = args.devcontainer_dir
    if not devcontainer_dir.exists():
        print(f"ERROR: Directory not found: {devcontainer_dir}")
        sys.exit(1)

    if not devcontainer_dir.is_dir():
        print(f"ERROR: Not a directory: {devcontainer_dir}")
        sys.exit(1)

    print("Removing firewall components from devcontainer...\n")

    # Remove firewall script
    remove_firewall_script(devcontainer_dir)

    # Update Dockerfile
    dockerfile_path = devcontainer_dir / "Dockerfile"
    if not remove_firewall_from_dockerfile(dockerfile_path):
        sys.exit(1)

    # Update devcontainer.json
    devcontainer_json_path = devcontainer_dir / "devcontainer.json"
    if not remove_firewall_from_devcontainer_json(devcontainer_json_path):
        sys.exit(1)

    print("\n✓ Successfully removed firewall components!")
    print("You can now rebuild the container without network restrictions.")


if __name__ == "__main__":
    main()
