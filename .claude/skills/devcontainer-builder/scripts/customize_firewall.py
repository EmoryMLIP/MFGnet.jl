#!/usr/bin/env python3
"""
Helper script to customize the allowed domains in the firewall configuration.

This script modifies the init-firewall.sh script to include user-specified domains.
"""

import argparse
import re
import sys
from pathlib import Path


def customize_firewall(firewall_script_path: Path, domains: list[str]) -> None:
    """
    Customize the firewall script to include specified domains.

    Args:
        firewall_script_path: Path to the init-firewall.sh script
        domains: List of domains to allow in the firewall
    """
    if not firewall_script_path.exists():
        print(f"ERROR: Firewall script not found at {firewall_script_path}")
        sys.exit(1)

    # Read the current script
    content = firewall_script_path.read_text()

    # Find the domain list section
    pattern = r'(for domain in \\\n)(.*?)(\n    "update\.code\.visualstudio\.com";)'

    # Build new domain list
    domain_lines = []
    for domain in domains:
        # Validate domain format (basic check)
        if not re.match(r'^[a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?)*$', domain):
            print(f"WARNING: Domain '{domain}' may not be valid")
        domain_lines.append(f'    "{domain}" \\')

    # Add default domains if not already included
    default_domains = [
        "api.anthropic.com",
        "sentry.io",
        "statsig.anthropic.com",
        "statsig.com",
        "marketplace.visualstudio.com",
        "vscode.blob.core.windows.net",
        "update.code.visualstudio.com"
    ]

    # Combine user domains with defaults (remove duplicates)
    all_domains = list(dict.fromkeys(domains + default_domains))

    # Build the new domain list section
    new_domain_section = "\\\n".join([f'    "{d}"' for d in all_domains[:-1]])
    new_domain_section += f' \\\n    "{all_domains[-1]}";'

    # Replace in content
    new_content = re.sub(
        r'for domain in \\\n.*?update\.code\.visualstudio\.com";',
        f'for domain in \\\n{new_domain_section}',
        content,
        flags=re.DOTALL
    )

    # Write back
    firewall_script_path.write_text(new_content)
    print(f"Successfully updated {firewall_script_path}")
    print(f"Added {len(domains)} custom domain(s)")


def main():
    parser = argparse.ArgumentParser(
        description="Customize allowed domains in the devcontainer firewall"
    )
    parser.add_argument(
        "firewall_script",
        type=Path,
        help="Path to the init-firewall.sh script"
    )
    parser.add_argument(
        "domains",
        nargs="+",
        help="Domains to allow through the firewall"
    )

    args = parser.parse_args()
    customize_firewall(args.firewall_script, args.domains)


if __name__ == "__main__":
    main()
