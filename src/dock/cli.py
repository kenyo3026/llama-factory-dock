"""
CLI Entry Point for LlamaFactory Dock

Provides command-line interface using the Main class.
"""

import sys
from .main import Main, MainArgs


class CliArgs(MainArgs):
    """CLI-specific arguments (inherits from MainArgs with CLI defaults)"""
    pass


def main() -> int:
    """CLI entry point"""
    Main.args = CliArgs
    return Main.run()


if __name__ == "__main__":
    sys.exit(main())
