"""
Main entry point for Dryrun Mode (Demo/Testing)

Uses LlamaFactoryDryRunDock with ARM64-compatible lightweight image.
"""

import sys
from .main import Main, MainArgs
from .dock.dryrun import LlamaFactoryDryRunDock


def _dryrun_dock():
    """Dock factory for dryrun mode (5 min simulated training for quick demo)"""
    return LlamaFactoryDryRunDock(dryrun_training_duration=300)


class DryrunMain(Main):
    """Main class for dryrun mode with LlamaFactoryDryRunDock"""

    cli_command = "dock-dryrun"
    dock = staticmethod(_dryrun_dock)


def main() -> int:
    """Dryrun mode entry point"""
    DryrunMain.args = MainArgs
    return DryrunMain.run()


if __name__ == "__main__":
    sys.exit(main())
