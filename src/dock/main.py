"""
Core Main Module for LlamaFactory Dock

Provides the main entry point for different environments (CLI, API, development)
with different default configurations.
"""

import argparse
import logging
import pathlib
import sys
from dataclasses import dataclass, fields
from typing import Union, Type, Optional
from rich.console import Console
from rich.table import Table

from .dock import LlamaFactoryDock
from .utils.logger import enable_rich_logger

console = Console()


@dataclass
class MainArgs:
    """Base arguments for Main"""
    # Config file for training (LlamaFactory format YAML/JSON)
    config: Union[str, pathlib.Path, None] = None

    # Job management
    job_or_container_id: str = None
    force: bool = False

    # Command group (e.g. "train")
    command: str = None
    # Subcommand within the group (e.g. "start", "stop", "help")
    subcommand: str = None

    # Logs
    tail: int = 100

    # Logger configuration
    log_dir: Union[str, pathlib.Path, None] = None
    log_level: str = "INFO"

    @classmethod
    def from_args(cls):
        """Parse command line arguments"""
        parser = argparse.ArgumentParser(
            description="LlamaFactory Training Dock",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )

        # Global options
        parser.add_argument(
            '--log-dir',
            dest='log_dir',
            help='Directory for log files (default: ./logs)',
        )
        parser.add_argument(
            '--log-level',
            dest='log_level',
            choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
            default='INFO',
            help='Logging level (default: INFO)',
        )

        subparsers = parser.add_subparsers(dest='command', help='Available command groups', required=True)

        # --- train group ---
        train_parser = subparsers.add_parser('train', help='Training commands')
        train_sub = train_parser.add_subparsers(dest='subcommand', help='Training subcommands', required=True)

        id_help = "Job ID or container ID"

        # train start
        start_parser = train_sub.add_parser('start', help='Start a training job')
        start_parser.add_argument(
            '-c', '--config',
            required=True,
            help='Path to LlamaFactory config file (YAML/JSON)'
        )

        # train stop
        stop_parser = train_sub.add_parser('stop', help='Stop a training job')
        stop_parser.add_argument('job_or_container_id', help=id_help)
        stop_parser.add_argument('-f', '--force', action='store_true', help='Force stop (minimal wait, like kill)')

        # train pause
        pause_parser = train_sub.add_parser('pause', help='Pause a training job')
        pause_parser.add_argument('job_or_container_id', help=id_help)

        # train resume
        resume_parser = train_sub.add_parser('resume', help='Resume a paused training job')
        resume_parser.add_argument('job_or_container_id', help=id_help)

        # train status
        status_parser = train_sub.add_parser('status', help='Get job status')
        status_parser.add_argument('job_or_container_id', help=id_help)

        # train list
        train_sub.add_parser('list', help='List all jobs')

        # train logs
        logs_parser = train_sub.add_parser('logs', help='Show job logs')
        logs_parser.add_argument('job_or_container_id', help=id_help)
        logs_parser.add_argument('--tail', type=int, default=cls.tail, help='Number of lines')

        # train delete
        delete_parser = train_sub.add_parser('delete', help='Delete a training job')
        delete_parser.add_argument('job_or_container_id', help=id_help)
        delete_parser.add_argument('-f', '--force', action='store_true', help='Force remove (kill if running, like docker rm -f)')

        # train checkpoints
        checkpoints_parser = train_sub.add_parser('checkpoints', help='List checkpoints for a job')
        checkpoints_parser.add_argument('job_or_container_id', help=id_help)

        # train help
        train_sub.add_parser('help', help='Show llamafactory-cli train --help from the Docker image')

        args = parser.parse_args()

        # Get valid field names for this dataclass
        field_names = {f.name for f in fields(cls)}

        # Only pass arguments that are fields of the dataclass
        return cls(**{k: v for k, v in args.__dict__.items() if k in field_names})


class Main:
    """Core main class for LlamaFactory Dock"""

    args: Type[MainArgs] = MainArgs
    cli_command: str = "dock"
    dock = staticmethod(LlamaFactoryDock)

    @classmethod
    def run(cls):
        """Main execution entry point"""
        args = cls.args.from_args()

        # Setup logger
        log_dir = pathlib.Path(args.log_dir) if args.log_dir else pathlib.Path('./logs')
        log_level = getattr(logging, args.log_level, logging.INFO)
        logger = enable_rich_logger(
            level=log_level,
            directory=log_dir,
            name='llama-factory-dock',
        )

        logger.info("LlamaFactory Dock initialized")
        logger.debug(f"Log directory: {log_dir}")
        logger.debug(f"Log level: {args.log_level}")

        # Initialize dock (subclasses override dock)
        dock = cls.dock(logger=logger)

        # Route to command handlers
        cli_cmd = getattr(cls, 'cli_command', 'dock')
        if args.command == 'train':
            if args.subcommand == 'start':
                return cls.cmd_training_start(dock, args, cli_command=cli_cmd)
            elif args.subcommand == 'stop':
                return cls.cmd_training_stop(dock, args)
            elif args.subcommand == 'pause':
                return cls.cmd_training_pause(dock, args)
            elif args.subcommand == 'resume':
                return cls.cmd_training_resume(dock, args)
            elif args.subcommand == 'status':
                return cls.cmd_training_status(dock, args)
            elif args.subcommand == 'list':
                return cls.cmd_training_list(dock, args)
            elif args.subcommand == 'logs':
                return cls.cmd_training_logs(dock, args)
            elif args.subcommand == 'delete':
                return cls.cmd_training_delete(dock, args)
            elif args.subcommand == 'checkpoints':
                return cls.cmd_training_checkpoints(dock, args)
            elif args.subcommand == 'help':
                return cls.cmd_training_help(dock, args)
            else:
                console.print("[red]Unknown train subcommand[/red]")
                return 1
        else:
            console.print("[red]Unknown command group[/red]")
            return 1

    @staticmethod
    def cmd_training_start(dock: LlamaFactoryDock, args, *, cli_command: str = "dock") -> int:
        """Start a training job"""
        try:
            console.print(f"[yellow]Starting training job with config: {args.config}[/yellow]")
            job = dock.start(args.config)

            if job.status == "failed":
                console.print(f"[red]❌ Failed to start: {job.error_message}[/red]")
                return 1

            console.print(f"\n[green]✓ Training started successfully![/green]\n")
            console.print(f"[bold cyan]Job ID:[/bold cyan] [yellow]{job.job_id}[/yellow]")
            console.print(f"[dim]Container ID: {job.container_id}[/dim]")
            console.print(f"[dim]Status: {job.status}[/dim]")
            console.print(f"[dim](You can use either job_id or container_id for commands)[/dim]")

            # Helpful next steps (use correct CLI command for current mode)
            console.print(f"\n[bold]Next steps:[/bold]")
            console.print(f"  • View logs:    [cyan]{cli_command} train logs {job.job_id}[/cyan]")
            console.print(f"  • Check status: [cyan]{cli_command} train status {job.job_id}[/cyan]")
            console.print(f"  • List all jobs: [cyan]{cli_command} train list[/cyan]")
            console.print(f"  • Stop training: [cyan]{cli_command} train stop {job.job_id}[/cyan]\n")

            return 0

        except Exception as e:
            console.print(f"[red]❌ Error: {e}[/red]")
            return 1

    @staticmethod
    def cmd_training_stop(dock: LlamaFactoryDock, args) -> int:
        """Stop a training job"""
        try:
            job = dock.stop(args.job_or_container_id, force=getattr(args, 'force', False))
            console.print(f"[green]✓ Training stopped[/green]")
            console.print(f"  Job ID: {job.job_id}")
            console.print(f"  Status: {job.status}")
            return 0
        except Exception as e:
            console.print(f"[red]❌ Error: {e}[/red]")
            return 1

    @staticmethod
    def cmd_training_pause(dock: LlamaFactoryDock, args) -> int:
        """Pause a training job"""
        try:
            job = dock.pause(args.job_or_container_id)
            console.print(f"[green]✓ Training paused[/green]")
            console.print(f"  Job ID: {job.job_id}")
            console.print(f"  Status: {job.status}")
            return 0
        except Exception as e:
            console.print(f"[red]❌ Error: {e}[/red]")
            return 1

    @staticmethod
    def cmd_training_resume(dock: LlamaFactoryDock, args) -> int:
        """Resume a training job"""
        try:
            job = dock.resume(args.job_or_container_id)
            console.print(f"[green]✓ Training resumed[/green]")
            console.print(f"  Job ID: {job.job_id}")
            console.print(f"  Status: {job.status}")
            return 0
        except Exception as e:
            console.print(f"[red]❌ Error: {e}[/red]")
            return 1

    @staticmethod
    def cmd_training_status(dock: LlamaFactoryDock, args) -> int:
        """Get job status"""
        try:
            job = dock.poll(args.job_or_container_id)

            console.print(f"\n[bold]Training Job: {job.job_id}[/bold]")
            console.print(f"Status: {job.status}")
            console.print(f"Progress: {job.get_progress_percentage():.1f}%")
            console.print(f"Created: {job.created_at}")

            if job.started_at:
                console.print(f"Started: {job.started_at}")
            if job.completed_at:
                console.print(f"Completed: {job.completed_at}")
            if job.error_message:
                console.print(f"[red]Error: {job.error_message}[/red]")

            return 0
        except Exception as e:
            console.print(f"[red]❌ Error: {e}[/red]")
            return 1

    @staticmethod
    def cmd_training_list(dock: LlamaFactoryDock, args) -> int:
        """List all jobs"""
        jobs = dock.list_jobs()

        if not jobs:
            console.print("[yellow]No training jobs found[/yellow]")
            return 0

        table = Table(title="Training Jobs", show_lines=True)
        table.add_column("Job ID", style="cyan", no_wrap=True)
        table.add_column("Container ID", style="dim", no_wrap=True)
        table.add_column("Status", style="green")
        table.add_column("Progress", style="yellow")
        table.add_column("Created")

        for job in jobs:
            progress = f"{job.get_progress_percentage():.1f}%"
            table.add_row(
                job.job_id,
                job.container_id,
                job.status,
                progress,
                job.created_at.strftime("%Y-%m-%d %H:%M"),
            )

        console.print(table)
        console.print(f"\n[dim]Tip: Use job_id or container_id with status/logs/stop etc.[/dim]\n")
        return 0

    @staticmethod
    def cmd_training_logs(dock: LlamaFactoryDock, args) -> int:
        """Show job logs"""
        try:
            logs = dock.poll_logs(args.job_or_container_id, tail=args.tail)

            console.print(f"\n[bold]Logs for job: {args.job_or_container_id}[/bold]\n")
            for line in logs:
                console.print(line, highlight=False)

            return 0
        except Exception as e:
            console.print(f"[red]❌ Error: {e}[/red]")
            return 1

    @staticmethod
    def cmd_training_delete(dock: LlamaFactoryDock, args) -> int:
        """Delete a training job"""
        try:
            dock.delete_job(args.job_or_container_id, force=getattr(args, 'force', False))
            console.print(f"[green]✓ Job deleted: {args.job_or_container_id}[/green]")
            return 0
        except Exception as e:
            console.print(f"[red]❌ Error: {e}[/red]")
            return 1

    @staticmethod
    def cmd_training_checkpoints(dock: LlamaFactoryDock, args) -> int:
        """List checkpoints for a job"""
        try:
            checkpoints = dock.get_checkpoints(args.job_or_container_id)

            if not checkpoints:
                console.print(f"[yellow]No checkpoints found for job: {args.job_or_container_id}[/yellow]")
                return 0

            console.print(f"\n[bold]Checkpoints for job: {args.job_or_container_id}[/bold]\n")
            for checkpoint in checkpoints:
                console.print(f"  • {checkpoint}")

            return 0
        except Exception as e:
            console.print(f"[red]❌ Error: {e}[/red]")
            return 1


    @staticmethod
    def cmd_training_help(dock: LlamaFactoryDock, args) -> int:
        """Show llamafactory-cli train --help from the Docker image"""
        try:
            console.print("[yellow]Fetching train help from Docker image...[/yellow]")
            help_text = dock.get_train_help()
            console.print(help_text, highlight=False)
            return 0
        except Exception as e:
            console.print(f"[red]❌ Error: {e}[/red]")
            return 1


def main() -> int:
    """CLI entry point function for setuptools console_scripts."""
    return Main.run()


if __name__ == "__main__":
    sys.exit(main())
