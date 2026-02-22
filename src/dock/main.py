"""
Core Main Module for LlamaFactory Dock

Provides the main entry point for different environments (CLI, API, development)
with different default configurations.
"""

import argparse
import pathlib
import sys
from dataclasses import dataclass, fields
from typing import Union, Type
from rich.console import Console
from rich.table import Table

from .dock import LlamaFactoryDock

console = Console()


@dataclass
class MainArgs:
    """Base arguments for Main"""
    # Config file for training (LlamaFactory format YAML/JSON)
    config: Union[str, pathlib.Path, None] = None

    # Job management
    job_or_container_id: str = None
    force: bool = False

    # Command type
    command: str = None

    # Logs
    tail: int = 100

    @classmethod
    def from_args(cls):
        """Parse command line arguments"""
        parser = argparse.ArgumentParser(
            description="LlamaFactory Training Dock",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )

        subparsers = parser.add_subparsers(dest='command', help='Available commands')

        # Start command
        start_parser = subparsers.add_parser('start', help='Start a training job')
        start_parser.add_argument(
            '-c', '--config',
            required=True,
            help='Path to LlamaFactory config file (YAML/JSON)'
        )

        id_help = "Job ID or container ID"
        # Stop command
        stop_parser = subparsers.add_parser('stop', help='Stop a training job')
        stop_parser.add_argument('job_or_container_id', help=id_help)
        stop_parser.add_argument('-f', '--force', action='store_true', help='Force stop (minimal wait, like kill)')

        # Pause command
        pause_parser = subparsers.add_parser('pause', help='Pause a training job')
        pause_parser.add_argument('job_or_container_id', help=id_help)

        # Resume command
        resume_parser = subparsers.add_parser('resume', help='Resume a paused training job')
        resume_parser.add_argument('job_or_container_id', help=id_help)

        # Poll command
        poll_parser = subparsers.add_parser('poll', help='Get job status (poll)')
        poll_parser.add_argument('job_or_container_id', help=id_help)

        # List command
        list_parser = subparsers.add_parser('list', help='List all jobs')

        # Logs command
        logs_parser = subparsers.add_parser('logs', help='Show job logs')
        logs_parser.add_argument('job_or_container_id', help=id_help)
        logs_parser.add_argument('--tail', type=int, default=cls.tail, help='Number of lines')

        # Delete command
        delete_parser = subparsers.add_parser('delete', help='Delete a training job')
        delete_parser.add_argument('job_or_container_id', help=id_help)
        delete_parser.add_argument('-f', '--force', action='store_true', help='Force remove (kill if running, like docker rm -f)')

        # Checkpoints command
        checkpoints_parser = subparsers.add_parser('checkpoints', help='List checkpoints for a job')
        checkpoints_parser.add_argument('job_or_container_id', help=id_help)

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

        # Initialize dock (subclasses override dock)
        dock = cls.dock()

        # Route to command handlers
        cli_cmd = getattr(cls, 'cli_command', 'dock')
        if args.command == 'start':
            return cls.cmd_start(dock, args, cli_command=cli_cmd)
        elif args.command == 'stop':
            return cls.cmd_stop(dock, args)
        elif args.command == 'pause':
            return cls.cmd_pause(dock, args)
        elif args.command == 'resume':
            return cls.cmd_resume(dock, args)
        elif args.command == 'poll':
            return cls.cmd_poll(dock, args)
        elif args.command == 'list':
            return cls.cmd_list(dock, args)
        elif args.command == 'logs':
            return cls.cmd_logs(dock, args)
        elif args.command == 'delete':
            return cls.cmd_delete(dock, args)
        elif args.command == 'checkpoints':
            return cls.cmd_checkpoints(dock, args)
        else:
            console.print("[red]Unknown command[/red]")
            return 1

    @staticmethod
    def cmd_start(dock: LlamaFactoryDock, args, *, cli_command: str = "dock") -> int:
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
            console.print(f"  • View logs:   [cyan]{cli_command} logs {job.job_id}[/cyan]")
            console.print(f"  • Check status: [cyan]{cli_command} poll {job.job_id}[/cyan]")
            console.print(f"  • List all jobs: [cyan]{cli_command} list[/cyan]")
            console.print(f"  • Stop training: [cyan]{cli_command} stop {job.job_id}[/cyan]\n")

            return 0

        except Exception as e:
            console.print(f"[red]❌ Error: {e}[/red]")
            return 1

    @staticmethod
    def cmd_stop(dock: LlamaFactoryDock, args) -> int:
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
    def cmd_pause(dock: LlamaFactoryDock, args) -> int:
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
    def cmd_resume(dock: LlamaFactoryDock, args) -> int:
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
    def cmd_poll(dock: LlamaFactoryDock, args) -> int:
        """Get job status (poll)"""
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
    def cmd_list(dock: LlamaFactoryDock, args) -> int:
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
        console.print(f"\n[dim]Tip: Use job_id or container_id with poll/logs/stop etc.[/dim]\n")
        return 0

    @staticmethod
    def cmd_logs(dock: LlamaFactoryDock, args) -> int:
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
    def cmd_delete(dock: LlamaFactoryDock, args) -> int:
        """Delete a training job"""
        try:
            dock.delete_job(args.job_or_container_id, force=getattr(args, 'force', False))
            console.print(f"[green]✓ Job deleted: {args.job_or_container_id}[/green]")
            return 0
        except Exception as e:
            console.print(f"[red]❌ Error: {e}[/red]")
            return 1

    @staticmethod
    def cmd_checkpoints(dock: LlamaFactoryDock, args) -> int:
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


def main() -> int:
    """CLI entry point function for setuptools console_scripts."""
    return Main.run()


if __name__ == "__main__":
    sys.exit(main())
