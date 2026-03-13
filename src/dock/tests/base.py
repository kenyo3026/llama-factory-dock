"""
Base utilities for test report generation.

Provides git-aware report naming and pytest output capture.
Use via run_tests_with_report() from each test file's __main__ block.
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional


def get_git_info() -> Tuple[Optional[str], bool, bool]:
    """Get current git commit SHA and dirty status."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--short=8', 'HEAD'],
            capture_output=True,
            text=True,
            check=True,
            timeout=5
        )
        commit_sha = result.stdout.strip()

        result = subprocess.run(
            ['git', 'status', '--porcelain'],
            capture_output=True,
            text=True,
            check=True,
            timeout=5
        )
        is_dirty = bool(result.stdout.strip())
        return commit_sha, is_dirty, True

    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None, False, False


def generate_report_name(test_name: str, commit_sha: Optional[str], is_dirty: bool, extension: str = 'log') -> str:
    """Generate report filename: report_{name}_{sha}[_dirty].log"""
    if commit_sha:
        dirty_suffix = '_dirty' if is_dirty else ''
        return f"report_{test_name}_{commit_sha}{dirty_suffix}.{extension}"
    return f"report_{test_name}.{extension}"


def run_tests_with_report(test_file: str, test_name: str) -> int:
    """
    Run pytest and save output to reports/ with git-aware naming.

    Call from if __name__ == "__main__" in each test file.
    """
    commit_sha, is_dirty, git_available = get_git_info()
    report_name = generate_report_name(test_name, commit_sha, is_dirty)

    test_path = Path(test_file)
    reports_dir = test_path.parent / 'reports'
    reports_dir.mkdir(exist_ok=True)
    report_path = reports_dir / report_name

    # Project root: src/dock/tests/ -> parents[3] = project root
    project_root = test_path.resolve().parents[3]

    if git_available:
        print(f"Git commit: {commit_sha}")
        print(f"Working directory: {'dirty' if is_dirty else 'clean'}")
    print(f"Report will be saved to: {report_path}")
    print("=" * 70)

    result = subprocess.run(
        [sys.executable, '-m', 'pytest', test_file, '-v', '--tb=short'],
        capture_output=True,
        text=True,
        cwd=project_root,
    )

    header = f"""{'=' * 70}
Test Report
{'=' * 70}
Test: {test_name}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Git Commit: {commit_sha if git_available else 'N/A'}
Working Dir: {'dirty' if is_dirty else 'clean'}
Exit Code: {result.returncode}
{'=' * 70}

"""

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(header)
        f.write(result.stdout)
        if result.stderr:
            f.write("\n\n=== STDERR ===\n" + result.stderr)

    print(result.stdout)
    if result.stderr:
        print("\n=== STDERR ===")
        print(result.stderr)
    print("=" * 70)
    print(f"Report saved: {report_path}")
    print("=" * 70)

    return result.returncode
