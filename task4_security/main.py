#!/usr/bin/env python3
"""
Task 4: Security Vulnerability Assessment

This task demonstrates OWASP A03:2021 Injection vulnerabilities:
1. SQL Injection
2. Command Injection

For each vulnerability:
- Vulnerable code with successful attack demonstrations
- Comprehensive risk assessments (before and after)
- Mitigated code with attack failure demonstrations
- Acceptance criteria (AC-1 through AC-5) verification

Run modes:
  python main.py --mode demo        Show vulnerability & mitigation demos
  python main.py --mode evaluate    Run all acceptance criteria tests
  python main.py --mode all         Demo + evaluation (default)
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path


def print_banner():
    """Print welcome banner."""
    print()
    print("=" * 80)
    print("TASK 4: SECURITY VULNERABILITY ASSESSMENT")
    print("OWASP A03:2021 - Injection Vulnerabilities")
    print("=" * 80)
    print()
    print("VULNERABILITIES COVERED:")
    print("  1. SQL Injection (CWE-89)")
    print("     - Authentication bypass")
    print("     - Data extraction (UNION-based)")
    print("     - Boolean blind injection")
    print("     - Audit log manipulation")
    print()
    print("  2. Command Injection (CWE-78)")
    print("     - Information disclosure (whoami, env)")
    print("     - File access (cat, ls)")
    print("     - Reverse shell (bash, nc)")
    print("     - System destruction (rm -rf)")
    print()
    print("MITIGATIONS PROVIDED:")
    print("  1. SQL Injection → Parameterized queries (prepared statements)")
    print("  2. Command Injection → Subprocess isolation (shell=False)")
    print()
    print("=" * 80)
    print()


def run_demo():
    """Run vulnerability and mitigation demonstrations."""
    print("RUNNING DEMONSTRATIONS")
    print("=" * 80)
    print()

    demos = [
        ("SQL Injection (Vulnerable)", "vulnerabilities/sql_injection_vulnerable.py"),
        ("SQL Injection (Mitigated)", "mitigations/sql_injection_mitigated.py"),
        ("Command Injection (Vulnerable)", "vulnerabilities/command_injection_vulnerable.py"),
        ("Command Injection (Mitigated)", "mitigations/command_injection_mitigated.py"),
    ]

    for demo_name, demo_file in demos:
        print()
        print("=" * 80)
        print(f"DEMO: {demo_name}")
        print("=" * 80)
        print()

        try:
            result = subprocess.run(
                [sys.executable, demo_file],
                timeout=30
            )
            if result.returncode != 0:
                print(f"⚠️  Demo exited with code {result.returncode}")
        except subprocess.TimeoutExpired:
            print(f"⚠️  Demo timed out")
        except FileNotFoundError:
            print(f"✗ Error: Demo file not found: {demo_file}")
        except Exception as e:
            print(f"✗ Error running demo: {e}")

    print()
    print("=" * 80)
    print("DEMONSTRATIONS COMPLETE")
    print("=" * 80)
    print()


def run_evaluation():
    """Run acceptance criteria evaluation."""
    print("RUNNING ACCEPTANCE CRITERIA EVALUATION")
    print("=" * 80)
    print()

    try:
        result = subprocess.run(
            [sys.executable, "evaluation/evaluator.py"],
            timeout=60
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("✗ Evaluation timed out")
        return False
    except FileNotFoundError:
        print("✗ Error: Evaluator not found")
        return False
    except Exception as e:
        print(f"✗ Error running evaluation: {e}")
        return False


def show_file_structure():
    """Show Task 4 directory structure."""
    print()
    print("FILE STRUCTURE")
    print("-" * 80)
    print()
    print("task4_security/")
    print("├── vulnerabilities/")
    print("│   ├── sql_injection_vulnerable.py          ← SQL injection demo")
    print("│   └── command_injection_vulnerable.py      ← Command injection demo")
    print("├── mitigations/")
    print("│   ├── sql_injection_mitigated.py           ← SQL injection fixed")
    print("│   └── command_injection_mitigated.py       ← Command injection fixed")
    print("├── assessments/")
    print("│   ├── sql_injection_assessment.md          ← Risk assessment (before/after)")
    print("│   └── command_injection_assessment.md      ← Risk assessment (before/after)")
    print("├── evaluation/")
    print("│   └── evaluator.py                         ← AC-1 through AC-5 tests")
    print("└── main.py                                  ← Entry point (this file)")
    print()


def show_acceptance_criteria():
    """Show acceptance criteria description."""
    print()
    print("ACCEPTANCE CRITERIA")
    print("-" * 80)
    print()
    print("AC-1: VULNERABILITY DEMONSTRATION")
    print("  ✓ SQL Injection vulnerabilities demonstrated with 4+ attack scenarios")
    print("  ✓ Command Injection vulnerabilities demonstrated with 5+ attack scenarios")
    print("  ✓ All attacks execute successfully on vulnerable code")
    print()
    print("AC-2: RISK ASSESSMENT")
    print("  ✓ Comprehensive risk assessment for each vulnerability")
    print("  ✓ Before mitigation: Likelihood, Impact, Business loss documented")
    print("  ✓ After mitigation: Residual risks identified (not fully mitigated)")
    print("  ✓ Both assessments show before/after comparison with metrics")
    print()
    print("AC-3: MITIGATION IMPLEMENTATION")
    print("  ✓ SQL Injection: Parameterized queries implemented")
    print("  ✓ Command Injection: Subprocess isolation (shell=False) implemented")
    print("  ✓ Input validation and error handling included")
    print("  ✓ Mitigated code is production-ready")
    print()
    print("AC-4: ATTACK PREVENTION")
    print("  ✓ All SQL injection attacks blocked on mitigated code")
    print("  ✓ All command injection attacks blocked on mitigated code")
    print("  ✓ Failed attacks documented with before/after comparison")
    print("  ✓ Normal functionality preserved on mitigated code")
    print()
    print("AC-5: SECURITY EVALUATION METRICS")
    print("  ✓ Vulnerability count: 2 vulnerabilities assessed")
    print("  ✓ Attack scenarios: 9 total attack scenarios (SQL + Command)")
    print("  ✓ Mitigation effectiveness: 100% attack blocking rate")
    print("  ✓ Security controls: Parameterization + Subprocess isolation")
    print("  ✓ Residual risk assessment: Documented edge cases and remaining risks")
    print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Task 4: Security Vulnerability Assessment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                   # Run demo + evaluation
  python main.py --mode demo       # Demo only
  python main.py --mode evaluate   # Evaluation only
        """
    )
    parser.add_argument(
        "--mode",
        choices=["demo", "evaluate", "all"],
        default="all",
        help="Execution mode (default: all)"
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show task information and exit"
    )

    args = parser.parse_args()

    # Show banner
    print_banner()

    # Show info if requested
    if args.info:
        show_file_structure()
        show_acceptance_criteria()
        return 0

    # Run selected modes
    success = True

    if args.mode in ["demo", "all"]:
        run_demo()

    if args.mode in ["evaluate", "all"]:
        if not run_evaluation():
            success = False

    if success:
        print()
        print("=" * 80)
        print("✓ TASK 4 EXECUTION COMPLETE")
        print("=" * 80)
        print()
        print("Next steps:")
        print("  1. Review vulnerability demonstrations in:")
        print("     - vulnerabilities/sql_injection_vulnerable.py")
        print("     - vulnerabilities/command_injection_vulnerable.py")
        print()
        print("  2. Review risk assessments in:")
        print("     - assessments/sql_injection_assessment.md")
        print("     - assessments/command_injection_assessment.md")
        print()
        print("  3. Review mitigations in:")
        print("     - mitigations/sql_injection_mitigated.py")
        print("     - mitigations/command_injection_mitigated.py")
        print()
        return 0
    else:
        print()
        print("=" * 80)
        print("✗ TASK 4 EXECUTION FAILED")
        print("=" * 80)
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
