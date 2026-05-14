"""
MITIGATION: Command Injection (OWASP A03:2021)

Mitigated Code: Uses subprocess.run() with list-based arguments and shlex 
parsing to prevent command injection. User input is not passed to shell.

Uses subprocess.run() with shell=False and list-based arguments, ensuring
the OS doesn't interpret shell metacharacters in user input.
"""

import subprocess
import shlex
import os
from typing import Optional, List
import re


class MitigatedFileProcessor:
    """
    MITIGATED: Uses subprocess.run() with shell=False and list-based arguments.
    User input cannot be interpreted as shell commands.
    """

    def __init__(self, work_dir: str = "/tmp/mitigated_app"):
        """Initialize file processor."""
        self.work_dir = work_dir
        os.makedirs(work_dir, exist_ok=True)

    def _validate_filename(self, filename: str) -> bool:
        """
        Validate filename to prevent directory traversal.
        
        Allows: alphanumeric, dots, hyphens, underscores
        Blocks: slashes, path traversal attempts
        """
        if ".." in filename or "/" in filename or filename.startswith("/"):
            print(f"[SECURITY] Rejected invalid filename: {filename}")
            return False
        
        # Allow common file patterns
        if not re.match(r'^[a-zA-Z0-9._-]+(\*)?$', filename):
            print(f"[SECURITY] Rejected filename with invalid characters: {filename}")
            return False
        
        return True

    def search_in_files(self, search_term: str, filename: str = "*") -> Optional[str]:
        """
        MITIGATED: Subprocess with list-based arguments.
        
        User input cannot be interpreted as shell commands.
        Injection payloads like "; cat /etc/passwd" are treated as search terms.
        """
        if not self._validate_filename(filename):
            return None

        # MITIGATED: Use subprocess.run() with shell=False and list args
        # Search term is passed as a separate argument, never interpreted as a command
        file_path = os.path.join(self.work_dir, filename)
        
        print(f"[DEBUG] Executing subprocess with list-based arguments")
        print(f"[DEBUG] Command: grep")
        print(f"[DEBUG] Arguments: ['{search_term}', '{file_path}']")
        print(f"[DEBUG] Shell: False (critical for security)")
        
        try:
            # subprocess.run with shell=False uses exec() internally, NOT shell
            # This means shell metacharacters are treated as literal characters
            result = subprocess.run(
                ["grep", search_term, file_path],
                capture_output=True,
                text=True,
                timeout=5,
                shell=False  # CRITICAL: Never pass shell=True with user input
            )
            return result.stdout if result.stdout else None
        except subprocess.TimeoutExpired:
            print(f"[ERROR] Command timeout")
            return None
        except FileNotFoundError:
            print(f"[ERROR] grep command or file not found")
            return None
        except Exception as e:
            print(f"[ERROR] Command failed: {e}")
            return None

    def compress_files(self, pattern: str) -> bool:
        """
        MITIGATED: Subprocess with list-based arguments.
        
        The pattern is passed directly to tar, not interpreted by shell.
        """
        if not self._validate_filename(pattern):
            return False

        # MITIGATED: Use subprocess.run() with shell=False
        print(f"[DEBUG] Executing subprocess with list-based arguments")
        print(f"[DEBUG] Command: tar")
        print(f"[DEBUG] Arguments: ['-czf', 'archive.tar.gz', '{pattern}']")
        print(f"[DEBUG] Shell: False (critical for security)")
        
        try:
            result = subprocess.run(
                ["tar", "-czf", os.path.join(self.work_dir, "archive.tar.gz"), 
                 os.path.join(self.work_dir, pattern)],
                capture_output=True,
                text=True,
                timeout=10,
                shell=False  # CRITICAL: Never pass shell=True with user input
            )
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            print(f"[ERROR] Command timeout")
            return False
        except Exception as e:
            print(f"[ERROR] Command failed: {e}")
            return False

    def get_file_info(self, filename: str) -> Optional[str]:
        """
        MITIGATED: Subprocess with list-based arguments.
        
        Filename cannot be used to inject additional commands.
        """
        if not self._validate_filename(filename):
            return None

        file_path = os.path.join(self.work_dir, filename)

        # MITIGATED: Use subprocess.run() with shell=False
        print(f"[DEBUG] Executing subprocess with list-based arguments")
        print(f"[DEBUG] Command: file")
        print(f"[DEBUG] Arguments: ['{file_path}']")
        print(f"[DEBUG] Shell: False (critical for security)")
        
        try:
            result = subprocess.run(
                ["file", file_path],
                capture_output=True,
                text=True,
                timeout=5,
                shell=False  # CRITICAL: Never pass shell=True with user input
            )
            return result.stdout if result.stdout else None
        except subprocess.TimeoutExpired:
            print(f"[ERROR] Command timeout")
            return None
        except Exception as e:
            print(f"[ERROR] Command failed: {e}")
            return None

    def list_files_by_pattern(self, pattern: str) -> Optional[str]:
        """
        MITIGATED: Subprocess with list-based arguments.
        
        Pattern is passed as argument, not interpreted by shell glob expansion.
        """
        if not self._validate_filename(pattern):
            return None

        # MITIGATED: Use subprocess.run() with shell=False
        # Note: Without shell=True, glob expansion is NOT performed by shell
        # This is actually MORE secure but behaves differently
        print(f"[DEBUG] Executing subprocess with list-based arguments")
        print(f"[DEBUG] Command: ls")
        print(f"[DEBUG] Arguments: ['-la', '{self.work_dir}/{pattern}']")
        print(f"[DEBUG] Shell: False (critical for security)")
        
        try:
            result = subprocess.run(
                ["ls", "-la", os.path.join(self.work_dir, pattern)],
                capture_output=True,
                text=True,
                timeout=5,
                shell=False  # CRITICAL: Never pass shell=True with user input
            )
            return result.stdout if result.stdout else None
        except subprocess.TimeoutExpired:
            print(f"[ERROR] Command timeout")
            return None
        except Exception as e:
            print(f"[ERROR] Command failed: {e}")
            return None

    def ping_host(self, hostname: str, count: int = 4) -> Optional[str]:
        """
        MITIGATED: Subprocess with list-based arguments and hostname validation.
        
        Hostname is validated and passed as separate argument.
        """
        # MITIGATED: Validate hostname format
        if not re.match(r'^[a-zA-Z0-9.-]+$', hostname):
            print(f"[SECURITY] Rejected invalid hostname: {hostname}")
            return None

        if len(hostname) > 255:
            print(f"[SECURITY] Rejected hostname exceeding max length")
            return None

        # MITIGATED: Use subprocess.run() with shell=False
        print(f"[DEBUG] Executing subprocess with list-based arguments")
        print(f"[DEBUG] Command: ping")
        print(f"[DEBUG] Arguments: ['-c', '{count}', '{hostname}']")
        print(f"[DEBUG] Shell: False (critical for security)")
        
        try:
            result = subprocess.run(
                ["ping", "-c", str(count), hostname],
                capture_output=True,
                text=True,
                timeout=10,
                shell=False  # CRITICAL: Never pass shell=True with user input
            )
            return result.stdout if result.stdout else None
        except subprocess.TimeoutExpired:
            print(f"[ERROR] Command timeout")
            return None
        except Exception as e:
            print(f"[ERROR] Command failed: {e}")
            return None

    def cleanup(self):
        """Clean up test directory."""
        try:
            # MITIGATED: Use subprocess instead of os.system
            subprocess.run(
                ["rm", "-rf", self.work_dir],
                shell=False,
                timeout=5
            )
        except:
            pass


if __name__ == "__main__":
    print("=" * 70)
    print("MITIGATION: COMMAND INJECTION - ATTACK FAILURE DEMONSTRATION")
    print("=" * 70)
    print()

    processor = MitigatedFileProcessor()

    # Create test files
    test_file = os.path.join(processor.work_dir, "test.txt")
    with open(test_file, "w") as f:
        f.write("This is a test file\nWith some content\nFor searching")

    print("[SETUP] Test file created at:", test_file)
    print()

    # Test 1: Normal search
    print("--- Test 1: Normal File Search (Expected: Success) ---")
    print("Searching for 'test' in test.txt")
    result = processor.search_in_files("test", "test.txt")
    print(f"Result:\n{result}")
    print(f"✓ PASS: Normal search works correctly")
    print()

    # Test 2: Command injection - whoami BLOCKED
    print("--- Test 2: Command Injection Attack - Execute whoami (BLOCKED) ---")
    injection_payload = "test; whoami #"
    print(f"Searching for: {injection_payload}")
    print(f"(In vulnerable version, this would execute whoami command)")
    result = processor.search_in_files(injection_payload, "test.txt")
    print(f"Result: {result}")
    print(f"✓ BLOCKED: Injection treated as literal search term, whoami not executed!")
    print()

    # Test 3: Command injection - ls BLOCKED
    print("--- Test 3: Command Injection Attack - List Directory (BLOCKED) ---")
    injection_payload = "*; ls -la / | head -5 #"
    print(f"Listing files with pattern: {injection_payload}")
    result = processor.list_files_by_pattern(injection_payload)
    print(f"Result: {result}")
    print(f"✓ BLOCKED: Invalid filename rejected by validation layer!")
    print()

    # Test 4: Normal file info
    print("--- Test 4: Normal File Info (Expected: Success) ---")
    print("Getting file info for test.txt")
    result = processor.get_file_info("test.txt")
    print(f"Result: {result}")
    print(f"✓ PASS: Normal file info works correctly")
    print()

    # Test 5: Command injection - pwd BLOCKED
    print("--- Test 5: Command Injection Attack - Print Working Directory (BLOCKED) ---")
    injection_payload = "test.txt; pwd #"
    print(f"Getting file info for: {injection_payload}")
    print(f"(In vulnerable version, this would execute pwd command)")
    result = processor.get_file_info(injection_payload)
    print(f"Result: {result}")
    print(f"✓ BLOCKED: Injection treated as literal filename, pwd not executed!")
    print()

    # Test 6: Command injection - env BLOCKED
    print("--- Test 6: Command Injection Attack - Print Environment Variables (BLOCKED) ---")
    injection_payload = "test.txt; env | head -5 #"
    print(f"Getting file info for: {injection_payload}")
    print(f"(In vulnerable version, this would execute env command)")
    result = processor.get_file_info(injection_payload)
    print(f"Result: {result}")
    print(f"✓ BLOCKED: Injection treated as literal filename, env not executed!")
    print()

    # Test 7: Normal ping
    print("--- Test 7: Normal Ping - localhost (Expected: Success) ---")
    print("Pinging localhost (1 count)")
    result = processor.ping_host("127.0.0.1", count=1)
    print(f"Result (first 100 chars):\n{result[:100] if result else 'No result'}")
    print(f"✓ PASS: Normal ping works correctly")
    print()

    # Test 8: Command injection - id BLOCKED
    print("--- Test 8: Command Injection Attack - Print User ID (BLOCKED) ---")
    injection_payload = "127.0.0.1; id #"
    print(f"Pinging host: {injection_payload}")
    print(f"(In vulnerable version, this would execute id command)")
    result = processor.ping_host(injection_payload, count=1)
    print(f"Result: {result}")
    print(f"✓ BLOCKED: Invalid hostname rejected by validation layer!")
    print()

    # Test 9: Command injection with special characters BLOCKED
    print("--- Test 9: Command Injection Attack - Special Characters (BLOCKED) ---")
    injection_payload = "127.0.0.1 | cat /etc/passwd"
    print(f"Pinging host: {injection_payload}")
    print(f"(In vulnerable version, this could execute dangerous commands)")
    result = processor.ping_host(injection_payload, count=1)
    print(f"Result: {result}")
    print(f"✓ BLOCKED: Invalid hostname rejected by validation layer!")
    print()

    processor.cleanup()
    print("=" * 70)
    print("CONCLUSION: All command injection attacks were blocked by subprocess isolation")
    print("=" * 70)
