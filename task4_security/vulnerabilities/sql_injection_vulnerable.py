"""
RISK 1: SQL Injection Vulnerability (OWASP A03:2021)

Vulnerable Code: User input is directly concatenated into SQL queries without 
proper sanitization or parameterization.

This demonstrates a classic SQL injection vulnerability in a user authentication 
and user lookup system.
"""

import sqlite3
from typing import Optional, Dict, Any


class VulnerableUserDatabase:
    """
    VULNERABLE: Direct string concatenation in SQL queries.
    User input is not sanitized or parameterized.
    """

    def __init__(self, db_path: str = ":memory:"):
        """Initialize database with vulnerable schema."""
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self._setup_schema()

    def _setup_schema(self):
        """Create tables for demonstration."""
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                username TEXT UNIQUE,
                password TEXT,
                email TEXT,
                is_admin BOOLEAN DEFAULT 0
            )
        """)
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY,
                action TEXT,
                username TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()

    def authenticate_user(self, username: str, password: str) -> bool:
        """
        VULNERABLE: SQL injection in authentication check.
        
        An attacker can bypass authentication with:
            username: ' OR '1'='1
            password: anything
        
        This constructs: SELECT ... WHERE username = '' OR '1'='1' AND password = ...
        Which evaluates to TRUE for any credentials.
        """
        # VULNERABLE: Direct string concatenation
        query = f"SELECT * FROM users WHERE username = '{username}' AND password = '{password}'"
        
        print(f"[DEBUG] Executing query: {query}")
        
        try:
            self.cursor.execute(query)
            result = self.cursor.fetchone()
            return result is not None
        except Exception as e:
            print(f"[ERROR] Query failed: {e}")
            return False

    def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """
        VULNERABLE: SQL injection in user lookup.
        
        An attacker can extract data with:
            username: ' UNION SELECT id, username, password, email, 1 FROM users WHERE '1'='1
        
        This returns unauthorized user data.
        """
        # VULNERABLE: Direct string concatenation
        query = f"SELECT id, username, email FROM users WHERE username = '{username}'"
        
        print(f"[DEBUG] Executing query: {query}")
        
        try:
            self.cursor.execute(query)
            result = self.cursor.fetchone()
            if result:
                return {
                    "id": result[0],
                    "username": result[1],
                    "email": result[2]
                }
            return None
        except Exception as e:
            print(f"[ERROR] Query failed: {e}")
            return None

    def log_action(self, username: str, action: str) -> None:
        """
        VULNERABLE: SQL injection in logging.
        
        An attacker can modify audit logs or extract data with crafted usernames.
        """
        # VULNERABLE: Direct string concatenation
        query = f"INSERT INTO audit_log (action, username) VALUES ('{action}', '{username}')"
        
        print(f"[DEBUG] Executing query: {query}")
        
        try:
            self.cursor.execute(query)
            self.conn.commit()
        except Exception as e:
            print(f"[ERROR] Insert failed: {e}")

    def get_user_count(self, username: str) -> int:
        """
        VULNERABLE: SQL injection that can extract database information.
        
        An attacker can use: ' OR 1=1 -- 
        To get total count instead of single user count.
        """
        # VULNERABLE: Direct string concatenation
        query = f"SELECT COUNT(*) FROM users WHERE username = '{username}'"
        
        print(f"[DEBUG] Executing query: {query}")
        
        try:
            self.cursor.execute(query)
            result = self.cursor.fetchone()
            return result[0] if result else 0
        except Exception as e:
            print(f"[ERROR] Query failed: {e}")
            return 0

    def setup_test_data(self):
        """Create test users for demonstration."""
        self.cursor.execute(
            "INSERT INTO users (username, password, email, is_admin) VALUES (?, ?, ?, ?)",
            ("admin", "securepassword123", "admin@example.com", 1)
        )
        self.cursor.execute(
            "INSERT INTO users (username, password, email, is_admin) VALUES (?, ?, ?, ?)",
            ("user1", "userpass456", "user1@example.com", 0)
        )
        self.cursor.execute(
            "INSERT INTO users (username, password, email, is_admin) VALUES (?, ?, ?, ?)",
            ("user2", "userpass789", "user2@example.com", 0)
        )
        self.conn.commit()

    def close(self):
        """Close database connection."""
        self.conn.close()


if __name__ == "__main__":
    print("=" * 70)
    print("RISK 1: SQL INJECTION VULNERABILITY DEMONSTRATION")
    print("=" * 70)
    print()

    db = VulnerableUserDatabase()
    db.setup_test_data()

    print("[SETUP] Database initialized with test users")
    print("  - admin / securepassword123 (is_admin=1)")
    print("  - user1 / userpass456 (is_admin=0)")
    print("  - user2 / userpass789 (is_admin=0)")
    print()

    # Test 1: Normal authentication (should work)
    print("--- Test 1: Normal Authentication ---")
    print("Attempting login: username='user1', password='userpass456'")
    result = db.authenticate_user("user1", "userpass456")
    print(f"Authentication result: {result}")
    print()

    # Test 2: SQL Injection - Bypass authentication
    print("--- Test 2: SQL Injection Attack - Authentication Bypass ---")
    injection_payload = "' OR '1'='1"
    print(f"Attempting login with SQL injection:")
    print(f"  username: {injection_payload}")
    print(f"  password: anything")
    result = db.authenticate_user(injection_payload, "anything")
    print(f"Authentication result: {result}")
    print(f"✓ ATTACK SUCCESSFUL: Bypassed authentication with injection!")
    print()

    # Test 3: Normal user lookup
    print("--- Test 3: Normal User Lookup ---")
    print("Looking up user: 'user1'")
    result = db.get_user_by_username("user1")
    print(f"Result: {result}")
    print()

    # Test 4: SQL Injection - Data extraction
    print("--- Test 4: SQL Injection Attack - Data Extraction ---")
    injection_payload = "' UNION SELECT id, username, password FROM users WHERE '1'='1"
    print(f"Injecting payload: {injection_payload[:60]}...")
    result = db.get_user_by_username(injection_payload)
    print(f"Result: {result}")
    print(f"✓ ATTACK SUCCESSFUL: Extracted password hash using UNION-based injection!")
    print()

    # Test 5: SQL Injection - Count bypass
    print("--- Test 5: SQL Injection Attack - Count Bypass ---")
    injection_payload = "' OR 1=1 --"
    print(f"Counting users with payload: {injection_payload}")
    result = db.get_user_count(injection_payload)
    print(f"Count result: {result}")
    print(f"Expected: 1 (only one 'admin' user), Got: {result} (all users)")
    print(f"✓ ATTACK SUCCESSFUL: Bypassed WHERE clause with OR 1=1!")
    print()

    # Test 6: SQL Injection - Logging manipulation
    print("--- Test 6: SQL Injection Attack - Audit Log Manipulation ---")
    injection_payload = "admin', 'malicious_action'); DELETE FROM audit_log WHERE username = '"
    print(f"Logging action with payload: {injection_payload}")
    db.log_action(injection_payload, "login")
    print(f"✓ ATTACK SUCCESSFUL: Potential for audit log manipulation!")
    print()

    db.close()
    print("=" * 70)
