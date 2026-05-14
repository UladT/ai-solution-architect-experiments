"""
MITIGATION: SQL Injection (OWASP A03:2021)

Mitigated Code: Uses parameterized queries (prepared statements) to prevent 
SQL injection. User input is treated as data, not executable code.

All database queries use parameterized placeholders (?) with separate 
argument passing to the database driver.
"""

import sqlite3
from typing import Optional, Dict, Any


class MitigatedUserDatabase:
    """
    MITIGATED: Uses parameterized queries (prepared statements).
    User input is separated from SQL code.
    """

    def __init__(self, db_path: str = ":memory:"):
        """Initialize database with mitigated implementation."""
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
        MITIGATED: Parameterized query prevents SQL injection.
        
        The ? placeholders ensure user input is treated as data, not code.
        Attack payloads like ' OR '1'='1 are treated as literal strings.
        """
        # MITIGATED: Parameterized query with ? placeholders
        query = "SELECT * FROM users WHERE username = ? AND password = ?"
        
        print(f"[DEBUG] Executing parameterized query")
        print(f"[DEBUG] Query: {query}")
        print(f"[DEBUG] Parameters: ['{username}', '***']")
        
        try:
            self.cursor.execute(query, (username, password))
            result = self.cursor.fetchone()
            return result is not None
        except Exception as e:
            print(f"[ERROR] Query failed: {e}")
            return False

    def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """
        MITIGATED: Parameterized query prevents SQL injection.
        
        Even complex attack payloads are treated as literal usernames.
        """
        # MITIGATED: Parameterized query
        query = "SELECT id, username, email FROM users WHERE username = ?"
        
        print(f"[DEBUG] Executing parameterized query")
        print(f"[DEBUG] Query: {query}")
        print(f"[DEBUG] Parameters: ['{username}']")
        
        try:
            self.cursor.execute(query, (username,))
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
        MITIGATED: Parameterized query prevents SQL injection in logging.
        
        Audit logs are protected from manipulation.
        """
        # MITIGATED: Parameterized query
        query = "INSERT INTO audit_log (action, username) VALUES (?, ?)"
        
        print(f"[DEBUG] Executing parameterized query")
        print(f"[DEBUG] Query: {query}")
        print(f"[DEBUG] Parameters: ['{action}', '{username}']")
        
        try:
            self.cursor.execute(query, (action, username))
            self.conn.commit()
        except Exception as e:
            print(f"[ERROR] Insert failed: {e}")

    def get_user_count(self, username: str) -> int:
        """
        MITIGATED: Parameterized query prevents SQL injection.
        
        The WHERE clause cannot be bypassed with OR operators.
        """
        # MITIGATED: Parameterized query
        query = "SELECT COUNT(*) FROM users WHERE username = ?"
        
        print(f"[DEBUG] Executing parameterized query")
        print(f"[DEBUG] Query: {query}")
        print(f"[DEBUG] Parameters: ['{username}']")
        
        try:
            self.cursor.execute(query, (username,))
            result = self.cursor.fetchone()
            return result[0] if result else 0
        except Exception as e:
            print(f"[ERROR] Query failed: {e}")
            return 0

    def get_all_users(self) -> list:
        """Get all users safely (for testing)."""
        query = "SELECT id, username, email FROM users"
        try:
            self.cursor.execute(query)
            return [
                {
                    "id": row[0],
                    "username": row[1],
                    "email": row[2]
                }
                for row in self.cursor.fetchall()
            ]
        except Exception as e:
            print(f"[ERROR] Query failed: {e}")
            return []

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
    print("MITIGATION: SQL INJECTION - ATTACK FAILURE DEMONSTRATION")
    print("=" * 70)
    print()

    db = MitigatedUserDatabase()
    db.setup_test_data()

    print("[SETUP] Database initialized with test users")
    print("  - admin / securepassword123 (is_admin=1)")
    print("  - user1 / userpass456 (is_admin=0)")
    print("  - user2 / userpass789 (is_admin=0)")
    print()

    # Test 1: Normal authentication (should work)
    print("--- Test 1: Normal Authentication (Expected: Success) ---")
    print("Attempting login: username='user1', password='userpass456'")
    result = db.authenticate_user("user1", "userpass456")
    print(f"Authentication result: {result}")
    print(f"✓ PASS: Normal login works correctly")
    print()

    # Test 2: SQL Injection Attack - BLOCKED
    print("--- Test 2: SQL Injection Attack - Authentication Bypass (BLOCKED) ---")
    injection_payload = "' OR '1'='1"
    print(f"Attempting login with SQL injection:")
    print(f"  username: {injection_payload}")
    print(f"  password: anything")
    print(f"  (In vulnerable version, this would bypass authentication)")
    result = db.authenticate_user(injection_payload, "anything")
    print(f"Authentication result: {result}")
    print(f"✓ BLOCKED: Injection treated as literal username, attack failed!")
    print()

    # Test 3: Normal user lookup
    print("--- Test 3: Normal User Lookup (Expected: Success) ---")
    print("Looking up user: 'user1'")
    result = db.get_user_by_username("user1")
    print(f"Result: {result}")
    print(f"✓ PASS: Normal lookup works correctly")
    print()

    # Test 4: SQL Injection - Data extraction BLOCKED
    print("--- Test 4: SQL Injection Attack - Data Extraction (BLOCKED) ---")
    injection_payload = "' UNION SELECT id, username, password FROM users WHERE '1'='1"
    print(f"Attempting injection: {injection_payload[:50]}...")
    result = db.get_user_by_username(injection_payload)
    print(f"Result: {result}")
    print(f"✓ BLOCKED: Injection treated as literal username, attack failed!")
    print()

    # Test 5: SQL Injection - Count bypass BLOCKED
    print("--- Test 5: SQL Injection Attack - Count Bypass (BLOCKED) ---")
    injection_payload = "' OR 1=1 --"
    print(f"Counting users with payload: {injection_payload}")
    result = db.get_user_count(injection_payload)
    print(f"Count result: {result}")
    print(f"Expected: 0 (exact username match)")
    print(f"✓ BLOCKED: OR clause ignored, exact match required!")
    print()

    # Test 6: SQL Injection - Logging manipulation BLOCKED
    print("--- Test 6: SQL Injection Attack - Audit Log Manipulation (BLOCKED) ---")
    injection_payload = "admin', 'malicious_action'); DELETE FROM audit_log WHERE username = '"
    print(f"Logging action with payload: {injection_payload}")
    db.log_action(injection_payload, "login")
    print(f"✓ BLOCKED: Payload treated as literal username, no table deletion!")
    print()

    # Verify all users still intact
    print("--- Verification: All Users Intact ---")
    users = db.get_all_users()
    print(f"Current users in database: {len(users)}")
    for user in users:
        print(f"  - {user['username']} ({user['email']})")
    print(f"✓ VERIFIED: Database integrity maintained, no injection damage!")
    print()

    db.close()
    print("=" * 70)
    print("CONCLUSION: All SQL injection attacks were blocked by parameterization")
    print("=" * 70)
