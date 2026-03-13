import hashlib
import secrets

def hash_password(password: str) -> str:
    """Hash password with salt using SHA256"""
    salt = secrets.token_hex(16)
    hash_obj = hashlib.sha256((salt + password).encode())
    return f"{salt}${hash_obj.hexdigest()}"

def verify_password(password: str, hashed: str) -> bool:
    """Verify password against hash"""
    try:
        salt, hash_value = hashed.split("$")
        hash_obj = hashlib.sha256((salt + password).encode())
        return hash_obj.hexdigest() == hash_value
    except:
        return False

def generate_emp_id() -> str:
    """Generate unique employee ID"""
    return f"EMP{secrets.token_hex(4).upper()}"