import socket
import getpass

# utils.py
def resolve_save_root(default_root):
    """
    Returns the machine-specific save root based on the hostname.
    Falls back to default_root if the host is not recognized.
    """
    hostname = socket.gethostname()
    username = getpass.getuser()  # Automatically fetches the current user (e.g., 'hcv530')
    
    print(f"[tool.utils, resolve_save_root] Detected Host: {hostname} | User: {username}")
    
    if "pc282" in hostname:
        return f'/media/{username}/T7/'
    elif "viking" in hostname:
        return f'/mnt/scratch/users/{username}/'
    else:
        # Removed the 'raise ValueError' so the fallback actually works
        print(f"[tool.utils, resolve_save_root] Host '{hostname}' not recognized. Using default root.")
        return default_root