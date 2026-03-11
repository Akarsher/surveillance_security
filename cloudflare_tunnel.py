"""
Cloudflare Quick Tunnel Manager
FREE - No domain required!
URL format: https://random-words.trycloudflare.com
"""

import subprocess
import threading
import re
import time
import json
import os
from datetime import datetime

class CloudflareTunnel:
    def __init__(self):
        self.process = None
        self.public_url = None
        self.running = False
        self.error = None
        self.started_at = None
        self.url_history_file = "tunnel_url_history.json"
    
    def start(self, port: int = 8000):
        """Start Cloudflare Quick Tunnel"""
        if self.running:
            return self.public_url
        
        self.error = None
        
        def run_tunnel():
            try:
                # Start cloudflared with quick tunnel
                # Using --protocol http2 to avoid QUIC errors
                self.process = subprocess.Popen(
                    [
                        "cloudflared", 
                        "tunnel",
                        "--url", f"http://localhost:{port}",
                        "--protocol", "http2",  # Force HTTP/2 to avoid QUIC errors
                        "--no-autoupdate"
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
                )
                
                self.running = True
                self.started_at = datetime.now().isoformat()
                
                # Read output to find the public URL
                for line in self.process.stdout:
                    line = line.strip()
                    if line:
                        # Only print important lines, skip repetitive errors
                        if "ERR" not in line or "trycloudflare" in line.lower():
                            print(f"[Cloudflare] {line}")
                    
                    # Look for the trycloudflare.com URL
                    match = re.search(r'(https://[a-z0-9-]+\.trycloudflare\.com)', line)
                    if match:
                        self.public_url = match.group(1)
                        self._save_url_to_history()
                        self._print_banner(port)
                
                self.running = False
                print("[Cloudflare] Tunnel stopped")
                
            except FileNotFoundError:
                self.error = "cloudflared not found! Install with: winget install Cloudflare.cloudflared"
                print(f"❌ {self.error}")
                self.running = False
            except Exception as e:
                self.error = str(e)
                print(f"❌ Tunnel error: {e}")
                self.running = False
        
        # Start tunnel in background thread
        thread = threading.Thread(target=run_tunnel, daemon=True)
        thread.start()
        
        # Wait for URL to be captured (max 30 seconds)
        print("⏳ Starting Cloudflare tunnel...")
        for i in range(30):
            if self.public_url:
                return self.public_url
            if self.error:
                return None
            time.sleep(1)
            if i % 5 == 0 and i > 0:
                print(f"⏳ Still connecting... ({i}s)")
        
        if not self.public_url:
            self.error = "Timeout waiting for tunnel URL"
            print("❌ Timeout waiting for tunnel URL")
        
        return self.public_url
    
    def _print_banner(self, port: int):
        """Print success banner with URL"""
        print("")
        print("=" * 65)
        print("🌐 CLOUDFLARE TUNNEL ACTIVE - FREE PUBLIC ACCESS")
        print("=" * 65)
        print(f"")
        print(f"   📍 Local URL:    http://localhost:{port}")
        print(f"   🌍 Public URL:   {self.public_url}")
        print(f"")
        print(f"   📱 Share this URL to access from ANYWHERE!")
        print(f"   🏫 Works from your college, phone, or any network!")
        print(f"")
        print("=" * 65)
        print("⚠️  Note: URL changes when you restart. Check dashboard for current URL.")
        print("=" * 65)
        print("")
    
    def _save_url_to_history(self):
        """Save URL to history file"""
        try:
            history = []
            if os.path.exists(self.url_history_file):
                with open(self.url_history_file, 'r') as f:
                    history = json.load(f)
            
            history.insert(0, {
                "url": self.public_url,
                "created_at": datetime.now().isoformat(),
                "port": 8000
            })
            
            # Keep only last 10 URLs
            history = history[:10]
            
            with open(self.url_history_file, 'w') as f:
                json.dump(history, f, indent=2)
                
        except Exception as e:
            print(f"Warning: Could not save URL history: {e}")
    
    def stop(self):
        """Stop the tunnel"""
        if self.process:
            print("🛑 Stopping Cloudflare tunnel...")
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except:
                try:
                    self.process.kill()
                except:
                    pass
            self.process = None
        
        self.running = False
        self.public_url = None
        self.error = None
        print("✅ Tunnel stopped")
    
    def get_status(self):
        """Get current tunnel status"""
        return {
            "running": self.running,
            "public_url": self.public_url,
            "error": self.error,
            "started_at": self.started_at,
            "type": "cloudflare_quick_tunnel"
        }
    
    def get_url_history(self):
        """Get URL history"""
        try:
            if os.path.exists(self.url_history_file):
                with open(self.url_history_file, 'r') as f:
                    return json.load(f)
        except:
            pass
        return []

# Global instance
cf_tunnel = CloudflareTunnel()