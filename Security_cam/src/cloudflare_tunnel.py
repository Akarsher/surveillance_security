import os
import subprocess
import json

class CloudflareTunnel:
    def __init__(self):
        self.running = False
        self.public_url = None
        self.error = None

    def start(self, port):
        try:
            # Start the Cloudflare tunnel using subprocess
            command = f"cloudflared tunnel --url http://localhost:{port}"
            process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.running = True
            self.public_url = f"http://your-tunnel-url.com"  # Replace with actual URL logic
            return self.public_url
        except Exception as e:
            self.error = str(e)
            self.running = False
            return None

    def stop(self):
        if self.running:
            # Logic to stop the tunnel
            self.running = False
            self.public_url = None

    def get_status(self):
        return {
            "running": self.running,
            "public_url": self.public_url,
            "error": self.error
        }

    def get_url_history(self):
        # Logic to retrieve URL history
        return []  # Replace with actual history retrieval logic

cf_tunnel = CloudflareTunnel()