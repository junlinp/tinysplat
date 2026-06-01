"""
Script to download the Lena test image.
Run this once to download the test image.
"""

import os
from pathlib import Path

# Create data directory
data_dir = Path(__file__).parent / "data"
data_dir.mkdir(exist_ok=True)

lena_path = data_dir / "lena.png"

if lena_path.exists():
    print(f"Lena image already exists at {lena_path}")
    exit(0)

print("Downloading Lena test image...")

try:
    import urllib.request
    import ssl
    
    url = "https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png"
    
    # Create SSL context
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    
    # Use urlopen with context
    with urllib.request.urlopen(url, context=ctx) as response:
        with open(lena_path, 'wb') as out_file:
            out_file.write(response.read())
    
    print(f"Successfully downloaded to {lena_path}")
except Exception as e:
    print(f"Failed to download: {e}")
    print("\nAlternative: Download manually from:")
    print("https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png")
    print(f"Save it to: {lena_path}")
    exit(1)

