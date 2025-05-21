import os
import requests
from urllib.parse import urljoin, urlparse, parse_qs
from bs4 import BeautifulSoup
import sys

# Configuration: base share link and target directory
SHARE_ID = 'f6hdp1zTzh'
BASE_URL = f"https://sid.erda.dk/cgi-sid/ls.py?share_id={SHARE_ID}"
TARGET_SUBDIR = f'./data/{sys.argv[1]}'  # the subdirectory to download


def download_directory(current_dir, local_parent=''):
    """
    Recursively download all files under the given current_dir (relative to share).
    Creates local directories to match the share's structure.
    """
    # Build the URL to list this directory
    list_url = f"{BASE_URL}&current_dir={current_dir}&flags=f"
    res = requests.get(list_url, headers={"User-Agent": "Mozilla/5.0"})
    res.raise_for_status()
    soup = BeautifulSoup(res.text, 'html.parser')

    # Ensure the local directory exists
    # local_dir = os.path.join(local_parent, os.path.basename(current_dir.rstrip('/')))
    # local_dir='./data/enhancer_annotation'
    local_dir=TARGET_SUBDIR
    if local_dir and not os.path.exists(local_dir):
        os.makedirs(local_dir)
    
    # Iterate over all <a> links in the directory listing
    for a in soup.find_all('a'):
        href = a.get('href')
        text = a.get_text()

        # Skip navigation links or parent directories
        if not href or text in ('.', '..') or text == '':
            continue

        parsed = urlparse(href)
        if parsed.path.startswith('/share_redirect/'):
            # It's a file link: download it
            file_url = urljoin("https://sid.erda.dk", href)
            filename = os.path.basename(parsed.path)
            local_path = os.path.join(local_dir, filename)
            # Stream the download to local file
            r = requests.get(file_url, stream=True)
            r.raise_for_status()
            with open(local_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        else:
            # It's a directory link: parse query param for current_dir
            qs = parse_qs(parsed.query)
            next_dir = qs.get('current_dir', [])
            if next_dir:
                next_dir = next_dir[0]
                download_directory(next_dir, local_parent=local_dir)

# Start the recursive download from the enhancer_annotation directory
if os.path.exists("./data")==False:
    os.makedirs("./data")
download_directory(TARGET_SUBDIR, local_parent='.')
