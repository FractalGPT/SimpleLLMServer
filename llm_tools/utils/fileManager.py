import os
import requests


def create_folder(name: str):
    """Create folder if it does not exist."""
    os.makedirs(name, exist_ok=True)


def download_file(url: str, directory: str, filename: str):
    """
    Downloads a file from a specified URL and saves it to a given folder with a given filename.

    :param url: URL of the file to download
    :param directory: Path to the folder where the file will be saved
    :param filename: Name of the file to save
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

    filepath = os.path.join(directory, filename)

    response = requests.get(url)
    response.raise_for_status()

    with open(filepath, 'wb') as file:
        file.write(response.content)
