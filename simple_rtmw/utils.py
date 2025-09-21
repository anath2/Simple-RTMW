"""utility functions for downloading checkpoints."""

import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm


def extract_zip(zip_file_path: Path, extract_to_path: Path):
    """Extracts contents of the zip file to a path.

    Args:
        zip_file_path: Path of the zip file
        extract_to_path: Path of the extracted contents.
    """
    if not extract_to_path.exists():
        extract_to_path.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to_path)


def download_url_to_file(
        url: str,
        dst: Path,
        progress: bool = True,
        download_chunk_size: int = 8192,
    ):
    """Download object at the given URL to a local path.

    Args:
        url: URL of the object to download
        dst: Path where the final result would be saved
        progress: whether or not to display a progress bar to stderr Defaults to True.
        download_chunk_size: Size of the chunks to download.
    """
    headers = {'User-Agent': 'mmlmtools'}
    response = requests.get(url, headers=headers, stream=True)
    response.raise_for_status()

    # Get file size from Content-Length header
    file_size = None
    content_length = response.headers.get('Content-Length')
    if content_length is not None:
        file_size = int(content_length)

    # Save it to a temp file and rename it atomically
    dst = dst.expanduser().absolute()

    with tempfile.NamedTemporaryFile(mode='wb', dir=dst.parent, delete=False) as f:
        temp_file = Path(f.name)

    try:
        with tqdm(
            total=file_size,
            disable=not progress,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=download_chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

        f.close()
        Path(f.name).rename(dst)
    finally:
        f.close()
        temp_file = Path(f.name)
        if temp_file.exists():
            temp_file.unlink()


def download_checkpoint(
    url: str,
    dst_dir: Path,
    progress: bool = True,
) -> str:
    """Download the checkpoint from the given URL.

    If an end2end.onnx file already exists in dst_dir,
    it will be returned directly.

    Args:
        url: URL to download the checkpoint from
        dst_dir: directory in which to save the model
        progress: whether or not to display a progress bar to stderr.

    Returns:
        str: The path of the onnx model file.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)

    # Check if any end2end.onnx file already exists
    existing_onnx = list(dst_dir.rglob('*end2end.onnx'))
    if existing_onnx:
        return str(existing_onnx[0])

    # Download the zip file
    filename = url.split('/')[-1]
    cached_file = dst_dir / filename

    sys.stderr.write(f'Downloading: "{url}" to {cached_file}\n')
    download_url_to_file(url, cached_file, progress=progress)

    # Extract zip and find onnx file
    tmp_dir = dst_dir / "tmp"
    extract_zip(cached_file, tmp_dir)

    # Find the end2end.onnx file in extracted contents
    extracted_onnx = list(tmp_dir.rglob('*end2end.onnx'))
    if not extracted_onnx:
        raise FileNotFoundError(f"No *end2end.onnx file found in {cached_file}")

    # Move onnx file to dst_dir and clean up
    final_onnx_path = dst_dir / extracted_onnx[0].name
    extracted_onnx[0].rename(final_onnx_path)
    cached_file.unlink()

    # Clean up tmp directory
    shutil.rmtree(tmp_dir)

    return str(final_onnx_path)
