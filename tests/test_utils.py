import os
import pytest
from simple_rtmw.utils import download_checkpoint


@pytest.fixture(scope="module")
def url() -> str: 
    return'http://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip'
  

@pytest.mark.network
def test_download_checkpoint_end_to_end(url, tmp_path):
    result_path = download_checkpoint(url, tmp_path, progress=False)
    
    # Should return path to extracted onnx file
    assert result_path.endswith('.onnx')
    assert os.path.exists(result_path)
    
    # Should create the onnx file in the destination directory
    onnx_files = [f for f in os.listdir(tmp_path) if f.endswith('.onnx')]
    assert len(onnx_files) == 1
    
    # Should not leave zip file behind
    zip_files = [f for f in os.listdir(tmp_path) if f.endswith('.zip')]
    assert len(zip_files) == 0
    
    # Second call should return same path without re-downloading
    result_path_2 = download_checkpoint(url, tmp_path, progress=False)
    assert result_path == result_path_2
    
