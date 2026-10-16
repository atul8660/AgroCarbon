import pytest
from unittest.mock import patch, MagicMock
from app.services.storage_service import StorageService
import boto3
from moto import mock_aws

@pytest.fixture
def mock_s3():
    with mock_aws():
        # Setup mock bucket
        s3 = boto3.client('s3', region_name='us-east-1')
        s3.create_bucket(Bucket='tree-carbon-system')
        yield s3

def test_upload_full_tree(mock_s3):
    service = StorageService()
    service.bucket = 'tree-carbon-system'
    url = service.upload_full_tree(b"fake_image_data", "test_tree.jpg")
    
    assert "https://tree-carbon-system.s3.us-east-1.amazonaws.com/uploads/full-tree/" in url
    assert ".jpg" in url

def test_generate_public_url():
    service = StorageService()
    service.bucket = 'tree-carbon-system'
    url = service.generate_public_url("uploads/full-tree/123.jpg")
    assert url == "https://tree-carbon-system.s3.us-east-1.amazonaws.com/uploads/full-tree/123.jpg"

def test_upload_report(mock_s3, tmp_path):
    # Create fake pdf
    test_pdf = tmp_path / "test_report.pdf"
    test_pdf.write_bytes(b"pdf data")
    
    service = StorageService()
    service.bucket = 'tree-carbon-system'
    url = service.upload_report(str(test_pdf), "pdf")
    
    assert "https://tree-carbon-system.s3.us-east-1.amazonaws.com/reports/pdf/" in url
    assert ".pdf" in url
