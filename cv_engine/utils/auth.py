"""
Authentication utilities for secure Backend communication
Supports API Key and mTLS authentication
"""
import requests
import logging
from pathlib import Path

logger = logging.getLogger("CV_ENGINE.AUTH")

class AuthenticatedSession:
    """
    Wrapper around requests.Session with authentication support
    Handles API Key and mTLS for secure cloud-to-grid communication
    """
    
    def __init__(self, api_key=None, enable_mtls=False, cert_path=None, key_path=None, ca_path=None):
        self.session = requests.Session()
        self.api_key = api_key
        self.enable_mtls = enable_mtls
        
        # API Key Authentication
        if self.api_key:
            self.session.headers.update({
                'Authorization': f'Bearer {self.api_key}',
                'X-API-Key': self.api_key  # Fallback header format
            })
            logger.info("API Key authentication enabled")
        
        # mTLS Authentication
        if self.enable_mtls:
            if not all([cert_path, key_path, ca_path]):
                logger.error("mTLS enabled but certificate paths not provided")
                raise ValueError("mTLS requires cert_path, key_path, and ca_path")
            
            cert_path = Path(cert_path)
            key_path = Path(key_path)
            ca_path = Path(ca_path)
            
            if not cert_path.exists():
                raise FileNotFoundError(f"Client certificate not found: {cert_path}")
            if not key_path.exists():
                raise FileNotFoundError(f"Client key not found: {key_path}")
            if not ca_path.exists():
                raise FileNotFoundError(f"CA certificate not found: {ca_path}")
            
            self.session.cert = (str(cert_path), str(key_path))
            self.session.verify = str(ca_path)
            logger.info("mTLS authentication enabled")
        
        # User-Agent for identification
        self.session.headers.update({
            'User-Agent': 'HAWKEYE-CV-Engine/2.1'
        })
    
    def post(self, url, json=None, timeout=1.0):
        """POST request with authentication"""
        try:
            response = self.session.post(url, json=json, timeout=timeout)
            return response
        except requests.exceptions.SSLError as e:
            logger.error(f"SSL/TLS error: {e}")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise
    
    def get(self, url, timeout=1.0):
        """GET request with authentication"""
        try:
            response = self.session.get(url, timeout=timeout)
            return response
        except requests.exceptions.SSLError as e:
            logger.error(f"SSL/TLS error: {e}")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise
    
    def close(self):
        """Close the session"""
        self.session.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def create_authenticated_session(config):
    """
    Factory function to create authenticated session from config
    
    Args:
        config: Config object with authentication settings
    
    Returns:
        AuthenticatedSession instance
    """
    return AuthenticatedSession(
        api_key=config.API_KEY if config.API_KEY else None,
        enable_mtls=config.ENABLE_MTLS,
        cert_path=config.MTLS_CERT_PATH if config.ENABLE_MTLS else None,
        key_path=config.MTLS_KEY_PATH if config.ENABLE_MTLS else None,
        ca_path=config.MTLS_CA_PATH if config.ENABLE_MTLS else None
    )
