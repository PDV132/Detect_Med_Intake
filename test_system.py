#!/usr/bin/env python3

import requests
import json
import time
import os
from pathlib import Path

def test_api_health():
    """Test if API is running"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def test_mock_detection():
    """Test mock detection endpoint"""
    try:
        response = requests.post("http://localhost:8000/test-detection/", timeout=10)
        if response.status_code == 200:
            result = response.json()
            print("✅ Mock detection test passed")
            print(json.dumps(result, indent=2))
            return True
        else:
            print(f"❌ Mock detection test failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Mock detection test error: {e}")
        return False

def main():
    print("🧪 Testing Medicine Intake Detection System")
    print("=" * 50)
    
    # Test 1: API Health
    print("1. Testing API health...")
    if test_api_health():
        print("✅ API is running")
    else:
        print("❌ API is not accessible")
        print("   Make sure to run: uvicorn api_backend:app --reload")
        return
    
    print()
    
    # Test 2: Mock Detection
    print("2. Testing mock detection...")
    test_mock_detection()
    print()
    
    print("🎉 System testing complete!")

if __name__ == "__main__":
    main()