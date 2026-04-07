#!/usr/bin/env python3
"""Test that the environment works with the API"""

import json
import requests

# Test reset endpoint
response = requests.post(
    "http://localhost:5000/reset",
    json={"task_level": "easy"},
    timeout=5
)

if response.status_code == 200:
    data = response.json()
    print("✓ API reset works!")
    print(f"  Observation: {data['observation']}")
else:
    print("✗ API test failed")

# Test health
response = requests.get("http://localhost:5000/health", timeout=5)
if response.status_code == 200:
    print("✓ Health check works!")
else:
    print("✗ Health check failed")
