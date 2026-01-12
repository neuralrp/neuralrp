#!/usr/bin/env python3
import requests
import json

print("Testing API response structure...")

try:
    response = requests.post(
        'http://127.0.0.1:5001/api/chat',
        json={
            'messages': [{'role': 'user', 'content': 'test'}],
            'summary': '',
            'characters': [],
            'world_info': None,
            'settings': {
                'system_prompt': 'test',
                'user_persona': '',
                'reinforce_freq': 5,
                'temperature': 0.7,
                'max_length': 250,
                'max_context': 4096,
                'summarize_threshold': 0.85,
                'performance_mode_enabled': True
            },
            'mode': 'narrator'
        },
        timeout=5
    )
    
    print(f"Status: {response.status_code}")
    print(f"Headers: {dict(response.headers)}")
    
    try:
        data = response.json()
        print(f"JSON Response: {json.dumps(data, indent=2)}")
    except:
        print(f"Raw Response: {response.text[:1000]}")
        
except Exception as e:
    print(f"Error: {e}")