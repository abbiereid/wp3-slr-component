import os
import requests
import json

file_name = '/home/mcdcoste/Downloads/DENKEN-A-2824.mp4'

with open(file_name, 'rb') as file:
    slr_metadata = {
        'sourceLanguage': 'LSE'
    }

    files = {
        'video': (os.path.basename(file_name), file, 'video/mp4'),
        'metadata': ('metadata.json', json.dumps(slr_metadata), 'application/json')
    }

    result = requests.post("http://localhost:5002/extract_features", files=files)
    json_out = result.json()
    print(json_out)