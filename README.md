# SignON SLR Component

This is the repository containing the code of the SignON SLR component.
This component is a web service built with Flask, that extracts some sign language representation from an input video
on request, and returns a JSON response.

## Installation

This component is built to run in a Docker container (see `Dockerfile`).
Inside the running container, the `signon` user will launch the web service and listen on port 5002.

As you can see from inspecting the `Dockerfile`, requirements are installed in the `signon` user's home directory.
The environment variable `SIGNON_SLR_COMPONENT_VERSION` is set so that the web service has access to the version
of the component that is currently running. This means that this environment variable must be updated to
coincide with the tag of the Docker image for consistency.

The component expects model checkpoints under `/model`. These checkpoints can be found in the [slr-pipeline repository documentation](https://github.com/signon-project/wp3-slr-pipeline/blob/main/documentation/inference.md).
The path to the currently used checkpoint is set on [this line](https://github.com/signon-project/wp3-slr-component/blob/main/web_service/app.py#L9).

## Testing

To test locally, you can use the following commands:

```bash
docker build -t signon/wp3/slr .
docker run -v /model:/model --name signon_wp3_slr -d --publish 5002:5002 signon/wp3/slr
```

You can send requests to this component using `multipart/form-data`.
For example, to send a file from a Python script,

```python
with open(file_name, 'rb') as file:
    slr_metadata = {
        'sourceLanguage': 'VGT'
    }
    
    files = {
        'video': (os.path.basename(file_name), file, 'video/mp4'),
        'metadata': ('metadata.json', json.dumps(slr_metadata), 'application/json')
    }

    result = requests.post("http://localhost:5002/extract_features", files=files)
    json_out = result.json()
    print(json_out)
```

To stop the container, you can use:

```bash
docker stop signon_wp3_slr
```

# LICENSE

This code is licensed under the Apache License, Version 2.0 (LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
