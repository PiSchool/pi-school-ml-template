import json
import os


def get_config():
    # the secret configuration has to be in a json file called secrets.json
    secrets = "secrets.json"
    if os.path.isfile(secrets):
        file_reader = open(secrets)
        config = json.load(file_reader)
        file_reader.close()
    else:
        config = {}
    # the other configuration entries
    config.update(
        {
            "bucketName": "jenkins-pischool",
            "bucketDataPath": "data/",
            "localDataPath": "data/",
            "modelPath": "models/",
            "modelLocalPath": "models/",
            "dataHashFile": ".datahash"
        }
    )
    return config
