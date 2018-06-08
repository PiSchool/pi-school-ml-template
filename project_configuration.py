import secrets


def get_config():
    config = secrets.getSecrets()
    config.update(
        {
            "bucketName": "test-project-id",
            "bucketDataPath": "data/",
            "localDataFile": "data/",
            "modelPath": "models/",
            "modelLocalPath": "models/",
            "dataHashFile": ".datahash"
        }
    )
    return config
