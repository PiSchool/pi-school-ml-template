

import os.path
import hashlib
import boto3
from sklearn.externals import joblib
import subprocess

import project_configuration
import logger
log = logger.getLogger() 


def get_data():
    """ Get your data from s3 if they does not already  exist localy"""
    config = project_configuration.get_config()
    # check if data already exist
    last_hash = ""
    current_hash = str(hash_data())
    if os.path.isfile(config["dataHashFile"]):
        file_reader = open(config["dataHashFile"])
        last_hash = file_reader.read()
        file_reader.close()

    if not(last_hash == current_hash):
        # the data are not present localy
        log.info("data will be downloaded from s3")
        # it s recommended to configure your credential in aws-cli with aws configure
        # but if needed you can pass aws credential as described here.
        # aws s3 sync ./models s3://covisian-kpi-anomalies/models --profile pischool
        if os.name == 'nt':
            # windows use aws_cli method
            aws_cli(['s3', 'sync',
                     f's3://{config["bucketName"]}/{config["bucketDataPath"]}',
                     config['bucketDataPath']])
        else:
            # unix use simply aws s3 sync
            s3SyncOutput = subprocess.check_output(
                ["aws", "s3", "sync", f's3://{config["bucketName"]}/{config["bucketDataPath"]}', config]).strip().decode("utf-8")
            log.info(["s3SyncOutput:", s3SyncOutput])
        # session = boto3.Session()
        # s3 = session.resource('s3')
        # # download file
        # # in this particular case  we have 1 file.
        # s3.Bucket(config["bucketName"]).download_file(
        #     f'{config["bucketDataPath"]}data-set.csv', f'{config["localDataFile"]}data-set.csv')
        # when we download data, we save in a file a hash representing the distant data directory
        # like this we do not download several time unchanged data.
        log.info(f'{current_hash} will be added to {config["dataHashFile"]}')
        file_writer = open(config["dataHashFile"], "w")
        file_writer.write(current_hash)
        file_writer.close()
    else:
        log.info("data already existing localy, no need to download it again")

    return f'{config["localDataFile"]}data-set.csv'


def hash_data():
    """ list data directory of the project on s3 and get a hash representing it. 
    This allow to detect changes in the dataset. """
    config = project_configuration.get_config()
    session = boto3.Session()
    s3 = session.resource('s3')
    hasher = hashlib.md5()
    [hasher.update(x.key.encode('utf-8')) for x in s3.Bucket(config["bucketName"]).objects.filter(
        Prefix='data/')]
    data_h = hasher.hexdigest()
    log.info(f"hash : {data_h} ")
    return data_h


def get_model(model_name, ext='pkl'):
    """ Get the model you previously saved from s3 """
    config = project_configuration.get_config()
    session = boto3.Session()
    s3 = session.resource('s3')
    s3.Bucket(config["bucketName"]).download_file(
        f'{config["modelPath"]}{model_name}.{ext}', f'{config["modelLocalPath"]}{model_name}.{ext}')
    return f'{config["modelLocalPath"]}{model_name}.{ext}'


def save_and_push_model(model, ext='pkl'):
    """ save you model in a file and send it to s3 """
    config = project_configuration.get_config()
    local_file = f'{config["modelLocalPath"]}{model.name}.{ext}'
    # dump your sklearn model into a file
    joblib.dump(model, local_file)

    session = boto3.Session()
    s3 = session.resource('s3')
    # upload model to s3
    data = open(local_file, 'rb')
    s3.Bucket(config["bucketName"]).put_object(
        Key=f'{config["modelPath"]}{model.name}.{ext}', Body=data)


def aws_cli(*cmd):
    from awscli.clidriver import create_clidriver
    " Use this to run AWS CLI commands "
    old_env = dict(os.environ)
    try:

        # Environment
        env = os.environ.copy()
        env['LC_CTYPE'] = u'en_US.UTF'
        os.environ.update(env)

        # Run awscli in the same process
        exit_code = create_clidriver().main(*cmd)

        # Deal with problems
        if exit_code > 0:
            raise RuntimeError('AWS CLI exited with code {}'.format(exit_code))
    finally:
        os.environ.clear()
        os.environ.update(old_env)
