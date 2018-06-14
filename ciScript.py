"""
Continuous integration script to interact with pip module 
"""

__author__ = "Guillaume Ringwald"
__version__ = "0.1.0"
__license__ = "MIT"


import datetime
import subprocess
import json
import awsLogger
import argparse
import sys

import training
import logger

log = logger.getLogger()


parser = argparse.ArgumentParser()
parser.add_argument("--awsId", help="the aws access key id to use")
parser.add_argument("--awsSecret", help="the aws api secret key")
parser.add_argument("--projectId", help="the id of the project to work with")
parser.add_argument(
    '--rest', help="all other arguments necessary as a json object, for exemple  \"{\\\"bucketName\\\":\\\"jenkins-pischool\\\"}\"")
args = parser.parse_args()
# log.info(["args:", args])

restArgs = {}
if args.rest:
    restArgs = json.loads(args.rest)

# define data object to log


def makeLine():
    """generate a log line that will be added to the performence report after changes on your repository master branch."""
    # gather values
    toLog = {
        "timestamp": datetime.datetime.utcnow().strftime('%Y %m %d - %H:%M:%S'),
        "gitHash": subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8"),

    }
    metrics = training.test()
    toLog.update(metrics)
    toLog["canTrain"] = training.proof_of_training()

    return toLog


def main(args):
    """ This method is called when you run this file on the command line"""
    log.info("start main")

    config = {}
    if args.awsId is not None:
        config["awsId"] = args.awsId

    if args.awsSecret is not None:
        config["awsSecret"] = args.awsSecret

    if args.projectId is not None:
        config["projectId"] = args.projectId

    if not(restArgs == {}):
        config = {**config, **restArgs}

    log.debug(["config: ", config])
    # call awsLogger
    r = awsLogger.logAndPush(makeLine(), config)
    print(r)
    if not(r):
        sys.exit('Could not push log file to s3')


if __name__ == "__main__":
    """ This is executed when run from the command line """
    main(args)
