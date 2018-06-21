# Pi School Project Template

This repository contains a template for structuring a machine learning project on GitHub
with a continuous integration procedure.

We implement a Random Forest model for classifying taken from sklearn.

## Installation

Clone the repository by running:

```
git clone https://github.com/sscardapane/pi-school-ml-template.git
```

Inside config.json, specify the S3 bucket where data should be downloaded from.
You should have command line authorization to access the bucket. To do this,
install the AWS command line interface (CLI) and run

```
aws configure
```

Finally, run training with:

```
python training.py
```

## Code organization

Following standard good practices, the code is organized in three major files:

    * `data.py` contains all the code concerning data loading and preprocessing. 

    * `model.py` contains the logic of the model.

    * `training.py` is a script implementing the actual training / test logic.

Additional documentation and comments are provided inside the files. Additionally auxiliary files are:

    * `logger.py` is used to log information for the current training procedure.
    
    * `s3_helper.py` implements the logic for communicating with the S3 bucket.

## Other files

The repository also contains a few auxiliary files:

    * This README, containing all instructions for installing and running the project.

    * The LICENSE under which the files are released.

    * A .gitignore file customized for Python projects.

    * A requirements.txt file containing a full specification of the libraries used in the project, including their versions.

To generate a custom requirements.txt file for your project, install the `pipreqs` library and run:

```
pipreqs /path/to/project
```
