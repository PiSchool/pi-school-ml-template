# Pi School Project Template

This repository contains a template for structuring a machine learning project on GitHub.

We implement a neural network for classifying the Iris dataset, made from scratch
with TensorFlow (in eager execution). For a basic tutorial on TF eager check out:

    https://medium.com/tensorflow/building-an-iris-classifier-with-eager-execution-13c00a32adb0

## Installation

Clone the repository by running:

```
git clone https://github.com/sscardapane/pi-school-ml-template.git
```

Run training with:

```
python training.py
```

## Code organization

Following standard good practices, the code is organized in three major files:

    * `data.py` contains all the code concerning data loading and preprocessing. It returns iterators that can be used to cycle over the data.

    * `model.py` contains the logic of the model.

    * `training.py` is a script implementing the actual training / test logic.

Additional documentation and comments are provided inside the files.

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

## End of project

In your requirements.txt, remove the dependency to awsLogger and delete ciScript.py
