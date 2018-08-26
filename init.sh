#!/bin/sh

pipenv --site-packages
pipenv install ".[examples]"
pipenv install -e .
