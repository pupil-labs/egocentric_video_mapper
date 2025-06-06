.. image:: https://img.shields.io/pypi/v/skeleton.svg
   :target: `PyPI link`_

.. image:: https://img.shields.io/pypi/pyversions/skeleton.svg
   :target: `PyPI link`_

.. _PyPI link: https://pypi.org/project/skeleton

.. image:: https://github.com/jaraco/skeleton/workflows/tests/badge.svg
   :target: https://github.com/jaraco/skeleton/actions?query=workflow%3A%22tests%22
   :alt: tests

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
   :alt: Code style: Black

.. .. image:: https://readthedocs.org/projects/skeleton/badge/?version=latest
..    :target: https://skeleton.readthedocs.io/en/latest/?badge=latest

.. image:: https://img.shields.io/badge/skeleton-2022-informational
   :target: https://blog.jaraco.com/skeleton

Introduction
============
This project allows you to obtain gaze provided by Pupil Labs Neon eye tracker in an alternative ego-centric camera view.

Requirements
============
You should have Python 3.10 or higher installed in your system. You should also have an available GPU. 
This repository assumes a minimum technical knowledge of the command line and Python, if you are not familiar with them, or if you do not have access to a local GPU, we recommend that you use our Google Colab notebook instead:

.. image:: https://img.shields.io/static/v1?label=&message=Open%20in%20Google%20Colab&color=blue&labelColor=grey&logo=Google%20Colab&logoColor=#F9AB00
   :target: https://colab.research.google.com/drive/1PixYZFYm5O2Uc3sG5X2WHpPUg1DdfeV3?usp=sharing

Installation
============
Open the terminal and go to the directory where you would like to clone the repository
::
   cd /path/to/your/directory

Clone the repository by running the following command from the terminal:
::
   git clone git@github.com:pupil-labs/egocentric_video_mapper.git

Create a virtual environment for the project:
::
   python3.10 -m venv egocentric_mapper
   source egocentric_mapper/bin/activate

Or if you are using conda:
::
   conda create -n egocentric_mapper python=3.10
   conda activate egocentric_mapper

Go to the project directory:
::
   cd /path/to/your/directory/egocentric_video_mapper
   pip install -e .

Download the directory with model weights for `EfficientLOFTR <https://github.com/zju3dv/EfficientLoFTR/>`__ from the following `download link <https://drive.google.com/drive/folders/1GOw6iVqsB-f1vmG6rNmdCcgwfB4VZ7_Q>`__  and place it in the src/pupil_labs/egocentric_video_mapper/efficient_loftr directory.

Run it!
============
To run the project, you can open 'PL-mapper.ipynb' in the IDE of your choice and run the cells. Conversely, you can run the following command from the terminal and it will open the notebook in your browser ready to run the cells:
::
   jupyter notebook --port=9000 src/pupil_labs/egocentric_video_mapper/PL-mapper.ipynb


Support
========

For any questions/bugs, reach out to our `Discord server <https://pupil-labs.com/chat/>`__  or send us an email to info@pupil-labs.com. 
