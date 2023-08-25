*************************
Installation instructions
*************************

Install from PIP
=========================

For now, everyone must install from source. Installation from PIP will be made
available soon.


Install from source
=========================

We suggest to always use the latest release, which can be found here:
https://github.com/Bayer-Group/pybalance/releases.

To install from source, clone the repository and checkout the desired release:

	>>> git clone https://github.com/Bayer-Group/pybalance.git
	>>> cd pybalance
	>>> git checkout vX.Y.Z

where vX.Y.Z is the tag of the desired release.

From here, create a virtual environment using your favorite environment manager
(e.g. condas). Within that virtual environment, install the required python
dependencies:

	>>> (yourenv) pip install -r environments/requirements.txt

If using GPU acceleration (only needed for GeneticMatcher), then install these
additional dependencies:

	>>> (yourenv) pip install -r environments/requirements_gpu.txt

Then install the pybalance code:

	>>> (yourenv) python setup.py install


Use with docker
=========================

We also maintain a docker environment for those who prefer not to use virtual
environments. If you do not need GPU acceleration, we suggest to use the
"development" Docker environment. To build this Docker image, run:

	>>> docker build -t pybal:dev -f environments/Dockerfile.dev .

To enter the development environment, run:

	>>> docker run -v /path/to/pybalance:/pybalance/ -it pybal:dev

where `/path/to/pybalance` is the local top level directory for the
repository.  Attaching the repository in this way will allow you to immediately
see changes in the code in your Docker image.

To access the jupyter environment, use docker compose:

	>>> docker-compose up jupyter

and navigate to the link printed on the screen to connect to the server. Note
that the notebook should be saved in `/pybalance` if you want it to be
persistent.

We also provide a GPU environment if you have acceess to a GPU for balance
calculation acceleration and want to use the GeneticMatcher. To build this
environment, simply run:

	>>> docker build -t pyblal:gpu -f environments/Dockerfile.gpu .
