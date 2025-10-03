Contributing
=====

We welcome pull requests from anyone!

Formatting
------------

Code formatting, import sorting, and spell checks are done automatically with ``black``, ``isort`` and ``codespell``.

You can automatically enforce the formatting guidelines with

.. code-block:: console

   pip install black isort codespell
   bash fixFormat.sh


Tests
------------
Please ensure your contribution passes the tests in the CI (``.github/worklows/ci.yml``).

To run the unit tests

.. code-block:: console

   conda activate bird
   pip install pytest
   BIRD_HOME=`python -c "import bird; print(bird.BIRD_DIR)"`
   cd ${BIRD_HOME}/../
   pytest .

To run the regression tests

.. code-block:: console

   source <OpenFOAM-9 installation directory>/etc/<your-shell>rc
   conda activate bird
   pip install pytest
   BIRD_HOME=`python -c "import bird; print(bird.BIRD_DIR)"`
   cd ${BIRD_HOME}/../tutorial_cases
   bash run_all.sh


Demonstrating and documenting your contribution
------------
We prefer the use of docstrings and type hinting. A good example to follow are functions in ``bird/utilities/ofio.py``.
 
If you add a new capability, please make sure to add relevant unit tests in the ``tests/`` folder. A good example to follow are tests ``tests/io``.
