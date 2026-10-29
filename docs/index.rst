.. module:: allel2

scikit-allel2 - Explore and analyse genetic variation
=====================================================

This package provides utilities for exploratory analysis of large
scale genetic variation data. It is based on numpy_, scipy_ and other
general-purpose Python scientific libraries.

scikit-allel2 continues `scikit-allel <https://github.com/cggh/scikit-allel>`_
for Python 3.12+, NumPy 2, zarr 3 and pandas 3. If you are coming from
scikit-allel, see :doc:`migrating`.

* Source: https://github.com/ltalignani/scikit-allel2
* Documentation: https://scikit-allel2.readthedocs.io/
* Download: https://pypi.org/project/scikit-allel2/

If you find a bug or would like to suggest a feature, please `raise an issue
on GitHub <https://github.com/ltalignani/scikit-allel2/issues/new>`_.

This site provides reference documentation for `scikit-allel2`. For
worked examples with real data, see the following articles:

* `Selecting variants and samples <http://alimanfoo.github.io/2018/04/09/selecting-variants.html>`_
* `Extracting data from VCF files <http://alimanfoo.github.io/2017/06/14/read-vcf.html>`_
* `A tour of scikit-allel2 <http://alimanfoo.github.io/2016/06/10/scikit-allel-tour.html>`_
* |Estimating FST|
* `Fast PCA <http://alimanfoo.github.io/2015/09/28/fast-pca.html>`_
* `Mendelian transmission <http://alimanfoo.github.io/2017/02/14/mendelian-transmission.html>`_

.. |Estimating FST| raw:: HTML

    <a href="http://alimanfoo.github.io/2015/09/21/estimating-fst.html">Estimating F<sub>ST</sub></a>

If you would like to cite `scikit-allel2`, please cite scikit-allel using the
DOI below.

.. image:: https://zenodo.org/badge/7890/cggh/scikit-allel.svg
   :target: https://zenodo.org/badge/latestdoi/7890/cggh/scikit-allel

Why "scikit-allel"?
-------------------

"`SciKits <http://www.scipy.org/scikits.html>`_" (short for SciPy Toolkits)
are add-on packages for SciPy, hosted and developed separately and
independently from the main SciPy distribution.

"Allel" (Greek ἀλλήλ) is the root of the word
"`allele <https://en.wikipedia.org/wiki/Allele>`_" short for "allelomorph", a
word coined by William Bateson to mean variant forms of a gene. Today we use
"allele" to mean any of the variant forms found at a site of genetic variation,
such as the different nucleotides observed at a single nucleotide polymorphism
(SNP).

Installation
------------

`scikit-allel2` requires Python 3.12 or later, on Linux or macOS (Windows
is not supported). It can be installed via pip::

    $ pip install scikit-allel2

This will install `scikit-allel2` along with its required dependencies
(numpy_, dask_ and zarr_). Some features of `scikit-allel2` require
optional third-party packages to be installed, including scipy_,
matplotlib_, seaborn_, pandas_, scikit-learn_ and h5py_. To
install `scikit-allel2` with all optional dependencies via pip, do::

    $ pip install scikit-allel2[optional]

If you have never installed Python before, you might find the
following article useful: `Installing Python for data analysis
<http://alimanfoo.github.io/2017/05/18/installing-python.html>`_

Contributing
------------

This is academic software, written in the cracks of free time between other commitments, by people
who are often learning as we code. We greatly appreciate bug reports, pull requests, and any other
feedback or advice. If you do find a bug, we'll do our best to fix it, but apologies in advance if
we are not able to respond quickly. If you are doing any serious work with this package, please do
not expect everything to work perfectly first time or be 100% correct. Treat everything with a
healthy dose of suspicion, and don't be afraid to dive into the source code if you have to. Pull
requests are always welcome.

Contents
--------

.. toctree::
    :maxdepth: 2

    model
    stats
    io
    chunked
    migrating
    migrating_chunked
    util
    release

Acknowledgments
---------------

Development of this package is supported by the
`MRC Centre for Genomics and Global Health <http://www.cggh.org>`_.

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _numpy: http://www.numpy.org
.. _scipy: http://www.scipy.org/
.. _matplotlib: http://matplotlib.org/
.. _seaborn: http://stanford.edu/~mwaskom/software/seaborn/
.. _pandas: http://pandas.pydata.org/
.. _scikit-learn: http://scikit-learn.org/
.. _h5py: http://www.h5py.org/
.. _numexpr: https://github.com/pydata/numexpr
.. _zarr: https://zarr.readthedocs.io/en/stable/
.. _dask: http://dask.pydata.org/
.. _petl: http://petl.readthedocs.org/
.. _cython: https://cython.org/