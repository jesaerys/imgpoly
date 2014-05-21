imgpoly
=======

Determine footprints for FITS images.

This package is centered on the `find_borders` function, which calculates
the border of an input boolean array using a technique inspired by the
`marching squares algorithm <http://en.wikipedia.org/wiki/Marching_squares>`_. This
is particularly useful for astronomical CCD images, which often have a
"vignette" of non-data pixels around the edge.

A small set of helper functions is provided, each tailored to FITS images
from a particular instrument. HST/ACS/WFC, HST/WFC3, and GALEX are
currently supported. The main difference between analyses for any two
instruments is how the boolean arrays (data pixels vs. border pixels) for
`find_borders` are constructed as different detectors have different
shapes.

`imgpoly` depends on the following packages:

- `astropy <http://www.astropy.org>`_
- `numpy <http://www.numpy.org>`_
- `shapely <https://github.com/Toblerity/Shapely>`_
- `geoutil <https://github.com/jesaerys/geoutil>`_

Documentation build instructions (requires `sphinx <http://sphinx-doc.org/>`_
and `numpydoc <https://github.com/numpy/numpydoc>`_)::

  cd imgpoly/docs
  make html
  open _build/html/index.html
