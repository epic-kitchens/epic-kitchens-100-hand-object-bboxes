.. epic-hand-object-detections documentation master file, created by
   sphinx-quickstart on Mon Jun  1 11:36:02 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to epic-hand-object-detections's documentation!
=======================================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:



Installation
------------

.. code-block:: bash

    $ pip install git+https://github.com/epic-kitchens/epic-kitchens-100-hand-object-bboxes.git

Usage
-----

Visualise the hand-object detections like so:

.. code-block:: python

    from pathlib import Path
    import PIL.Image
    from epic_kitchens.hoa.io import load_detections
    from epic_kitchens.hoa.visualisation import DetectionRenderer

    class LazyFrameLoader:
        def __init__(self, path: Union[Path, str], frame_template: str = 'frame_{:010d}.jpg'):
            self.path = Path(path)
            self.frame_template = frame_template

        def __getitem__(self, idx: int) -> PIL.Image.Image:
            return PIL.Image.open(str(self.path / self.frame_template.format(idx + 1)))

    detections = load_detections('detections/P01_101.pkl')
    frames = LazyFrameLoader('frames/P01_101')
    renderer = DetectionRenderer(hand_threshold=0.5, object_threshold=0.5)

    frame_idx = 100
    renderer.render_detections(frames[frame_idx], detections[frame_idx])

An Jupyter notebook example is included that demonstrates how to
detections and visualise them.


Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
