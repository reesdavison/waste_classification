# Waste classification

A company collects bin bags with different colours from commercial buildings like restaurants, offices etc. The different colours of the bin bags denote their usage: General Waste, Mixed Recycling, and Compostable packaging (see below). They are then taken to their sorting facility where they need to be sorted by colour. Detect and classify the bags.

<p align="center">
  <img src=data/general_waste/0.jpg width="250" alt="Image 1">
  <img src=data/compostable_waste/0.jpg width="250" alt="Image 2">
  <img src=data/mixed_recycling/0.jpg width="250" alt="Image 3">
</p>

<p align="center">
  <em>General Waste | Compostable Waste | Mixed Recycling</em>
</p>

The dataset is a number of photos on a carpet of the different waste bags under different lighting conditions to create a small dataset comprised of: 

- Images of the bin bags e.g. 0.jpg, 1.jpg 
- Images of the background e.g. median_dark.jpg 

All images are in `/data/` in this repository. 

## Task
Define a classification algorithm and write up a report in `notebooks/waste_classification.ipynb`

## Running notebooks/waste_classification.ipynb
- In theory it should run on mac or linux with the pinned dependencies.
- I've put some effort into maintaining the Dockerfile - so that should be your easiest bet.
- Most development I did on a mac using poetry for development as a package manager.
- If you want to run locally follow the [instructions](https://python-poetry.org/docs/#installation) for installing poetry. Then run `poetry install` in a new environment of your choosing - conda, venv etc.
- Let me know if you run into any issues.
