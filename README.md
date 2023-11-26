# The bin bag challenge

A client called First Mile collects bin bags with different colours from commercial buildings like restaurants, offices etc. around London. The different colours of the bin bags denote their usage: General Waste, Mixed Recycling, and Compostable packaging (see below). They are then taken to their sorting facility where they need to be sorted by colour. The customer has asked us to do a quick review of the detection capabilities we could offer. 

<p align="center">
  <img src=data/general_waste/0.jpg width="250" alt="Image 1">
  <img src=data/compostable_waste/0.jpg width="250" alt="Image 2">
  <img src=data/mixed_recycling/0.jpg width="250" alt="Image 3">
</p>

<p align="center">
  <em>General Waste | Compostable Waste | Mixed Recycling</em>
</p>

The product team decides to take a number of photos on a carpet of the different bin bags under different lighting conditions to create a small dataset comprised of: 

- Images of the bin bags e.g. 0.jpg, 1.jpg 
- Images of the background e.g. median_dark.jpg 

All images are in data/ in this repository. 

## Instructions
1. Come up with a good method of classifying these 3 bag types using the code and data in this repo
2. Summarise your results in the jupyter notebook with enough detail to present to the client e.g. images / graphs
3. Leave the project / code in a good state to handover to a new member of the team - anything they might need to ramp up quickly
4. Write the majority of your code externally to the jupyter notebook, but you can import functions etc. 
3. Add tests to your code! Remember that these will need their own testing environment 

## Development Prerequisites 
To help your development, we have created this dev environment. Please feel free to modify it as you see fit - this is just to get you started (e.g. mounting volumes, adding directories, requirements etc)

To run this dev environment you will need docker and docker-compose installed

Please clone this repository and make your own private private one. When you've completed the challenge you can add us to your private repo so that we can see the code

## Getting started
1. cd ML_tech_lead_interview 
2. make dev_bin_bag_challenge

This will spin up a jupyter notebook in your browser - please use it for your development! 

Good luck! 


