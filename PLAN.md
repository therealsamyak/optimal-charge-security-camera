## Introduction

our project is an adaptive camera that switches YOLO models based on battery level and clean energy input
balances accuracy, efficiency, clean energy usage in real time

the battery is fully software simulated, with each task siphoning energy from the virtual battery based on the model used

clean energy numbers are taken from the Los Angeles Department of Water and Power (LDWP) dataset, which provides 5-minute granularity carbon intensity data for the year 2024. More on how this is used later. You might not need all the information provided in the dataset, keep that in mind.

model criteria is provided by the YOLOv10 team's benchmark data in model-data.csv. You might not need all the information provided in the dataset, keep that in mind. 

The user will input their desired accuracy and latency thresholds through a root level .json file `config.jsonc`. This will also include any hardcoded constant parameters that we want to use throughout the system (that is for you to decide based on the simulation requirements).


## Evaluation

We will run the camera for “24 hour” simulations, based on historical clean energy data for multiple different days throughout the year (one in each season; 4 days total. These days are january 5th, april 15th, july 4th, october 20th). We will repeat these simulations with different user-defined accuracy and latency metrics. The camera feed will be a static image, but we will have different static images of varying quality. For simplicity, the camera model will run on a consistent interval in every simulation. For each simulation, we compare multiple different controllers:

OUR CUSTOM CONTROLLER: Picks the model based on accuracy, latency, clean energy. Battery charging also determined by this controller. This controller will use an algorithm that gives weights to the user-defined accuracy & latency, the cleanliness of the energy coming in, and the existing model benchmarks provided by the YOLOv10 team.

ORACLE (OMNISCIENT): Takes in all problem information (historical clean energy data, accuracy, latency requirements), and uses a Mixed-Integer Linear Programming (MILP) solver (python has many pre-existing libraries that do this) to maximize the amount of clean energy used ONLY. Due to the nature of MILP solvers, optimizing for a proportion won’t work, hence why we are optimizing for amount of clean energy consumed instead. Full transparency, I got this idea from AI, but even then I think it should work.

BENCHMARK: Uses the most powerful model possible in the situation, ignoring clean energy entirely, charges battery only when low. This represents a ‘brute force’ approach.

Note: there are a LOT of simulations that need to be ran, as we have many many different combinations of test cases including different days, accuracy thresholds, and latency requirements, among other things. 

With these simulations, we will take 2 key metrics:

- “miss rate”: It’s possible that the battery might not have enough energy to run the stronger models necessary to meet the accuracy + latency thresholds given. A “small miss” is when the camera model output failed to meet the thresholds given. A “large miss” is when the camera battery was fully dead and couldn’t output anything. We will document both separately.

- “total energy used” and “total clean energy used”: This is the amount of energy used (and amount of clean energy used) as a numeric amount. We can then calculate percentages, and/or weighted sums (if we use more energy, but the energy is clean, is that better/worse than less energy consumed but less clean? That extra clean energy could have gone to a hospital which then could have burned 1 less coal etc. This is something to consider but might be out of scope)
  With this approach, we believe that the benchmarks will be thorough enough to show if our controller actually works or not. If you have any feedback, please let us know. I will also think on this over the weekend and if there are any updates I will also let you know.
