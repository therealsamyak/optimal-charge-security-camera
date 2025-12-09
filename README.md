## Optimal Charge Security Camera

We have a battery-powered security camera that runs local pre-trained image-recognition models. A battery is attached both to the charger, and the camera. The camera is context aware about clean energy information externally. Based on the information, we have a controller that looks at current battery percentage, information about external energy, and user-defined metrics (minimum accuracy, minimum latency, how often we run the model etc.), to decide which image-recognition model is loaded based on all these factors. We want to decide when to charge the battery vs when to use the energy to load a higher model for higher accuracy.

THE GOAL OF THIS CODEBASE IS TO OBTAIN A MODEL WITH GOOD WEIGHTS SO IT CAN BE RAN GENERAL-USE IN ALL SCENARIOS TO OPTIMIZE THIS.

### Image Models

The image models our controller will pick from are the YOLOv10 series of models, whose specifications are found in the `model-data/` folder.

### Clean Energy Data

The clean energy data is found in the `energy-data/` directory, and includes data over the year 2024 in 5-minute intervals. There are 4 CSVs, each corresponding to a region of the USA.

### Battery

The battery will be software simulated based on real-world battery behavior. We need a test-benchmark to measure the power usage of each model task for all model sizes, which we can then use to simulate the battery behavior. Apparently, this can be done by measuring the CPU clock or some sort of API, I am not 100% sure, we need to figure out a comprehensive way to measure this.

Once we have information, we can assign a global 'battery level' (4000 mAh to approximate modern small devices), and assign costs to each model based on the power usage we found.

Additionally, we need to simulate battery charging. Modern devices support up to 100W+ charging over USB-C, so we need to quantify and apply this to our full benchmark / simulation.

### Custom Controller

Our custom controller is as described in the earlier section, where it takes in the data, processes it internally somehow, then outputs a model which is then loaded into our camera.

Because this is a weight optimization problem, we should be able to do some machine learning approach to find the best weights, using our clean energy datasets and a MIPS solver (similar to the ORACLE controller mentioned later in this document).

Using those weights, we should be able to implement the controller and abstract it out as a function, such that we input all necessary information, and it ouputs the model it picks.

### Benchmarking

#### Simulation Structure

The task given to all models is to detect and classify the object in two different images, both found under the `benchmark-images/` directory. The output for both images should be 1 human only.

There will be multiple simulations, each with different user-defined accuracy and latency metrics.

We will run the simulation for '7 days', with a task call every 5 second. Note that we get a new clean energy datapoint every 5 minutes, so keep using the latest available clean energy datapoint that is not in the future until 5 minutes have passed, then use the new one.

We will run the simulation for 4 weeks, one from every season.

#### Controllers

There are 4 different controllers, with potentially more to be added later:

- **OUR CUSTOM CONTROLLER**: Picks the model based on accuracy, latency, clean energy. Battery charging also determined by this controller. This controller will use an algorithm that gives weights to the user-defined accuracy & latency, the cleanliness of the energy coming in, and the existing model benchmarks provided by the YOLOv10 team.
- **ORACLE (OMNISCIENT)**: Takes in all problem information (historical clean energy data, accuracy, latency requirements), and uses a Mixed-Integer Linear Programming (MILP) solver (python has many pre-existing libraries that do this) to maximize the amount of clean energy used ONLY. Due to the nature of MILP solvers, optimizing for a proportion won’t work, hence why we are optimizing for amount of clean energy consumed instead. Because this is the oracle controller, it can see the future and has full system knowledge of the clean energy information of that day, hence why an MILP solver is appropriate. _We don't expect our custom controller to beat it_.
- **NAIVE WEAK**: Always picks the smallest model, charges battery only when necessary.
- **NAIVE STRONG**: Always picks the largest model, charges battery only when necessary.
- There is a chance that we will add more controllers later based on pre-existing research papers, but for now do not worry about it. Just ensure that any implementation is future-proof such that its easier to add more later.

#### Metrics

- "Small Miss Rate": Number of times the model failed the task due to incorrect model output (ex. if the task is human recognition, the model successfully ran but did not recognize a human)
- "Large Miss Rate": Number of times the model failed the task due to insufficient energy (ex. the battery was out of charge so the model couldn't run at all)
- "Total Energy Used": Total amount of energy used by the battery over the course of the simulation.
- "Total Clean Energy Used": Total amount of clean energy used by the battery over the course of the simulation.

In short, we have **4 weeks, 4 locations, 4 controllers, 3 different accuracy/latency metrics, for a grand total of 192 simulations**.

### Major High-Level Deliverables

- Code to benchmark power consumption of each model task call. (already DONE)
- Code to train the custom controller.
- Code to run the simulations, and output a JSON file with metrics described in earlier sections. (which integrates the above 2 deliverables)

#### Important Code-related Instructions

- Ensure all code has detailed logging, so we can see what's happening at each step.
- Ensure that all code is future-proof and can be easily modified to add more controllers, metrics, or locations. Additionally ensure that it follows best practices with Python.
- Ensure all code / files are organized appropriately, according to best engineering practices with object-oriented and/or functional programming.

## Citations

Electricity Maps. "United States California LDWP 2024 5 minute Carbon Intensity Data". Electricity Maps, ver. July 2, 2025, https://www.electricitymaps.com.

Electricity Maps. "United States Florida FPL 2024 5 minute Carbon Intensity Data". Electricity Maps, ver. July 2, 2025, https://www.electricitymaps.com.

Electricity Maps. "United States Northwest PSEI 2024 5 minute Carbon Intensity Data". Electricity Maps, ver. July 2, 2025, https://www.electricitymaps.com.

Electricity Maps. "United States New York NYIS 2024 5 minute Carbon Intensity Data". Electricity Maps, ver. July 2, 2025, https://www.electricitymaps.com.
