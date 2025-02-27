# Evolutionary Algorithm

This project implements an evolutionary algorithm to optimize functions from the CEC 2017 benchmark suite. The algorithm includes features such as mutation, reproduction, and selection to evolve a population of individuals towards an optimal solution.

## Project Structure

- `evolutionaryAlgorithm.py`: The main script containing the implementation of the evolutionary algorithm, including the `Individual` and `Population` classes, and functions for running the algorithm and generating plots.
- `Classic evolutionary algorithm results.pdf`: It consists results for various starting parameters and talking about the implementation of the algorithm

## Requirements

- Python 3.6+
- NumPy
- Matplotlib
- cec2017

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/evolutionary-algorithm.git
    cd evolutionary-algorithm
    ```

2. Install the required packages:
    ```sh
    pip install numpy matplotlib
    ```

## Usage

To run the evolutionary algorithm and generate a plot of the optimization process, execute the [evolutionaryAlgorithm.py](http://_vscodecontentref_/0) script:

```sh
python evolutionaryAlgorithm.py
