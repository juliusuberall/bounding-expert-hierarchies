# Bounding Expert Hierarchies
> [Julius Überall](https://juliusuberall.com/), [Tobias Ritschel](https://www.homepages.ucl.ac.uk/~ucactri/) <br>
> University College London <br>
> X (__X__), September 2025 <br>
> [Project page]() | [Paper]() | [Video]() | [Presentation]() | [BibTeX]()

Python/JAX implementation of Bounding Expert Hierarchies, using neural networks to represent and learn bounding volumes of 2D, 3D, 4D and 4D+ spaces. Using Mixture of Experts (MoE) the data is distributed and learnt by multiple expert neural networks such that they indiviudally learn a fraction of the data and collectivly learn the whole data.

![Paper thumbnail](docs/4D_thumbnail.png)

## Getting Started
<details>
  <summary><strong>Clone</strong></summary>

&nbsp;<br>
Clone this repository using the command line:
```
git clone https://github.com/juliusuberall/bounding-expert-hierarchies.git
```
</details>
<details>
  <summary><strong>Install Dependencies</strong></summary>

&nbsp;<br>
> *The main dependencies we have are [JAX](https://docs.jax.dev/en/latest/index.html) and [Optax](https://optax.readthedocs.io/en/latest/), since these are used for the implementation of the neural networks. Everything else could be considered 'standard'.*

It is recommended to create a new python environment for the project to keep things tidy and manage python dependencies. 
Navigate to the root directory of the cloned reposiroty:
```bash
cd bounding-expert-hierarchies
```
Create a virtual environment in the cloned repository by using the command line:
```
python3 -m venv venv
```
Activate the new environment:
```
source venv/bin/activate
```
Install all dependencies specified with *requirements.txt* in the environment:
```
pip install -r requirements.txt
```
</details>

<details>
  <summary><strong>Run Experiments</strong></summary>

&nbsp;<br>
This section will focus on running the implemented experiments from the project. It introduces the experiment and implementation flow used in this repository and may help for setting up custom studies. 
```
Execute some shell script triggering all python code and producing numerical and visual results
```
</details>

<details>
  <summary><strong>Custom Experiments</strong></summary>

&nbsp;<br>
This section will focus on setting up your own experiments and / or replacing the experiment data.
</details>

<details>
  <summary><strong>Repository Structure</strong></summary>
    
```
bounding-expert-hierarchies/
│── .vscode/                  # Visual Studio Code Launch settings
│   ├── launch.json           # Debugging profiles
│   ├── tasks.json            # Task definition for full pipeline execution
│── configs/...               # Stores YAML configurations for all model architectures
│── data/...                  # 2D, 3D, 4D and 4D+ data e.g. image, geometry, samples 
│── docs/...                  # Github and project page content 
│── scripts/                  # All main scripts for execution
│   ├── analyze_inference.py  # Analyzes inference of models e.g. speed, accuracy 
│   ├── format_results.py     # Processes and visualizes analysis results
│   ├── train_models.py       # Instantiates models and trains until saturation
│── src/                      # Core source code and python module 
│   ├── dataloader.py         # Data loader 
│   ├── moe.py                # Mixture of Experts (MoE) Implementation 
│   ├── mlp.py                # Multilayer Perceptron (MLP) Implementation
│── .gitignore
│── requirements.txt
│── README.md 
```    
</details>

## Data
<details>
  <summary><strong>2D</strong></summary>

&nbsp;<br>
</details>

<details>
  <summary><strong>3D</strong></summary>

&nbsp;<br>
</details>

<details>
  <summary><strong>4D</strong></summary>

&nbsp;<br>
</details>

<details>
  <summary><strong>4D+</strong></summary>

&nbsp;<br>
</details>

## Citation

```bibtex
@article{something_interesting,
   author = {xxx, XXX},
   title = {xxx},
   booktitle = {xxx},
   year = {xxx},
   location = {xxx},
   publisher = {xxx},
   address = {xxx},
   pages = {xxx},
   doi = {xxx}
}
```
