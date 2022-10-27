![Pytest workflow](https://github.com/TheNoobInventor/sudoku-ai-solver/actions/workflows/.github/workflows/pytest.yml/badge.svg) &ensp; ![Mkdocs workflow](https://github.com/TheNoobInventor/sudoku-ai-solver/actions/workflows/.github/workflows/docs.yml/badge.svg)
# Sudoku AI Solver

<p align='center'>
    <img src='docs/images/sudoku-ai-solver.gif'>
</p>

## Docker container

The main Jupyter notebook, `sudoku_puzzle_extractor.ipynb`, and relevant files needed for this project can be run in a Docker container. 

First pull the image (with a compressed size of 1.93 GB) from the Docker Hub repository:
```
docker pull thenoobinventor/sudoku-ai-solver:latest
```

Then run a container (choose a name for it) based on the image:

```
docker run -it --rm -p 8890:8890 --name container_name thenoobinventor/sudoku-ai-solver:latest
```

The documentation for this project can be found [here](https://TheNoobInventor.github.io/sudoku-ai-solver/).

