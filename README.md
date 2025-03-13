# Reproducing the results
## Prerequisites

- [Docker Community Edition (CE)](https://www.docker.com/community-edition)

## Instructions

### The computational environment (Docker image)
In your terminal, navigate to the folder where you've extracted the capsule and execute the following command:
```shell
cd environment && docker build . --tag 51487ce3-72c6-4403-ac0c-f597d673d51f; cd ..
```

> This step will recreate the environment (i.e., the Docker image) locally, fetching and installing any required dependencies in the process. If any external resources have become unavailable for any reason, the environment will fail to build.

### Running the capsule to reproduce the results
In your terminal, navigate to the folder where you've extracted the capsule and execute the following command, adjusting parameters as needed:
```shell
docker run --platform linux/amd64 --rm \
  --workdir /code \
  --volume "$PWD/data":/data \
  --volume "$PWD/code":/code \
  --volume "$PWD/results":/results \
  51487ce3-72c6-4403-ac0c-f597d673d51f bash run
```
# For Developers

