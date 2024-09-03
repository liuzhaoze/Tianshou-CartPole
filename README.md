# Tianshou-CartPole

A demo using Tianshou to solve the Cart Pole task.

## Python Environment Setup

Create a new conda environment:

```bash
# Create a new conda environment
conda create -n tianshou python=3.11

# Activate the environment
conda activate tianshou
```

Install Tianshou:

```bash
git clone --branch v1.1.0 --depth 1 https://github.com/thu-ml/tianshou.git
cd tianshou
pip install poetry

# Change the source of poetry if necessary
poetry source add --priority=primary tsinghua https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

poetry lock --no-update
poetry install
```

Install cart-pole game:

```bash
pip install 'gymnasium[classic-control]==0.28.1'
```

Install TensorBoard and [WandB](https://wandb.ai/home) for logging:

```bash
pip install tensorboard
pip install wandb
```

## TODO

- `tianshou-cartpole.py`:
  - add `save_checkpoint_fn` and resume training
