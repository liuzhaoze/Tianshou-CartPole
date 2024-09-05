import argparse
import os
from datetime import datetime
from functools import partial

import gymnasium as gym
import numpy as np
import torch
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.highlevel.logger import LoggerFactoryDefault
from tianshou.policy import BasePolicy, DQNPolicy
from tianshou.trainer import OffpolicyTrainer
from tianshou.utils.net.common import Net


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", type=str, default="CartPole-v1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--hidden-sizes",
        type=int,
        nargs="*",
        default=[128, 128, 128, 128],
        help="Hidden layer sizes of DQN.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument(
        "--td-step", type=int, default=3, help="N-step in multi-step TD target."
    )
    parser.add_argument("--target-update-freq", type=int, default=100)
    parser.add_argument(
        "--buffer-size", type=int, default=1e5, help="Size of replay buffer."
    )
    parser.add_argument(
        "--epsilon-start",
        type=float,
        default=1.0,
        help="Start value of epsilon-greedy.",
    )
    parser.add_argument(
        "--epsilon-end", type=float, default=0.05, help="End value of epsilon-greedy."
    )
    parser.add_argument("--epsilon-test", type=float, default=0.005)
    parser.add_argument(
        "--reward-threshold",
        type=float,
        default=400.0,
        help="Reward goal for training task. Will be overwritten if `env.spec.reward_threshold` is available.",
    )
    parser.add_argument(
        "--train-env-num",
        type=int,
        default=10,
        help="Number of training environments the agent interacts with in parallel.",
    )
    parser.add_argument(
        "--test-env-num",
        type=int,
        default=10,
        help="Number of testing environments the agent interacts with in parallel.",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=50,
        help="Number of total epochs in training process.",
    )
    parser.add_argument("--step-per-epoch", type=int, default=1000)
    parser.add_argument("--step-per-collect", type=int, default=10)
    parser.add_argument("--episode-per-test", type=int, default=100)
    parser.add_argument("--update-per-step", type=float, default=0.1)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--logger", type=str, default="tensorboard", choices=["tensorboard", "wandb"]
    )
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--wandb-project", type=str, default="tianshou-cartpole")
    parser.add_argument("--resume-id", type=str, default=None)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--watch-episode-num", type=int, default=1)
    parser.add_argument("--render-fps", type=int, default=60)

    return parser.parse_known_args()[0]


def configure_log_path(args: argparse.Namespace) -> argparse.Namespace:
    args.now = datetime.now().strftime("%Y%m%d-%H%M%S")
    args.log_path = os.path.join(args.logdir, args.now)
    if args.logger == "wandb":
        os.makedirs(os.path.join(args.log_path, "wandb"), mode=755, exist_ok=True)
    return args


def get_env(args: argparse.Namespace, render_mode: str | None = None):
    return gym.make(args.task, render_mode=render_mode)


def add_env_info(args: argparse.Namespace) -> argparse.Namespace:
    env = get_env(args)
    args.state_space = env.observation_space
    args.state_shape = args.state_space.shape or int(args.state_space.n)
    args.action_space = env.action_space
    args.action_shape = args.action_space.shape or int(args.action_space.n)
    if env.spec.reward_threshold:
        args.reward_threshold = env.spec.reward_threshold
    return args


def get_policy(
    args: argparse.Namespace,
    policy: BasePolicy | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    use_best: bool = False,
) -> tuple[BasePolicy, torch.optim.Optimizer | None]:
    if policy is None:
        net = Net(
            state_shape=args.state_shape,
            action_shape=args.action_shape,
            hidden_sizes=args.hidden_sizes,
            device=args.device,
        ).to(args.device)

        if optimizer is None:
            optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)

        policy = DQNPolicy(
            model=net,
            optim=optimizer,
            action_space=args.action_space,
            discount_factor=args.gamma,
            estimation_step=args.td_step,
            target_update_freq=args.target_update_freq,
        )

    if use_best:
        path = os.path.join(args.log_path, "best.pth")
        policy.load_state_dict(torch.load(path, map_location=args.device))
        print(f"Load best policy from {path}")

    return policy, optimizer


def train(
    args: argparse.Namespace,
    policy: BasePolicy | None = None,
    optimizer: torch.optim.Optimizer | None = None,
):
    train_envs = DummyVectorEnv(
        [partial(get_env, args) for _ in range(args.train_env_num)]
    )
    test_envs = DummyVectorEnv(
        [partial(get_env, args) for _ in range(args.test_env_num)]
    )

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)

    # policy
    policy, optimizer = get_policy(args, policy, optimizer)

    # replay buffer
    buffer = VectorReplayBuffer(
        total_size=args.buffer_size, buffer_num=args.train_env_num
    )

    # collector
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(policy, test_envs, exploration_noise=True)

    # logger
    logger_factory = LoggerFactoryDefault()
    if args.logger == "wandb":
        logger_factory.logger_type = "wandb"
        logger_factory.wandb_project = args.wandb_project
    else:
        logger_factory.logger_type = "tensorboard"

    logger = logger_factory.create_logger(
        log_dir=args.log_path,
        experiment_name=os.path.join(args.task, str(args.seed), args.now),
        run_id=args.resume_id,
        config_dict=vars(args),
    )

    # train
    def train_fn(num_epoch: int, step_idx: int) -> None:
        # nature DQN setting, linear decay in the first 1M steps
        if step_idx <= 1e6:
            epsilon = args.epsilon_start - step_idx / 1e6 * (
                args.epsilon_start - args.epsilon_end
            )
        else:
            epsilon = args.epsilon_end
        policy.set_eps(epsilon)

    def test_fn(num_epoch: int, step_idx: int) -> None:
        policy.set_eps(args.epsilon_test)

    def stop_fn(mean_rewards: float) -> bool:
        return mean_rewards >= args.reward_threshold

    def save_best_fn(policy: BasePolicy) -> None:
        path = os.path.join(args.log_path, "best.pth")
        torch.save(policy.state_dict(), path)
        print(f"Save best policy to {path}")

    result = OffpolicyTrainer(
        policy=policy,
        max_epoch=args.num_epochs,
        batch_size=args.batch_size,
        train_collector=train_collector,
        test_collector=test_collector,
        step_per_epoch=args.step_per_epoch,
        step_per_collect=args.step_per_collect,
        episode_per_test=args.episode_per_test,
        update_per_step=args.update_per_step,
        train_fn=train_fn,
        test_fn=test_fn,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        logger=logger,
        resume_from_log=args.resume_id is not None,
    ).run()

    return result, policy


def watch(
    args: argparse.Namespace, policy: BasePolicy | None = None, use_best: bool = False
):
    env = DummyVectorEnv([partial(get_env, args, render_mode="human")])
    policy, optimizer = get_policy(args, policy, use_best=use_best)
    policy.eval()
    policy.set_eps(args.epsilon_test)
    collector = Collector(policy, env, exploration_noise=True)
    result = collector.collect(
        n_episode=args.watch_episode_num,
        render=1 / args.render_fps,
        reset_before_collect=True,
    )

    return result


if __name__ == "__main__":
    args = get_args()
    args = configure_log_path(args)
    args = add_env_info(args)

    result, policy = train(args)

    if args.watch:
        result = watch(args, policy, use_best=True)
