from gymnasium.envs.registration import register

register(
    id="Balatro-v0",
    entry_point="balatro_gym.envs.balatro_env:BalatroEnv",
)

register(
    id="Balatro-Easy-v0",
    entry_point="balatro_gym.envs.balatro_env:BalatroEnv",
    kwargs={"config_preset": "easy"},
)

register(
    id="Balatro-Medium-v0",
    entry_point="balatro_gym.envs.balatro_env:BalatroEnv",
    kwargs={"config_preset": "medium"},
)

register(
    id="Balatro-Hard-v0",
    entry_point="balatro_gym.envs.balatro_env:BalatroEnv",
    kwargs={"config_preset": "hard"},
)
