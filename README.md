# Gym-Kniffel

This is a [Gym](https://gym.openai.com/)-compliant implementation of the "dice-poker" game Kniffel® (which is just the german version of the widely-known Yahtzee®). 

I think Kniffel can serve as an interesting problem to reinforcement learning. This is mostly because of the high variance in the higher score ranges (sometimes you're just unlucky) and the possibility of doing extremely stupid stuff even if you're lucky in your rolls.

## Example

As stated above, this is a [Gym](https://gym.openai.com/)-compliant environment, so you can use it with the most machine learning libraries. The manual usage is as follows:

```python
from kniffel import Kniffel

env = Kniffel(seed=1337)
observation = env.reset()
done = False
while not done:
  action = env.action_space.sample()  # normally you would use `observation` to infer an action
  observation, reward, done, info = env.step(action)
```

A call to `env.render()` at any time results in an ansi encoded representation of the current board state (including the dices and how many rolls you have remaining). The slots which are valid moves for the current dice faces are rendered bold.

## Spaces

While you have the power to implement your own Action- and Observation spaces (see below), you also can just use the ones I have pre-implemented.

### Observation Space

The default observation space is a `gym.spaces.Dict` consisting of the following spaces:

#### num\_rolls\_remaining

A `gym.spaces.Discrete` which encodes how many re-rolls are left. Because the environment automatically rolls for the first time this part of the observation space can just be a 1 or a 0.

#### dices:

A `gym.spaces.MultiDiscrete` which encodes the current dices -- every single one from a 0 to a 5.

#### board:

A 1-dimensional `gym.spaces.Box` which resembles the current board state in points. A 0 in a slot can mean both that it's crossed out or just not filled yet.

#### filled\_slots:

A `gym.spaces.MultiBinary` which indicates wether a slot has been used up (1) or not (0). This can act like a mask for the `board`-Observation.

#### slots\_value:

A 1-dimensional `gym.spaces.Box` which contains the value any given slot would yield if you would use the current dices for it.

### Action Space

Just like the observation space, the action space is a `gym.spaces.Dict` with the following components:

#### select\_action

A `gym.spaces.Discrete` binary number which selects what will be done: rerolled (0) or written in a slot (1).

#### dices\_hold

A `gym.spaces.MultiBinary` which controls the dice that will be held on a reroll (1) and whose who would not (0).

#### board\_selection

A `gym.spaces.Discrete` which controls the slot that will be used in case of a `select_action` of 1.

## Rewards

The default reward will be given as the delta of the score before and after the selected action, so the bonus of the upper board or multiple kniffels will be taken into account.
Also, that means, a reroll is always a 0.

If an action is selected that would be illegal (like selecting a slot that has already been filled) the given reward is a -1.

## Extensibility

You can define your own custom observations, actions and rewards by subclassing `KniffelBase`. Until I write more documentation for that, you have to look at the code (and how `Kniffel` is implemented).

## Speed

This implementation is built with speed in mind (because of the many sample inefficient reinforcement learning algorithms). On my machine (a late 2016 MacBook Pro with a *3,3 GHz Intel Core i7*), a full game of Kniffel using only random actions takes around 12 milliseconds. You can test it on your system with the following command:

```txt
$ python3 -m timeit -n 100\
  -s 'from itertools import repeat; from kniffel import Kniffel; e = Kniffel(); d = False'\
  'e.reset(); any(d for _, _, d, _ in (e.step(e.action_space.sample()) for _ in repeat(0)))'
100 loops, best of 5: 12 msec per loop
```

## Disclaimer

As of now, everything is implemented and usable, but not tested or fully documented. Interfaces might change later. Also, I will release this as a package some time later.
