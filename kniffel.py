#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""This module implements a typed Environment for the dice game "Kniffel".

The German Kniffel is the same game as the american Yahtzee,
I just like the german name better.
"""

from typing import Mapping, Any, AnyStr
from collections import defaultdict

from abc import abstractmethod

import numpy as np

import gym


class KniffelError(Exception):
    """The Exception that will be raised if the user tries anything illegal.
    """


class KniffelBase(gym.Env):
    """An extendable OpenAI Gym compliant Environment for Kniffel.
    """
    observation_space: gym.spaces.Space
    action_space: gym.spaces.Space

    metadata: Mapping[AnyStr, Any] = {'render.modes': ['ansi']}

    _max_points = [5, 10, 15, 20, 25, 30, 30, 30, 25, 30, 40, 50, 30]

    _slot_validators = [
        # Upper Board
        lambda self: self._dice_frequencies[0] > 0,
        lambda self: self._dice_frequencies[1] > 0,
        lambda self: self._dice_frequencies[2] > 0,
        lambda self: self._dice_frequencies[3] > 0,
        lambda self: self._dice_frequencies[4] > 0,
        lambda self: self._dice_frequencies[5] > 0,
        # Dreierpasch
        lambda self: np.max(list(self._dice_frequencies.values())) >= 3,
        # Viererpasch
        lambda self: np.max(list(self._dice_frequencies.values())) >= 4,
        # Full-House
        lambda self: (2 in self._dice_frequencies.values()\
                      and 3 in self._dice_frequencies.values())\
                     or 5 in self._dice_frequencies.values(),
        # Kleine Straße
        lambda self: any(v in [[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5]]
                         for v in (self._dices_sorted[0:4], self._dices_sorted[1:5])),
        # Große Straße
        lambda self: self._dices_sorted in ([0, 1, 2, 3, 4], [1, 2, 3, 4, 5]),
        # Kniffel
        lambda self: 5 in self._dice_frequencies.values(),
        # Chance
        lambda self: True,
    ]

    _slot_points = [
        # Upper board
        lambda self: np.sum(self._dices == 0) * 1,
        lambda self: np.sum(self._dices == 1) * 2,
        lambda self: np.sum(self._dices == 2) * 3,
        lambda self: np.sum(self._dices == 3) * 4,
        lambda self: np.sum(self._dices == 4) * 5,
        lambda self: np.sum(self._dices == 5) * 6,
        # Dreierpasch
        lambda self: np.sum(self._dices) + 5,
        # Viererpasch
        lambda self: np.sum(self._dices) + 5,
        # Full-House
        lambda self: 25,
        # Kleine Straße
        lambda self: 30,
        # Große Straße
        lambda self: 40,
        # Kniffel
        lambda self: 50,
        # Chance
        lambda self: np.sum(self._dices) + 5,
    ]

    def __init__(self):
        self.reset()

    def __init_subclass__(cls):
        assert getattr(cls, "action_space", None) is not None,\
            "Please define an action_space in {cls.__class__.__name__}!"
        assert getattr(cls, "observation_space", None) is not None,\
            "Please define an observation_space in {cls.__class__.__name__}!"

    @property
    def _bonus(self):
        """Calculates the current bonus of the upper board state.
        """
        return 35 if np.sum(np.maximum(self._upper, 0)) >= 63 else 0

    @property
    def _slots_value(self):
        """Returns the value for all slots with the current dices.

        An already filled slot is always a 0.
        """
        return np.fromiter((points(self) if (valid(self) and not filled) else 0
                            for valid, points, filled in
                            zip(self._slot_validators, self._slot_points, self._filled_mask)),
                           dtype=np.int64)

    def render(self, mode='ansi'):
        assert mode == 'ansi', f"Rendering mode '{mode}' is not supported!"

        bonus = self._bonus
        upper = np.sum(np.maximum(self._upper, 0))
        lower = np.sum(np.maximum(self._lower, 0))

        def fmt(val, filled):
            if val == 0:
                return 'xxx' if filled else ''
            return val

        d_t = ["   ", "o  ", "o  ", "o o", "o o", "o o"]
        d_m = [" o ", "   ", " o ", "   ", " o ", "o o"]
        d_b = ["   ", "  o", "  o", "o o", "o o", "o o"]

        can_be_filled = [val(self) for val in self._slot_validators]
        print( # yeah.. not pretty. it is one statement, though! TODO pretty
            "╔═══════════════╦═════╗" "╔═════╗\n"
            f"║ \033[{'1' if can_be_filled[0] else ''}mEinser\033[0m  ⚀ ⚀ ⚀ " "║ {:>3} ║" f"║ {d_t[self._dices[0]]} ║\n"
            f"║ \033[{'1' if can_be_filled[1] else ''}mZweier\033[0m  ⚁ ⚁ ⚁ " "║ {:>3} ║" f"║ {d_m[self._dices[0]]} ║\n"
            f"║ \033[{'1' if can_be_filled[2] else ''}mDreier\033[0m  ⚂ ⚂ ⚂ " "║ {:>3} ║" f"║ {d_b[self._dices[0]]} ║\n"
            f"║ \033[{'1' if can_be_filled[3] else ''}mVierer\033[0m  ⚃ ⚃ ⚃ " "║ {:>3} ║╠═════╣\n"
            f"║ \033[{'1' if can_be_filled[4] else ''}mFünfer\033[0m  ⚄ ⚄ ⚄ " "║ {:>3} ║" f"║ {d_t[self._dices[1]]} ║\n"
            f"║ \033[{'1' if can_be_filled[5] else ''}mSechser\033[0m ⚅ ⚅ ⚅ " "║ {:>3} ║" f"║ {d_m[self._dices[1]]} ║\n"
            "╟───────────────╫─────╢" f"║ {d_b[self._dices[1]]} ║\n"
            f"║ Gesamt        ║ {upper:>3} ║╠═════╣\n"
            f"║ Bonus         ║ {bonus:>3} ║║ {d_t[self._dices[2]]} ║\n"
            f"║ Gesamt oben   ║ {upper+bonus:>3} ║║ {d_m[self._dices[2]]} ║\n"
            f"╠═══════════════╬═════╣║ {d_b[self._dices[2]]} ║\n"
            f"║ \033[{'1' if can_be_filled[6] else ''}mDreierpasch\033[0m   " "║ {:>3} ║╠═════╣\n"
            f"║ \033[{'1' if can_be_filled[7] else ''}mViererpasch\033[0m   " "║ {:>3} ║" f"║ {d_t[self._dices[3]]} ║\n"
            f"║ \033[{'1' if can_be_filled[8] else ''}mFull-House\033[0m    " "║ {:>3} ║" f"║ {d_m[self._dices[3]]} ║\n"
            f"║ \033[{'1' if can_be_filled[9] else ''}mKleine Straße\033[0m " "║ {:>3} ║" f"║ {d_b[self._dices[3]]} ║\n"
            f"║ \033[{'1' if can_be_filled[10] else ''}mGroße Straße\033[0m  " "║ {:>3} ║╠═════╣\n"
            f"║ \033[{'1' if can_be_filled[11] else ''}mKniffel\033[0m       " "║ {:>3} ║" f"║ {d_t[self._dices[4]]} ║\n"
            f"║ \033[{'1' if can_be_filled[12] else ''}mChance\033[0m        " "║ {:>3} ║" f"║ {d_m[self._dices[4]]} ║\n"
            f"╟───────────────╫─────╢║ {d_b[self._dices[4]]} ║\n"
            f"║ Gesamt unten  ║ {lower:>3} ║╚═════╝\n"
            f"║ Gesamt oben   ║ {upper+bonus:>3} ║\n"
            f"║ Endstand      ║ {lower+upper+bonus:>3} ║\n"
            "╚═══════════════╩═════╝"
            .format(*(fmt(val, filled) for val, filled
                      in zip(self._board, self._filled_mask)))
        )

    def _roll(self, mask=None):
        """Rolls the dice and decrements the remaining roll counter.

        The mask is a boolean array specifying which dice you want to hold.
        If the mask isn't specified, it counts as a full reroll.
        If no rolls are available or everything is held, raise a KniffelError.
        """
        if self._num_rolls_remaining <= 0:
            raise KniffelError("You don't have rolls left!")
        mask = np.asarray(mask if mask is not None else [False] * 5)
        if mask.all():
            raise KniffelError("Keeping everything is not a valid move!")
        self._dices[~mask] = np.random.randint(low=0, high=6, size=(5,))[~mask]
        self._num_rolls_remaining -= 1

        # Count the dices because some detections are easier with them
        self._dice_frequencies.clear()
        for dice in self._dices:
            self._dice_frequencies[dice] += 1

        # Sort the dices because some detections are easier with them
        self._dices_sorted = sorted(self._dices)

    def _points_at(self, index):
        """Calculates a mask for the board that shows current possible moves.
        """
        if self._slot_validators[index](self):
            return self._slot_points[index](self)
        return 0

    def _select(self, index):
        """Fills a given slot in the board.

        Writes the current rolled dices in the given slot index.
        If nothing is possible in that slot, it counts as crossed out.

        If the slot is already filled, raise a KniffelError.
        If no Error is raised, also re-roll and reset the counter.
        """
        if self._filled_mask[index]:
            raise KniffelError("The selected field already is filled!")
        self._filled_mask[index] = True

        # in the case of multiple kniffels, give 50 extra points and a joker
        # for a given field which counts as the best possible value
        if self._filled_mask[11] and self._board[11] > 0 and self._slot_validators[11](self):
            self._board[11] += 50
            self._board[index] = self._max_points[index]
        else:
            # write the points in the given slot
            points = self._points_at(index)
            self._board[index] = points

        # re-roll
        self._num_rolls_remaining = 3
        self._roll()

    def reset(self):
        self._board = np.zeros((6 + 7,), dtype=np.int64)
        self._filled_mask = np.zeros(self._board.shape, dtype=np.int64)
        self._upper = self._board[:6]
        self._lower = self._board[-7:]

        self._num_rolls_remaining = 3
        self._dices = np.zeros((5,), dtype=np.int64)
        self._dice_frequencies = defaultdict(int)
        self._roll()

        return self.observe()

    def step(self, action):
        bonus = self._bonus
        score = np.sum(np.maximum(self._board, 0))
        reward = self.act(action)
        return self.observe(),\
            reward,\
            (self._filled_mask > 0).all(),\
            {"full_score": score+bonus}

    @abstractmethod
    def act(self, action) -> float:
        """Steps the world with a given action in respective to self.action_space.
        """
        raise NotImplementedError

    @abstractmethod
    def observe(self):
        """Returns the current state in respective to self.observation_space.
        """
        raise NotImplementedError


class Kniffel(KniffelBase):
    """A default implementation of Kniffel with simple dict spaces.
    """
    action_space: gym.spaces.Dict = gym.spaces.Dict({
        "dices_hold": gym.spaces.MultiBinary(5),
        "board_selection": gym.spaces.Discrete(13),
        "select_action": gym.spaces.Discrete(2),
    })

    observation_space: gym.spaces.Dict = gym.spaces.Dict({
        "board": gym.spaces.Box(low=0, high=50, shape=(13,), dtype=np.int64),
        "filled_slots": gym.spaces.Box(low=0, high=50, shape=(13,), dtype=np.int64),
        "slots_value": gym.spaces.Box(low=0, high=50, shape=(13,), dtype=np.int64),
        "num_rolls_remaining": gym.spaces.Discrete(3),
        "dices": gym.spaces.MultiDiscrete([6] * 5)
    })

    def observe(self):
        return {
            "board": self._board,
            "filled_slots": self._filled_mask,
            "slots_value": self._slots_value,
            "num_rolls_remaining": self._num_rolls_remaining,
            "dices": self._dices,
        }

    def act(self, action) -> float:
        pre = np.sum(self._board) + self._bonus
        try:
            if action['select_action'] == 0:
                self._roll(action['dices_hold'])
            else:
                self._select(action['board_selection'])
        except KniffelError:
            # On a KniffelError do nothing, but reward a -1.
            return -1
        post = np.sum(self._board) + self._bonus
        return post - pre

def play_kniffel():
    """Calling the module should let you play kniffel interactively as a human.

    Right now it doesn't.
    """
    env = Kniffel()
    observation = env.reset()
    done = False
    steps = 0
    rewards = []
    while not done:
        steps += 1

        # move random but select the best possible slots
        action = env.action_space.sample()
        if any(m > 0 for m in observation['slots_value']):
            action['select_action'] = 1
            action['board_selection'] = np.argmax(observation['slots_value'])

        observation, reward, done, info = env.step(action)
        if done:
            env.reset()
            done = False
        rewards.append(reward)

    env.render()
    print(f"Board full in {steps} actions "
          f"with {info['full_score']} Points "
          f"(and a reward of {sum(rewards)})!")


if __name__ == "__main__":
    play_kniffel()
