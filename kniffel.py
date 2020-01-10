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

    metadata: Mapping[AnyStr, Any] = {'render.modes': ['human']}

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

    def render(self, mode='human'):
        assert mode == 'human', f"Rendering mode '{mode}' is not supported!"

        bonus = self._bonus
        upper = np.sum(np.maximum(self._upper, 0))
        lower = np.sum(np.maximum(self._lower, 0))

        def fmt(val, filled):
            if val == 0:
                return 'xxx' if filled else ''
            return val

        dices = defaultdict(int)
        for dice in self._dices:
            dices[dice] += 1

        print(
            "╔═══════════════╦═════╗\n"
            "║ Einser  ⚀ ⚀ ⚀ ║ {:>3} ║"
            f" {''.join('⚀' for _ in range(dices[0]))}\n"
            "║ Zweier  ⚁ ⚁ ⚁ ║ {:>3} ║"
            f" {''.join('⚁' for _ in range(dices[1]))}\n"
            "║ Dreier  ⚂ ⚂ ⚂ ║ {:>3} ║"
            f" {''.join('⚂' for _ in range(dices[2]))}\n"
            "║ Vierer  ⚃ ⚃ ⚃ ║ {:>3} ║"
            f" {''.join('⚃' for _ in range(dices[3]))}\n"
            "║ Fünfer  ⚄ ⚄ ⚄ ║ {:>3} ║"
            f" {''.join('⚄' for _ in range(dices[4]))}\n"
            "║ Sechser ⚅ ⚅ ⚅ ║ {:>3} ║"
            f" {''.join('⚅' for _ in range(dices[5]))}\n"
            "╟───────────────╫─────╢\n"
            f"║ Gesamt        ║ {upper:>3} ║\n"
            f"║ Bonus         ║ {bonus:>3} ║\n"
            f"║ Gesamt oben   ║ {upper+bonus:>3} ║\n"
            "╠═══════════════╬═════╣\n"
            "║ Dreierpasch   ║ {:>3} ║\n"
            "║ Viererpasch   ║ {:>3} ║\n"
            "║ Full-House    ║ {:>3} ║\n"
            "║ Kleine Straße ║ {:>3} ║\n"
            "║ Große Straße  ║ {:>3} ║\n"
            "║ Kniffel       ║ {:>3} ║\n"
            "║ Chance        ║ {:>3} ║\n"
            "╟───────────────╫─────╢\n"
            f"║ Gesamt unten  ║ {lower:>3} ║\n"
            f"║ Gesamt oben   ║ {upper+bonus:>3} ║\n"
            f"║ Endstand      ║ {lower+upper+bonus:>3} ║\n"
            "╚═══════════════╩═════╝"
            .format(*(fmt(val, filled)
                      for val, filled
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

        # Count the dices because some detections are easier with them
        dices = defaultdict(int)
        for dice in self._dices:
            dices[dice] += 1

        # Upper board
        if index <= 5:
            self._board[index] = np.sum(self._dices == index) * (index + 1)

        # Dreierpasch
        elif index == 6:
            if np.max(list(dices.values())) >= 3:
                self._board[index] = np.sum(self._dices) + 5

        # Viererpasch
        elif index == 7:
            if np.max(list(dices.values())) >= 4:
                self._board[index] = np.sum(self._dices) + 5

        # Full-House
        elif index == 8:
            if 5 in dices.values()\
              or (2 in dices.values() and 3 in dices.values()):
                self._board[index] = 25

        # Kleine Straße
        elif index == 9:
            sort = sorted(self._dices)
            if any(v in [[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5]]
                   for v in (sort[0:4], sort[1:5])):
                self._board[index] = 30

        # Große Straße
        elif index == 10:
            sort = sorted(self._dices)
            if sort in ([0, 1, 2, 3, 4], [1, 2, 3, 4, 5]):
                self._board[index] = 40
        # Kniffel
        elif index == 11:
            if 5 in dices.values():
                self._board[index] = 50
                # TODO: multiple kniffels!

        # Chance
        elif index == 12:
            # chance is always valid
            self._board[index] = np.sum(self._dices) + 5

        self._num_rolls_remaining = 3
        self._roll()

    def reset(self):
        self._board = np.zeros((6 + 7,), dtype=np.int64)
        self._filled_mask = np.zeros(self._board.shape, dtype=np.int64)
        self._upper = self._board[:6]
        self._lower = self._board[-7:]

        self._num_rolls_remaining = 3
        self._dices = np.zeros((5,), dtype=np.int64)
        self._roll()

        return self.observe()

    def step(self, action):
        bonus = self._bonus
        score = np.sum(np.maximum(self._board, 0))
        self.act(action)
        return self.observe(),\
            None,\
            (self._filled_mask > 0).all(),\
            {"full_score": score+bonus}

    @abstractmethod
    def act(self, action):
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
        "board_selection": gym.spaces.Discrete(6 + 7),
        "select_action": gym.spaces.Discrete(2),
    })

    observation_space: gym.spaces.Dict = gym.spaces.Dict({
        "num_rolls_remaining": gym.spaces.Discrete(3),
        "dices": gym.spaces.MultiDiscrete([6, 6, 6, 6, 6])
    })

    def observe(self):
        return {
            "board": self._board,
            "board_mask": self._filled_mask,
            "num_rolls_remaining": self._num_rolls_remaining,
            "dices": self._dices,
        }

    def act(self, action):
        try:
            if action['select_action'] == 0:
                self._roll(action['dices_hold'])
            else:
                self._select(action['board_selection'])
        except KniffelError:
            # On a KniffelError do nothing.
            # TODO: Reward -1, so invalid moves are discouraged.
            pass


def play_kniffel():
    """Calling the module should let you play kniffel interactively as a human.

    Right now it doesn't.
    """
    env = Kniffel()
    observation = env.reset()
    done = False
    steps = 0
    while not done:
        steps += 1
        observation, _, done, info = env.step(env.action_space.sample())
    print(f"Board full in {steps} actions with {info['full_score']} Points!")


if __name__ == "__main__":
    play_kniffel()
