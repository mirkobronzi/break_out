import argparse
import logging
import os
import random

import progressbar

from break_out_ml import BreakOutML, _prepare_batch_for_fit, define_options, \
    get_models

logger = logging.getLogger(__name__)


class BreakOutMLReply(BreakOutML):

    def __init__(self, render=False,  n_actions=4, model=None, gamma=0.99,
                 batch_size=32, memory_size=10000, epsilon_annihilation=20000,
                 debug=False):
        super(__class__, self).__init__(
            render=render, n_actions=n_actions, model=model,
            gamma=gamma, batch_size=batch_size,
            epsilon_annihilation=epsilon_annihilation, debug=debug)
        self.memory_size = memory_size
        self.cycles = 0
        self._last_memory_fill_rewards = 0

    def _play_and_fill_memory(self, frames, epsilon=None, update_steps=True):
        memory = [None] * self.memory_size
        for i in progressbar.progressbar(range(self.memory_size)):
            move_result = self.choose_and_perform_actions(
                frames, epsilon=epsilon)
            self.maybe_render()
            frames = move_result.next_frames
            self._last_memory_fill_rewards += move_result.reward
            memory[i] = move_result
            if update_steps:
                self._tot_steps += 1
            if move_result.is_done:
                frames = self._get_first_frames()
                self._games_played += 1
        return memory, frames

    def _fit_on_memory(self, memory, steps=9999999, display_progress=True):
        until = min(self.memory_size - self.batch_size, steps)
        if display_progress:
            pb = progressbar.bar.ProgressBar()
        else:
            pb = progressbar.bar.NullBar()
        for i in pb(
                range(0, until, self.batch_size)):
            moves = memory[i:i+self.batch_size]
            batch = _prepare_batch_for_fit(moves, self.n_actions)
            self._fit_batch(batch)

    def train_with_memory_cycles(self, steps, play_while_training, model_path):
        current_frames = self._get_first_frames()
        while self._tot_steps < steps:
            # fill memory
            logger.info('filling memory..')
            memory, current_frames = self._play_and_fill_memory(current_frames)
            logger.info(self.status())
            # shuffle memory
            random.shuffle(memory)
            logger.info('training the models..')
            self._fit_on_memory(memory)
            self.cycles += 1
            if play_while_training:
                self.play_one_game(100)
            self.save(model_path)

    def _perform_one_move(self, frames, epsilon=None):
        move_result = self.choose_and_perform_actions(
            frames, epsilon=epsilon)
        self.maybe_render()
        frames = move_result.next_frames
        if move_result.is_done:
            frames = self._get_first_frames()
            self._games_played += 1
        return move_result, frames

    def train_with_memory_updates(self, steps, play_while_training, model_path):
        current_frames = self._get_first_frames()
        memory, current_frames = self._play_and_fill_memory(
            current_frames, epsilon=1.0, update_steps=False)
        random.shuffle(memory)
        while self._tot_steps < steps:
            # train for just one mini-batch
            sample = random.sample(memory, self.batch_size)
            self._fit_on_memory(sample,
                                steps=self.batch_size, display_progress=False)
            self._tot_steps += 1
            for _ in range(self.batch_size):
                move_result, current_frames = self._perform_one_move(
                    current_frames)
                store_to = random.randint(0, self.memory_size - 1)
                memory[store_to] = move_result
            if self.cycles % 100 == 0:
                logger.info(self.status())
            if self.cycles % 1000 == 0:
                logger.info('saving')
                self.save(model_path)
            self.cycles += 1
        self.save(model_path)

    def status(self):
        return "step {:5d} - cycles {:5d} - (eps {:.2f} - rand {:5d} / " \
               "model {:5d}) - games played {:4d} - " \
               "actions {} - same {:5d} / diff {:5d}".format(
            self._tot_steps,
            self.cycles,
            self.get_epsilon_for_iteration(),
            self._random_actions,
            self._model_actions,
            self._games_played,
            self.compute_action_percentage(),
            self.same_model, self.diff_model)


def main():
    parser = argparse.ArgumentParser(description='atari ML')
    define_options(parser)

    parser.add_argument('--memory-size', default=10000, type=int)
    parser.add_argument('--play-while-training',
                        help='will play a game after every memory cycle',
                        action='store_true')
    parser.add_argument('--memory-updates',
                        action='store_true')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    model = get_models(args)

    bomlr = BreakOutMLReply(
        model=model, debug=args.debug, n_actions=args.n_actions,
        render=args.render, epsilon_annihilation=args.epsilon_annihilation,
        batch_size=args.batch_size, memory_size=args.memory_size,
        gamma = args.gamma)
    if args.play:
        bomlr.play_one_game()
    elif args.memory_updates:
        logger.info('will train with memory update')
        bomlr.train_with_memory_updates(args.steps, args.play_while_training,
                                       args.model_path)
    else:
        logger.info('will train with memory cycles')
        bomlr.train_with_memory_cycles(args.steps, args.play_while_training,
                                       args.model_path)


if __name__ == "__main__":
    main()
