from vizdoom import *
import tensorflow as tf
import numpy as np
import random
import time
from skimage import transform
import dqn2
import mem
from collections import deque
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')          #ignore the skimags warnings
import argparse


# Model hyperparams
state_size = None
action_size = None
learning_rate = 0.0002
total_episodes = 500
max_steps = 100
batch_size = 64
explore_start = 1.0
explore_stop  = .01
decay_rate = 0.0001
gamma = 0.95
pretrain_length = batch_size
memory_size = 1000000
render = True
DQNetwork = None
stack_size = 4
saver = tf.train.Saver()
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=False, type=bool, default=False,
                help="Use the latest saved model")
args = vars(ap.parse_args())


def train(memory):
    global stacked_frames
    global saver

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    decay_step = 0
    game.init()
    for episode in range(total_episodes):
        print("Training on episode: {}".format(episode))
        step = 0
        episode_rewards = []
        game.new_episode()
        state = game.get_state().screen_buffer
        state, stacked_frames = stack_frames(stacked_frames, state, True)

        while step < max_steps:
            step += 1
            decay_step += 1
            action, explore_probability = predict_action(decay_step, possible_actions, state, sess)
            reward = game.make_action(action)
            done = game.is_episode_finished()
            episode_rewards.append(reward)

            if done:
                # the episode ends so no next state
                next_state = np.zeros((84, 84), dtype=np.int)
                next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

                # Set step = max_steps to end the episode
                step = max_steps

                # Get the total reward of the episode
                total_reward = np.sum(episode_rewards)

                print('Episode: {}'.format(episode),
                      'Total reward: {}'.format(total_reward),
                      'Training loss: {:.4f}'.format(loss),
                      'Explore P: {:.4f}'.format(explore_probability))

                memory.add((state, action, reward, next_state, done))

            else:
                # Get the next state
                next_state = game.get_state().screen_buffer

                # Stack the frame of the next_state
                next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

                # Add experience to memory
                memory.add((state, action, reward, next_state, done))

                # st+1 is now our current state
                state = next_state

            ### LEARNING PART
            # Obtain random mini-batch from memory
            batch = memory.sample(batch_size)

            states_mb = np.array([each[0] for each in batch])
            actions_mb = np.array([each[1] for each in batch])
            rewards_mb = np.array([each[2] for each in batch])
            next_states_mb = np.array([each[3] for each in batch])
            dones_mb = np.array([each[4] for each in batch])

            target_Qs_batch = []

            # Get Q values for next_state
            Qs_next_state = sess.run(DQNetwork.output, feed_dict={DQNetwork.inputs_: next_states_mb})

            # Set Q_target = r if the episode ends at s+1, otherwise set Q_target = r + gamma*maxQ(s', a')
            for i in range(0, len(batch)):
                terminal = dones_mb[i]

                # If we are in a terminal state, only equals reward
                if terminal:
                    target_Qs_batch.append(rewards_mb[i])

                else:
                    target = rewards_mb[i] + gamma * np.max(Qs_next_state[i])
                    target_Qs_batch.append(target)

            targets_mb = np.array([each for each in target_Qs_batch])

            loss, _ = sess.run([DQNetwork.loss, DQNetwork.optimizer],
                               feed_dict={DQNetwork.inputs_: states_mb,
                                          DQNetwork.target_Q: targets_mb,
                                          DQNetwork.actions_: actions_mb})

            # Write TF Summaries
            summary = sess.run(write_op, feed_dict={DQNetwork.inputs_: states_mb,
                                                    DQNetwork.target_Q: targets_mb,
                                                    DQNetwork.actions_: actions_mb})
            writer.add_summary(summary, episode)
            writer.flush()

            # Save model every 5 episodes
        if episode % 5 == 0:
            save_path = saver.save(sess, "./models/model.ckpt")
            print("Model Saved")


def infer():
    global env
    global saver

    sess = tf.Session()

    total_test_rewards = []
    game, possible_actions = create_environment()
    totalScore = 0
    saver.restore(sess, "./models/model.ckpt")
    game.init()

    for episode in range(1):
        total_rewards = 0

        state = env.reset()
        state, stacked_frames = stack_frames(stacked_frames, state, True)

        print("**************************************")
        print("EPISODSE ", episode)

        while True:
            # reshape the state
            state = state.reshape((1, *state_size))
            Qs = sess.run(DQNetwork.output, feed_dict={DQNetwork.inputs_: state})

            # Take the biggest Q value
            choice = np.argmax(Qs)
            action = possible_actions[choice]
            next_state, reward, done, _ = env.step(action)
            env.render()
            total_rewards += reward

            if done:
                print("Score", total_rewards)
                total_test_rewards.append(total_rewards)
                break

            next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
            state = next_state

    env.close()


def predict_action(decay_step, possible_actions, state, sess):
    global explore_start
    global explore_stop
    global decay_rate

    exp_exp_tradeoff = np.random.rand()
    explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)

    if explore_probability > exp_exp_tradeoff:
        action = random.choice(possible_actions)
    else:
        Qs = sess.run(DQNetwork.output, feed_dict={DQNetwork.inputs_: state.reshape((1, *state.shape))})

        choice = np.argmax(Qs)
        action = possible_actions[int(choice)]

    return action, explore_probability


def createnetwork():
    global DQNetwork
    global state_size
    global action_size
    global learning_rate

    tf.reset_default_graph()
    DQNetwork = dqn2.DQNetwork(state_size, action_size, learning_rate)


def preprocess_frame(frame):

    cropped_frame = frame[30:-10, 30: -30]
    normalized_frame = cropped_frame/255.0
    preprocessed_frame = transform.resize(normalized_frame, [84, 84])

    return preprocessed_frame


def stack_frames(frames2stack, state, is_new_episode):
    frame = preprocess_frame(state)

    if is_new_episode:
        # Clear the stacked frames
        frames2stack = deque([np.zeros((84, 84), dtype=np.int) for i in range(stack_size)], maxlen=4)

        # this is new episode so copy the first and only frame 4 times
        frames2stack.append(frame)
        frames2stack.append(frame)
        frames2stack.append(frame)
        frames2stack.append(frame)

        # stack the frames
        stacked_state = np.stack(frames2stack, axis=2)
    else:
        # Append frame to deque
        frames2stack.append(frame)
        stacked_state = np.stack(frames2stack, axis=2)

    return stacked_state, frames2stack


def create_environment():
    global game
    global state_size
    global action_size

    game = DoomGame()
    game.load_config("basic.cfg")
    game.set_doom_scenario_path("basic.wad")
    game.init()
    state_size = [84, 84, 4] # 4 Frames at 84x84 resolution

    # Possible actions
    left = [1, 0, 0]
    right = [0, 1, 0]
    shoot = [0, 0, 1]
    possible_actions = [left, right, shoot]
    action_size = 3
    return game, possible_actions


def test_environment():
    global game
    episodes = 10
    for i in range(episodes):
        game.new_episode()
        while not game.is_episode_finished():
            state = game.get_state()
            img = state.screen_buffer
            misc = state.game_variables
            action = random.choice(possible_actions)
            print("Taking action: {}".format(action))
            reward = game.make_action(action)
            print("Reward for above action: {}".format(reward))
            time.sleep(.2)
        print("Episode final reward score: {}".format(game.get_total_reward()))
        time.sleep(2)
    game.close()


if not args['model']:
    stacked_frames = deque([np.zeros((84, 84), dtype=np.int) for i in range(stack_size)], maxlen=4)
    game, possible_actions = create_environment()
    print("Game Environment created")
    createnetwork()
    print("network created")

    memory = mem.Memory(max_size=memory_size)
    game.new_episode()
    for i in range(pretrain_length):
        if i is 0:
            state = game.get_state().screen_buffer
            state, stacked_frames = stack_frames(stacked_frames, state, True)

            # Take a random action
            action = random.choice(possible_actions)

            # rewards
            reward = game.make_action(action)

            done = game.is_episode_finished()

            if done:
                next_state = np.zeros(state.shape)

                memory.add((state, action, reward, next_state, done))

                # Start a new episode
                game.new_episode()

                # Get fresh state
                state = game.get_state().screen_buffer

                # Stack frames
                state, stacked_frames = stack_frames(stacked_frames, state, True)
            else:
                next_state = game.get_state().screen_buffer
                next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

                memory.add((state, action, reward, next_state, done))

                state = next_state

    print("memory created")

    # Setup tensorboard
    writer = tf.summary.FileWriter("/tensorboard/dqn/1")
    tf.summary.scalar("Loss", DQNetwork.loss)
    write_op = tf.summary.merge_all()

    print("tensorboard setup")
    print("here we go...")

    train(memory)

    command = input("Training complete, press a key to test our model\n")
else:
    create_environment()
    infer()


