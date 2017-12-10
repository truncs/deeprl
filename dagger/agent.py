#!/usr/bin/env python

import argparse
import random

import gym
import numpy as np
import tensorflow as tf

import tf_util
from tf_util import save_state, load_state
import load_policy


class Agent(object):
    """
    Base agent class that has some framework for all other agents like
    having a train function that takes sample observations and actions
    and trains whatever internal policy/model the agent would have. The
    other important method of agent is given a set of observations, 
    predicts the action that should be take.

    """
    
    def __init__(self, model_file):
        self.model_file = model_file
        self.state_initialized = False
        self._setup()

    def _setup(self):
        pass

    def train(self, observations, actions):
        raise NotImplementedError
    
    def get_action(self, session, observation, actions_dim):
        raise NotImplementedError


class SimpleNetworkAgent(Agent):
    
    def __init__(self, model_file):
        super(SimpleNetworkAgent, self).__init__(model_file)

    def _make_graph(self, observations, actions_dim):
        observations_dim = np.shape(observations)[1]

        # Setup input/output placeholders
        self.observation = tf.placeholder(tf.float32, [None, observations_dim])

        # Setup the main network
        w1 = tf.get_variable("w1", [observations_dim, 128], dtype=tf.float32,
                         initializer=tf.contrib.layers.xavier_initializer()
        )
        b1 = tf.get_variable("b1", [128], dtype=tf.float32, initializer=tf.zeros_initializer)
        x1 = tf.nn.relu(tf.matmul(self.observation, w1) + b1)

        w2 = tf.get_variable("w2", [128, 64], dtype=tf.float32,
                         initializer=tf.contrib.layers.xavier_initializer()
        )
        b2 = tf.get_variable("b2", [64], dtype=tf.float32, initializer=tf.zeros_initializer)
        x2 = tf.nn.relu(tf.matmul(x1, w2) + b2)

        w3 = tf.get_variable("w3", [64, actions_dim], dtype=tf.float32,
                         initializer=tf.contrib.layers.xavier_initializer()
        )
        b3 = tf.get_variable("b3", [actions_dim], dtype=tf.float32, 
                             initializer=tf.zeros_initializer)
        self.action = tf.matmul(x2, w3) + b3
        
        return self.action

    def get_action(self, session, observations, action_dim):

        if not self.state_initialized:

            self._make_graph(observations, action_dim)
            load_state(self.model_file)
            self.state_initialized = True
        
        return session.run([self.action], feed_dict={self.observation: observations})
        
    def run_agent(self, envname, rollouts, max_steps=None, render=False):

        env = gym.make(envname)
        max_steps = max_steps or env.spec.timestep_limit
        
        returns = []
        observations = []
        actions = []
        
        # TODO this might not work globally
        action_dim = env.action_space.shape[0]

        with tf.Session() as sess:

            for i in range(rollouts):
                print('iter', i)
                obs = env.reset()
                done = False
                totalr = 0.
                steps = 0
                while not done:
                    reshapped_obs = np.reshape(obs, [1, -1])
                    action = self.get_action(sess, reshapped_obs, action_dim)
                    observations.append(obs)
                    actions.append(action)
                    obs, r, done, _ = env.step(action)
                    totalr += r
                    steps += 1
                    if render:
                        env.render()
                    if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                    if steps >= max_steps:
                        break
                returns.append(totalr)

            print('returns', returns)
            print('mean return', np.mean(returns))
            print('std of return', np.std(returns))


    def train(self, observations, actions, steps, batch_size=32):
        
        actions_dim = np.shape(actions)[2]

        action = self._make_graph(observations, actions_dim)

        y = tf.placeholder(tf.float32, [None, actions_dim])
        loss = tf.losses.mean_squared_error(tf.reshape(y, [-1, actions_dim]), self.action)
        train_step = tf.train.AdamOptimizer().minimize(loss)

        # Shuffle the observations
        p = np.random.permutation(len(observations))
        shuf_obs, shuf_actions = observations[p], actions[p]

        global_step = tf.Variable(0, name='global_step', trainable=False)

        # Create a tf session
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for step in range(steps):
                
                # Generate a random number between 0 and len(obs) - batch_size
                idx = random.randint(0, len(observations) - batch_size)

                x_batch = shuf_obs[idx: idx + batch_size]
                y_batch = np.reshape(shuf_actions[idx: idx + batch_size], [-1, actions_dim])

                loss_value, _ = sess.run(
                    [loss, train_step],
                    feed_dict={self.observation: x_batch, y: y_batch}
                )
                
                if step % 100 == 0:
                    print 'Loss value of %.5f at step %d' % (loss_value, step)
                
                global_step += 1
                
            # TODO maybe save the model file every once in a while
            saver = tf.train.Saver()
            saver.save(sess, self.model_file)


    def collect_dagger_data(self, expert_policy_file, expert_log_file, envname, rollouts, max_steps, dagger_log_file):

        env = gym.make(envname)
        max_steps = max_steps or env.spec.timestep_limit
        
        returns = []
        observations = []
        actions = []
        
        print('loading and building expert policy')
        policy_fn = load_policy.load_policy(expert_policy_file)
        print('loaded and built')

        # TODO this might not work globally
        action_dim = env.action_space.shape[0]

        with tf.Session() as sess:
            tf_util.initialize()
            
            for i in range(rollouts):
                print('iter', i)
                obs = env.reset()
                done = False
                totalr = 0.
                steps = 0
                while not done:
                    reshapped_obs = np.reshape(obs, [1, -1])
                    action = self.get_action(sess, reshapped_obs, action_dim)

                    # Get expert action
                    expert_action = policy_fn(obs[None,:])

                    observations.append(obs)
                    actions.append(expert_action)
                    obs, r, done, _ = env.step(action)
                    totalr += r
                    steps += 1

                    if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                    if steps >= max_steps:
                        break
                returns.append(totalr)


            expert_samples = np.load(expert_log_file)
            expert_observations, expert_actions = expert_samples['observations'], expert_samples['actions']
            observations = np.concatenate((observations, expert_observations), axis=0)
            actions = np.concatenate((actions, expert_actions), axis=0)

            expert_data = {'observations': np.array(observations),
                           'actions': np.array(actions)}

            # Save dagger data
            np.savez(
                dagger_log_file,
                observations=np.array(observations),
                actions=np.array(actions)
            )
  
            print('returns', returns)
            print('mean return', np.mean(returns))
            print('std of return', np.std(returns))
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--expert_log_file', type=str, default=None)
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--num_rollouts', type=int, default=20)
    parser.add_argument('--envname', type=str)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--model_file', type=str)
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--max_timesteps', type=int, default=1000)
    parser.add_argument('--dagger_log_file', type=str, default=None)
    parser.add_argument('--expert_policy_file', type=str, default=None)
    parser.add_argument('--collect_dagger_data', action='store_true')

    args = parser.parse_args()
    
    if args.train:
        samples = np.load(args.expert_log_file)
        observations, actions = samples['observations'], samples['actions']
        model = SimpleNetworkAgent(args.model_file)
        model.train(observations, actions, args.steps, args.batch_size)

    elif args.collect_dagger_data:
        model = SimpleNetworkAgent(args.model_file)
        model.collect_dagger_data(
            args.expert_policy_file, args.expert_log_file, args.envname, args.num_rollouts, 
            args.max_timesteps, args.dagger_log_file
        )
    else:
        model = SimpleNetworkAgent(args.model_file)
        model.run_agent(args.envname, args.num_rollouts, args.max_timesteps, args.render)


if __name__ == '__main__':
    main()
