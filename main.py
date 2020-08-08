import gym
from DDQN import DQNModel,DDQNAgent

def test_model():
    env = gym.make('CartPole-v0')
    print('num_actions: ', env.action_space.n)
    model = DQNModel(env.action_space.n, 'DQN')

    obs = env.reset()
    print('obs_shape: ', obs.shape)

    # tensorflow 2.0: no feed_dict or tf.Session() needed at all
    best_action, q_values = model.action_value(obs[None])
    print('res of test model: ', best_action, q_values)  # 0 [ 0.00896799 -0.02111824]

if __name__ == '__main__':
    test_model()

    env = gym.make("CartPole-v0")
    num_actions = env.action_space.n
    model = DQNModel(num_actions, 'DQN1')
    target_model = DQNModel(num_actions, 'DQN2')
    agent = DDQNAgent(model, target_model,  env)
    # test before
    rewards_sum = agent.evaluation(env)
    print("Before Training: %d out of 200" % rewards_sum) # 10 out of 200

    agent.train()
    # test after
    rewards_sum = agent.evaluation(env)
    print("After Training: %d out of 200" % rewards_sum) # 200 out of 200