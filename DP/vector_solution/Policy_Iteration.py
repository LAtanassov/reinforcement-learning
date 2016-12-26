import sys
if "../" not in sys.path:
  sys.path.append("../")
from lib.envs.gridworld import GridworldEnv
import numpy as np

env = GridworldEnv()


def policy_eval(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.

    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a (prob, next_state, reward, done) tuple.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: gamma discount factor.

    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with a random (all 0) value function
    V = np.zeros(env.nS)

    transistion_sa = np.array([[env.P[i][j][0][0] for j in env.P[i]] for i in env.P])
    nextstate_sa = np.array([[env.P[i][j][0][1] for j in env.P[i]] for i in env.P])
    reward_sa = np.array([[env.P[i][j][0][2] for j in env.P[i]] for i in env.P])
    while True:

        Vk = np.multiply(policy, reward_sa + discount_factor * V[nextstate_sa]).sum(axis=1)
        delta = np.max(np.subtract(V, Vk))
        V = Vk
        #print V
        if delta < theta:
            break;

    return np.array(V)

def policy_improvement(env, policy_eval_fn=policy_eval, discount_factor=1.0):
    """
    Policy Improvement Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.

    Args:
        env: The OpenAI envrionment.
        policy_eval_fn: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: Lambda discount factor.

    Returns:
        A tuple (policy, V).
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.

    """
    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA
    transistion_sa = np.array([[env.P[i][j][0][0] for j in env.P[i]] for i in env.P])
    nextstate_sa = np.array([[env.P[i][j][0][1] for j in env.P[i]] for i in env.P])
    reward_sa = np.array([[env.P[i][j][0][2] for j in env.P[i]] for i in env.P])

    while True:
        # Evaluate the current policy
        V = policy_eval_fn(policy, env, discount_factor)

        chosen_as = np.argmax(policy, axis=1)
        best_as =  np.argmax(reward_sa + discount_factor * V[nextstate_sa], axis=1)
        # If the policy is stable we've found an optimal policy. Return it
        if np.array_equal(chosen_as, best_as):
            return policy, V
        else:
            policy.fill(0)
            policy[np.arange(policy.shape[0]), best_as] = 1


    return policy, np.zeros(env.nS)

policy, v = policy_improvement(env)
print("Policy Probability Distribution:")
print(policy)
print("")

print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):")
print(np.reshape(np.argmax(policy, axis=1), env.shape))
print("")

print("Value Function:")
print(v)
print("")

print("Reshaped Grid Value Function:")
print(v.reshape(env.shape))
print("")

# Test the value function
expected_v = np.array([ 0, -1, -2, -3, -1, -2, -3, -2, -2, -3, -2, -1, -3, -2, -1,  0])
np.testing.assert_array_almost_equal(v, expected_v, decimal=2)