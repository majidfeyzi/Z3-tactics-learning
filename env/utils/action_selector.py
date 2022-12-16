
class ActionSelector:
    """
    Class to select tactics (actions) based on learning.
    """

    @staticmethod
    def select_tactics_based_on_learning(env, agent, actions, count):
        """
        Select tactics (actions) based on learning.
        :param env: env that agent need to act in that
        :param agent: agent that select actions (tactics) based on learning
        :param actions: actions that we can have
        :param count: number of tactics (actions) we want to select
        :return: selected tactics (actions)
        """
        state = env.reset()
        selected_actions = list()
        for _ in range(count):
            action_indices = agent.act(state, count)
            action_index = action_indices[0]
            for action_index in action_indices:
                action = actions[action_index]
                if action not in selected_actions:
                    selected_actions.append(action)
                    break
            if len(selected_actions) < count:
                next_state = env.step(action_index)
                state = next_state
        return selected_actions