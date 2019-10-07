
class Agent:
    def select_action(self, state):
        pass

    def optimize_model(self, state):
        pass

    def post_action_update(self, state, action, next_state, reward):
        pass

    def post_episode_update(self, **kwargs):
        pass
