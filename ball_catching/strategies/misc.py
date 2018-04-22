from ball_catching.dynamics.world import Strategy

# ----------------------------------------------------------------------------------------

class ZeroStrategy(Strategy):
    def step(self, i, x, dicts):
        return [0., 0.]
