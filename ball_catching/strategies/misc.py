from ball_catching.strategies.base import Strategy


# ----------------------------------------------------------------------------------------

class ZeroStrategy(Strategy):
    def step(self, i, x, dicts):
        return [0., 0.]
