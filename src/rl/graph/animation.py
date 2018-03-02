import matplotlib.pyplot as plt


class Animation:
    # TODO how to receive data?!

    def __init__(self, animakers):
        self.animakers = animakers

        self.p = mp.Process(target=self.target)
        self.p.daemon = True

    def start(self):
        self.p.start()

    def target(self):
        animations = [animaker() for animaker in self.animakers]
        plt.show()


def animate(animakers):
    # TODO each animation should...?

    t = th.Thread(target=target)
    t.daemon = True
    t.start()

    p = mp.Process(target=target)
    p.daemon = True
    p.start()

    return p
