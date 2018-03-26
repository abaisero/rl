class Algo:
    def run(self, env, policy, nepisodes, nsteps):
        policy.reset()
        for e in range(nepisodes):
            self.episode(env, policy, nsteps)

    def episode(env, policy, nsteps):
        raise NotImplementedError
