from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid


class Need:
    def __init__(self, name, initial=0.0, growth=0.05):
        self.name    = name
        self.urgency = initial
        self.growth  = growth   # how fast urgency builds up per step

    def tick(self):
        self.urgency = min(1.0, self.urgency + self.growth)

    def satisfy(self, amount=1.0):
        self.urgency = max(0.0, self.urgency - amount)

    def __repr__(self):
        return f"{self.name}({self.urgency:.2f})"


class NeedsAgent(Agent):
    # Base class for agents driven by needs.
    # Subclasses call add_need() in __init__ and then use most_urgent()
    # in their step() to decide what to do.

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self._needs = {}

    def add_need(self, name, initial=0.0, growth=0.05):
        self._needs[name] = Need(name, initial, growth)

    def need(self, name):
        return self._needs[name]

    def most_urgent(self):
        if not self._needs:
            return None
        return max(self._needs.values(), key=lambda n: n.urgency)

    def tick_all(self):
        for n in self._needs.values():
            n.tick()


# --- Wolf --------

class Wolf(NeedsAgent):
    # Two needs: hunger and reproduction.
    # Hunger grows faster so it takes priority most of the time,
    # but reproduction urgency builds up slowly and eventually
    # the wolf will look for a chance to breed.
    #
    # Mesa issue: to find nearby sheep I have to go through
    # self.model.grid directly. There's no cleaner way to do
    # environment perception in Mesa right now.

    def __init__(self, unique_id, model, energy=10):
        super().__init__(unique_id, model)
        self.energy = energy
        self.add_need("hunger",       initial=0.2, growth=0.08)
        self.add_need("reproduction", initial=0.0, growth=0.03)

    def step(self):
        self._move()
        self.energy -= 1
        self.tick_all()

        if self.energy <= 0:
            self.model.grid.remove_agent(self)
            self.model.schedule.remove(self)
            return

        top = self.most_urgent()
        if top.name == "hunger":
            self._eat()
        elif top.name == "reproduction":
            self._reproduce()

    def _move(self):
        neighbors = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=False)
        self.model.grid.move_agent(self, self.random.choice(neighbors))

    def _eat(self):
        here = self.model.grid.get_cell_list_contents([self.pos])
        prey = [a for a in here if isinstance(a, Sheep)]
        if prey:
            target = self.random.choice(prey)
            self.model.grid.remove_agent(target)
            self.model.schedule.remove(target)
            self.energy += 20
            self.need("hunger").satisfy(0.8)

    def _reproduce(self):
        if self.energy > 15:
            self.energy //= 2
            pup = Wolf(self.model.next_id(), self.model, energy=self.energy)
            self.model.grid.place_agent(pup, self.pos)
            self.model.schedule.add(pup)
            self.need("reproduction").satisfy(1.0)


# --- Sheep -----

class Sheep(NeedsAgent):
    # Two needs: hunger and fear.
    # Fear is interesting — it spikes when wolves are nearby and
    # drives the sheep to flee before it even thinks about eating.
    # When no wolves are close, fear decays on its own.
    #
    # This produces something that looks like flocking without any
    # explicit flocking rules, which is kind of cool.
    #
    # Mesa issue: scanning for nearby wolves is O(n) via get_neighbors.
    # A spatial query like "nearest agent of type X" would help a lot here.

    def __init__(self, unique_id, model, energy=10):
        super().__init__(unique_id, model)
        self.energy = energy
        self.add_need("hunger", initial=0.1, growth=0.06)
        self.add_need("fear",   initial=0.0, growth=0.0)   # fear is event-driven

    def step(self):
        self.energy -= 1
        self.tick_all()
        self._update_fear()

        if self.energy <= 0:
            self.model.grid.remove_agent(self)
            self.model.schedule.remove(self)
            return

        top = self.most_urgent()
        if top.name == "fear":
            self._flee()
        elif top.name == "hunger":
            self._graze()
        else:
            self._wander()

    def _update_fear(self):
        nearby = self.model.grid.get_neighbors(
            self.pos, moore=True, radius=2, include_center=False)
        wolf_count = sum(1 for a in nearby if isinstance(a, Wolf))
        if wolf_count > 0:
            self.need("fear").urgency = min(
                1.0, self.need("fear").urgency + 0.2 * wolf_count)
        else:
            self.need("fear").satisfy(0.1)

    def _flee(self):
        options = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=False)
        safest = min(options, key=lambda p: sum(
            1 for a in self.model.grid.get_cell_list_contents([p])
            if isinstance(a, Wolf)))
        self.model.grid.move_agent(self, safest)
        self.need("fear").satisfy(0.3)

    def _graze(self):
        here = self.model.grid.get_cell_list_contents([self.pos])
        grass = [a for a in here if isinstance(a, GrassPatch) and a.grown]
        if grass:
            grass[0].grown = False
            self.energy += 5
            self.need("hunger").satisfy(0.7)

    def _wander(self):
        options = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=False)
        self.model.grid.move_agent(self, self.random.choice(options))


# --- GrassPatch -----

class GrassPatch(Agent):
    def __init__(self, unique_id, model, grown=True, regrow_after=5):
        super().__init__(unique_id, model)
        self.grown       = grown
        self.regrow_after = regrow_after
        self._timer      = 0

    def step(self):
        if not self.grown:
            self._timer += 1
            if self._timer >= self.regrow_after:
                self.grown  = True
                self._timer = 0


# --- Model ------

class WolfSheepNeedsModel(Model):
    # Main difference from the standard Mesa Wolf-Sheep:
    # behavior emerges from competing needs instead of probability parameters.
    # A wolf doesn't reproduce with p=0.05 — it reproduces when its
    # reproduction urgency beats its hunger urgency AND it has enough energy.

    def __init__(self, width=20, height=20, n_wolves=10, n_sheep=50, n_grass=100):
        super().__init__()
        self.grid     = MultiGrid(width, height, torus=True)
        self.schedule = RandomActivation(self)
        self.running  = True

        for _ in range(n_grass):
            g = GrassPatch(self.next_id(), self,
                           grown=self.random.choice([True, False]))
            x, y = self.random.randrange(width), self.random.randrange(height)
            self.grid.place_agent(g, (x, y))
            self.schedule.add(g)

        for _ in range(n_sheep):
            s = Sheep(self.next_id(), self, energy=self.random.randint(5, 15))
            x, y = self.random.randrange(width), self.random.randrange(height)
            self.grid.place_agent(s, (x, y))
            self.schedule.add(s)

        for _ in range(n_wolves):
            w = Wolf(self.next_id(), self, energy=self.random.randint(8, 20))
            x, y = self.random.randrange(width), self.random.randrange(height)
            self.grid.place_agent(w, (x, y))
            self.schedule.add(w)

    def step(self):
        self.schedule.step()

    def wolves(self):
        return sum(1 for a in self.schedule.agents if isinstance(a, Wolf))

    def sheep(self):
        return sum(1 for a in self.schedule.agents if isinstance(a, Sheep))


if __name__ == "__main__":
    m = WolfSheepNeedsModel(n_wolves=8, n_sheep=40)
    print(f"start: {m.wolves()} wolves, {m.sheep()} sheep")
    for i in range(1, 21):
        m.step()
        if i % 5 == 0:
            print(f"step {i}: {m.wolves()} wolves, {m.sheep()} sheep")
