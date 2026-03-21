"""
wolf_sheep_needs.py — Needs-based Wolf-Sheep model for Mesa

This is a prototype implementation exploring needs-based behavioral
architecture in Mesa, part of the GSoC 2026 Behavioral Framework evaluation.

The standard Wolf-Sheep model uses hardcoded probabilities to control
agent behavior (e.g. wolf_reproduce = 0.05). This version replaces
those probabilities with explicit needs — hunger, energy, fear — that
compete for priority and drive behavior dynamically.

The key question this prototype investigates: how cleanly can a
needs-based behavioral architecture be expressed in Mesa, and what
primitives would Mesa need to provide to make this easier?

Current status: prototype / work in progress
Observations so far are documented in BEHAVIORAL_NOTES.md
"""

from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
import numpy as np


# =============================================================================
# Needs framework
#
# Mesa has no built-in concept of "needs" or "drives". This small utility
# is something I had to write from scratch. A NeedsAgent mixin that Mesa
# provided natively would make this pattern much easier to express.
# =============================================================================

class Need:
    """
    A single need with a current urgency level (0.0 to 1.0).

    Urgency increases over time and decreases when the need is satisfied.
    The agent's behavior is driven by whichever need is most urgent.
    """

    def __init__(self, name, initial=0.0, decay_rate=0.05):
        self.name       = name
        self.urgency    = initial
        self.decay_rate = decay_rate   # how fast urgency grows per step

    def tick(self):
        """Urgency increases each step the need goes unmet."""
        self.urgency = min(1.0, self.urgency + self.decay_rate)

    def satisfy(self, amount=1.0):
        """Urgency drops when the need is met."""
        self.urgency = max(0.0, self.urgency - amount)

    def __repr__(self):
        return f"Need({self.name}, urgency={self.urgency:.2f})"


class NeedsAgent(Agent):
    """
    Base class for agents with a needs-based behavioral architecture.

    Subclasses register needs and define how to satisfy each one.
    Each step, the agent identifies its most urgent need and acts on it.

    This is the utility class I wish Mesa provided out of the box.
    Without it, every needs-based model has to re-implement this logic.
    """

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.needs = {}   # name -> Need

    def add_need(self, name, initial=0.0, decay_rate=0.05):
        self.needs[name] = Need(name, initial, decay_rate)

    def most_urgent_need(self):
        if not self.needs:
            return None
        return max(self.needs.values(), key=lambda n: n.urgency)

    def tick_needs(self):
        for need in self.needs.values():
            need.tick()


# =============================================================================
# Wolf agent — needs: hunger, reproduction
# =============================================================================

class Wolf(NeedsAgent):
    """
    Wolf with explicit needs replacing hardcoded reproduction probability.

    Behavioral logic:
      - If hunger is most urgent → try to eat a nearby sheep
      - If reproduction need is most urgent → try to reproduce
      - Otherwise → move randomly

    In the standard Wolf-Sheep model, reproduction is controlled by a
    fixed probability (wolf_reproduce = 0.05). Here it emerges naturally
    from the reproduction need's urgency, which builds up over time and
    resets after a successful reproduction.

    Mesa observation: the agent needs to query the grid for nearby sheep,
    which requires direct access to self.model.grid. There is no built-in
    mechanism for agents to perceive their environment without going through
    the model object. A proper behavioral framework might provide an
    "observation range" primitive.
    """

    def __init__(self, unique_id, model, energy=10):
        super().__init__(unique_id, model)
        self.energy = energy

        # Register needs with different urgency growth rates
        self.add_need("hunger",       initial=0.2, decay_rate=0.08)
        self.add_need("reproduction", initial=0.0, decay_rate=0.03)

    def step(self):
        self.move()
        self.energy -= 1
        self.tick_needs()

        if self.energy <= 0:
            self.model.grid.remove_agent(self)
            self.model.schedule.remove(self)
            return

        # Act on the most urgent need
        urgent = self.most_urgent_need()

        if urgent and urgent.name == "hunger":
            self.eat()
        elif urgent and urgent.name == "reproduction":
            self.reproduce()

    def move(self):
        possible = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=False)
        new_pos = self.random.choice(possible)
        self.model.grid.move_agent(self, new_pos)

    def eat(self):
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        sheep_here = [a for a in cellmates if isinstance(a, Sheep)]
        if sheep_here:
            sheep = self.random.choice(sheep_here)
            self.model.grid.remove_agent(sheep)
            self.model.schedule.remove(sheep)
            self.energy += 20
            self.needs["hunger"].satisfy(0.8)

    def reproduce(self):
        if self.energy > 15:
            self.energy = self.energy // 2
            cub = Wolf(self.model.next_id(), self.model, energy=self.energy)
            self.model.grid.place_agent(cub, self.pos)
            self.model.schedule.add(cub)
            self.needs["reproduction"].satisfy(1.0)


# =============================================================================
# Sheep agent — needs: hunger (grass), fear (proximity to wolves)
# =============================================================================

class Sheep(NeedsAgent):
    """
    Sheep with explicit hunger and fear needs.

    Behavioral logic:
      - If fear is most urgent → flee (move away from nearest wolf)
      - If hunger is most urgent → graze (eat grass if available)
      - Otherwise → move randomly

    The fear need is interesting: it grows faster when wolves are nearby
    and decays naturally when no wolves are in range. This creates emergent
    flocking-like behavior without any explicit flocking rules.

    Mesa observation: computing "nearest wolf" requires scanning all
    neighbors via self.model.grid.get_neighbors(). This works but is
    O(n) in the number of neighbors. A spatial query primitive ("find
    nearest agent of type X within radius R") would be useful here.
    """

    def __init__(self, unique_id, model, energy=10):
        super().__init__(unique_id, model)
        self.energy = energy

        self.add_need("hunger", initial=0.1, decay_rate=0.06)
        self.add_need("fear",   initial=0.0, decay_rate=0.10)

    def step(self):
        self.energy -= 1
        self.tick_needs()
        self.update_fear()

        if self.energy <= 0:
            self.model.grid.remove_agent(self)
            self.model.schedule.remove(self)
            return

        urgent = self.most_urgent_need()

        if urgent and urgent.name == "fear":
            self.flee()
        elif urgent and urgent.name == "hunger":
            self.graze()
        else:
            self.move_random()

    def update_fear(self):
        """Fear urgency increases when wolves are nearby."""
        neighbors = self.model.grid.get_neighbors(
            self.pos, moore=True, radius=2, include_center=False)
        wolves_nearby = sum(1 for a in neighbors if isinstance(a, Wolf))
        if wolves_nearby > 0:
            # Fear spikes proportionally to number of wolves nearby
            self.needs["fear"].urgency = min(
                1.0, self.needs["fear"].urgency + 0.2 * wolves_nearby)
        else:
            # Fear decays naturally when no threat is present
            self.needs["fear"].satisfy(0.1)

    def flee(self):
        """Move to the neighbor cell with the fewest wolves."""
        possible = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=False)
        # Score each cell by number of wolves in it — pick the safest
        def wolf_count(pos):
            return sum(1 for a in self.model.grid.get_cell_list_contents([pos])
                       if isinstance(a, Wolf))
        safest = min(possible, key=wolf_count)
        self.model.grid.move_agent(self, safest)
        self.needs["fear"].satisfy(0.3)

    def graze(self):
        """Eat grass at current position if available."""
        grass_here = [a for a in self.model.grid.get_cell_list_contents([self.pos])
                      if isinstance(a, GrassPatch) and a.fully_grown]
        if grass_here:
            grass_here[0].fully_grown = False
            self.energy += 5
            self.needs["hunger"].satisfy(0.7)

    def move_random(self):
        possible = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=False)
        self.model.grid.move_agent(self, self.random.choice(possible))


# =============================================================================
# GrassPatch — environment resource
# =============================================================================

class GrassPatch(Agent):
    """Simple grass resource that regrows after being eaten."""

    def __init__(self, unique_id, model, fully_grown=True, regrowth_time=5):
        super().__init__(unique_id, model)
        self.fully_grown   = fully_grown
        self.regrowth_time = regrowth_time
        self._countdown    = 0

    def step(self):
        if not self.fully_grown:
            self._countdown += 1
            if self._countdown >= self.regrowth_time:
                self.fully_grown  = True
                self._countdown   = 0


# =============================================================================
# WolfSheepNeedsModel
# =============================================================================

class WolfSheepNeedsModel(Model):
    """
    Wolf-Sheep simulation using needs-based behavioral agents.

    Compared to the standard Mesa Wolf-Sheep model, behavior here emerges
    from competing needs rather than hardcoded probabilities. A wolf does
    not reproduce with probability 0.05 — it reproduces when its
    reproduction need becomes urgent enough relative to its hunger need.

    This makes the model more sensitive to initial conditions and produces
    richer emergent dynamics, but it also requires more scaffolding code
    because Mesa does not provide needs-based primitives natively.
    """

    def __init__(self, width=20, height=20,
                 n_wolves=10, n_sheep=50, n_grass=100):
        super().__init__()
        self.grid     = MultiGrid(width, height, torus=True)
        self.schedule = RandomActivation(self)
        self.running  = True

        # Place grass
        for i in range(n_grass):
            x = self.random.randrange(width)
            y = self.random.randrange(height)
            grass = GrassPatch(self.next_id(), self,
                               fully_grown=self.random.choice([True, False]))
            self.grid.place_agent(grass, (x, y))
            self.schedule.add(grass)

        # Place sheep
        for _ in range(n_sheep):
            x, y = self.random.randrange(width), self.random.randrange(height)
            sheep = Sheep(self.next_id(), self,
                          energy=self.random.randint(5, 15))
            self.grid.place_agent(sheep, (x, y))
            self.schedule.add(sheep)

        # Place wolves
        for _ in range(n_wolves):
            x, y = self.random.randrange(width), self.random.randrange(height)
            wolf = Wolf(self.next_id(), self,
                        energy=self.random.randint(8, 20))
            self.grid.place_agent(wolf, (x, y))
            self.schedule.add(wolf)

    def step(self):
        self.schedule.step()

    def count_wolves(self):
        return sum(1 for a in self.schedule.agents if isinstance(a, Wolf))

    def count_sheep(self):
        return sum(1 for a in self.schedule.agents if isinstance(a, Sheep))


# =============================================================================
# Quick demo
# =============================================================================

if __name__ == "__main__":
    model = WolfSheepNeedsModel(n_wolves=8, n_sheep=40)
    print(f"Step 0: {model.count_wolves()} wolves, {model.count_sheep()} sheep")

    for i in range(1, 21):
        model.step()
        if i % 5 == 0:
            print(f"Step {i}: {model.count_wolves()} wolves, "
                  f"{model.count_sheep()} sheep")