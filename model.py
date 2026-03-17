from mesa import Agent, Model
from mesa.time import RandomActivation


class WindTurbine(Agent):
    """Agent representing a wind turbine that generates energy."""

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.output = 0

    def step(self):
        # Simulate wind energy production
        self.output = 10
        print(f"[WindTurbine {self.unique_id}] Generated: {self.output} units")


class Load(Agent):
    """Agent representing an electrical load consuming energy."""

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.demand = 5

    def step(self):
        print(f"[Load {self.unique_id}] Consumed: {self.demand} units")


class Battery(Agent):
    """Agent representing a simple battery storage system."""

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.charge = 50

    def step(self):
        # Simple charging logic
        self.charge += 2
        print(f"[Battery {self.unique_id}] Charge level: {self.charge}")


class MicrogridModel(Model):
    """Main model representing a simple microgrid system."""

    def __init__(self):
        self.schedule = RandomActivation(self)

        # Initialize agents
        self._create_agents()

    def _create_agents(self):
        """Create and add all agents to the scheduler."""
        wind = WindTurbine(0, self)
        load = Load(1, self)
        battery = Battery(2, self)

        self.schedule.add(wind)
        self.schedule.add(load)
        self.schedule.add(battery)

    def step(self):
        """Advance the model by one step."""
        print("\n--- Simulation Step ---")
        self.schedule.step()


if __name__ == "__main__":
    model = MicrogridModel()

    for step in range(5):
        print(f"\nStep {step}")
        model.step()