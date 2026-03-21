from mesa import Agent, Model
from mesa.time import RandomActivation
import numpy as np
from ml_model import load_model, predict


class WindAgent(Agent):
    """Simulates a wind turbine whose output follows a smoothed sine wave."""

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self._raw_history = []
        self.production = 0.0

    def step(self):
        raw = np.sin(self.model.step_count / 3.0) * 5.0 + 10.0
        raw += np.random.uniform(-0.5, 0.5)
        raw = max(raw, 0.0)
        self._raw_history.append(raw)
        if len(self._raw_history) >= 3:
            self.production = float(np.mean(self._raw_history[-3:]))
        else:
            self.production = float(raw)


class LoadAgent(Agent):
    """Simulates a consumer whose demand cycles between two levels."""

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.consumption = 0.0

    def step(self):
        self.consumption = 5.0 if self.model.step_count % 10 < 5 else 10.0


class BatteryAgent(Agent):
    """Models a Battery Energy Storage System (BESS)."""

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.soc = 50.0
        self.capacity = 100.0
        self.efficiency = 0.95

    def charge(self, power):
        delta = min(power * self.efficiency, self.capacity - self.soc)
        self.soc = min(self.soc + delta, 100.0)

    def discharge(self, power):
        delta = min(power / self.efficiency, self.soc)
        self.soc = max(self.soc - delta, 0.0)


def rule_based_action(net, soc, price):
    if net > 0 and soc < 90:
        return "charge"
    elif net < 0 and soc > 20:
        return "discharge"
    else:
        return "idle"


class MicrogridModel(Model):

    def __init__(self):
        super().__init__()
        self.schedule = RandomActivation(self)
        self.step_count = 0

        self.wind    = WindAgent(0, self)
        self.load1   = LoadAgent(1, self)
        self.load2   = LoadAgent(2, self)
        self.battery = BatteryAgent(3, self)

        self.schedule.add(self.wind)
        self.schedule.add(self.load1)
        self.schedule.add(self.load2)

        self.grid_import     = 0.0
        self.price           = 1.0
        self.total_cost      = 0.0
        self.rule_total_cost = 0.0
        self.history         = []

        self.ml_model = load_model()

    def step(self):
        self.step_count += 1
        self.schedule.step()

        production  = float(self.wind.production)
        consumption = float(self.load1.consumption + self.load2.consumption)
        net         = production - consumption

        self.price = 1.0 if self.step_count % 10 < 5 else 3.0

        ml_action = predict(
            self.ml_model, production, consumption,
            self.battery.soc, self.price
        )

        self.grid_import = 0.0

        if ml_action == "charge" and net > 0:
            self.battery.charge(net)
        elif ml_action == "discharge" and self.battery.soc > 10:
            self.battery.discharge(abs(net))
        else:
            if net < 0:
                self.grid_import = abs(net)

        ml_cost          = self.grid_import * self.price
        self.total_cost += ml_cost

        rb_action      = rule_based_action(net, self.battery.soc, self.price)
        rb_grid_import = 0.0

        if rb_action == "charge" and net > 0:
            pass
        elif rb_action == "discharge" and self.battery.soc > 10:
            pass
        else:
            if net < 0:
                rb_grid_import = abs(net)

        rb_cost               = rb_grid_import * self.price
        self.rule_total_cost += rb_cost

        self.history.append({
            "production":      production,
            "consumption":     consumption,
            "battery":         self.battery.soc,
            "grid_import":     self.grid_import,
            "price":           self.price,
            "cost":            ml_cost,
            "total_cost":      self.total_cost,
            "rule_cost":       rb_cost,
            "rule_total_cost": self.rule_total_cost,
            "action":          ml_action,
            "rule_action":     rb_action,
        })
