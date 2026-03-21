from mesa import Agent, Model
from mesa.time import RandomActivation
import numpy as np
from ml_model import load_model, predict


# =============================================================================
# BDI structure for the BatteryAgent
#
# In a Belief-Desire-Intention architecture:
#   Beliefs  = what the agent currently knows about the world
#   Desire   = the agent's overarching goal
#   Intention = the action the agent commits to this step
#
# Mesa does not provide these primitives natively, so we define them here.
# This is one of the concrete gaps documented in BEHAVIORAL_NOTES.md.
# =============================================================================

class BatteryBeliefs:
    """
    Encapsulates everything the battery agent "knows" at a given step.

    In a proper BDI framework this would be a shared belief store that
    agents can subscribe to. In Mesa we have to pass values explicitly,
    which creates tight coupling between the battery and the model.
    This class at least makes the belief structure visible and explicit.
    """

    def __init__(self, soc, wind_production, load_consumption, price, net):
        self.soc              = soc               # current state of charge, %
        self.wind_production  = wind_production   # renewable output, kW
        self.load_consumption = load_consumption  # total demand, kW
        self.price            = price             # electricity tariff, $/kWh
        self.net              = net               # surplus (>0) or deficit (<0), kW

    def has_surplus(self):
        return self.net > 0

    def is_peak_price(self):
        return self.price >= 3.0

    def is_healthy(self):
        """SOC is in a safe operating range."""
        return 20.0 < self.soc < 90.0


def get_intention(beliefs, ml_model):
    """
    Translate beliefs into an intention using the RandomForest controller.

    This function represents the Belief-Desire-Intention reasoning cycle.
    The desire (minimize cost) is implicit in how the ML model was trained.
    The intention is the action the battery commits to for this step.

    The ML model is the primary decision-maker. The conditions below
    (net > 0 for charge, soc > 10 for discharge) are physical constraints,
    not behavioral rules — they prevent the battery from attempting
    physically impossible operations.
    """
    action = predict(ml_model, beliefs.wind_production,
                     beliefs.load_consumption, beliefs.soc, beliefs.price)

    # Physical feasibility checks — not behavioral logic
    if action == "charge" and not beliefs.has_surplus():
        action = "idle"   # cannot charge without surplus power
    if action == "discharge" and beliefs.soc <= 10:
        action = "idle"   # cannot discharge a nearly empty battery

    return action


# =============================================================================
# Mesa Agents
# =============================================================================

class WindAgent(Agent):
    """
    Simulates a wind turbine.

    Production follows a sine wave with mild random noise, smoothed over
    a 3-step moving average to avoid unrealistic step-to-step spikes.
    Range is roughly 5–15 kW depending on cycle phase.
    """

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self._raw_history = []
        self.production   = 0.0

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
    """
    Simulates a consumer with cyclic demand.

    Alternates between 5 kW (base load) and 10 kW (peak load) every
    5 steps. Two LoadAgents run in the model, so total demand varies
    between 10 and 20 kW.
    """

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.consumption = 0.0

    def step(self):
        self.consumption = 5.0 if self.model.step_count % 10 < 5 else 10.0


class BatteryAgent(Agent):
    """
    Models a Battery Energy Storage System (BESS).

    The battery does not make its own decisions — it receives instructions
    from the model's dispatch controller (see MicrogridModel.step).
    This separation reflects real BESS architecture where a Battery
    Management System issues charge/discharge commands.

    Round-trip efficiency of 0.95 reflects realistic lithium-ion losses.
    """

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.soc        = 50.0    # initial state of charge, %
        self.capacity   = 100.0   # nominal capacity, kWh
        self.efficiency = 0.95    # one-way efficiency

    def charge(self, power):
        delta    = min(power * self.efficiency, self.capacity - self.soc)
        self.soc = min(self.soc + delta, 100.0)

    def discharge(self, power):
        delta    = min(power / self.efficiency, self.soc)
        self.soc = max(self.soc - delta, 0.0)


# =============================================================================
# Rule-based baseline controller (runs as a shadow for cost comparison)
# =============================================================================

def rule_based_action(net, soc, price):
    """
    Simple deterministic heuristic — no learning, no forecasting.

    This is the baseline the ML controller is compared against.
    It runs as a "shadow" each step: it computes what it would do
    and tracks the hypothetical cost, but never modifies battery state.
    """
    if net > 0 and soc < 90:
        return "charge"
    elif net < 0 and soc > 20:
        return "discharge"
    else:
        return "idle"


# =============================================================================
# MicrogridModel
# =============================================================================

class MicrogridModel(Model):
    """
    Orchestrates all agents and runs the dispatch controllers.

    Each step:
      1. All agents update their state (wind produces, loads consume).
      2. The model assembles a BatteryBeliefs object from current state.
      3. get_intention() queries the RandomForest for a dispatch action.
      4. The action is executed on the real battery.
      5. The rule-based baseline runs as a shadow (no battery state change).
      6. Costs are tracked separately for both controllers.
    """

    def __init__(self):
        super().__init__()
        self.schedule   = RandomActivation(self)
        self.step_count = 0

        # Agents
        self.wind    = WindAgent(0, self)
        self.load1   = LoadAgent(1, self)
        self.load2   = LoadAgent(2, self)
        self.battery = BatteryAgent(3, self)

        for agent in (self.wind, self.load1, self.load2):
            self.schedule.add(agent)

        # Tracked state
        self.grid_import     = 0.0
        self.price           = 1.0
        self.total_cost      = 0.0
        self.rule_total_cost = 0.0
        self.history         = []

        # Train ML model once at startup
        self.ml_model = load_model()

    def step(self):
        self.step_count += 1
        self.schedule.step()

        production  = float(self.wind.production)
        consumption = float(self.load1.consumption + self.load2.consumption)
        net         = production - consumption

        # TOU pricing: off-peak $1, on-peak $3
        self.price = 1.0 if self.step_count % 10 < 5 else 3.0

        # ── BDI dispatch cycle ────────────────────────────────────────────
        beliefs   = BatteryBeliefs(self.battery.soc, production,
                                   consumption, self.price, net)
        ml_action = get_intention(beliefs, self.ml_model)

        self.grid_import = 0.0

        if ml_action == "charge":
            self.battery.charge(net)
        elif ml_action == "discharge":
            self.battery.discharge(abs(net))
        else:
            if net < 0:
                self.grid_import = abs(net)

        ml_cost          = self.grid_import * self.price
        self.total_cost += ml_cost

        # ── Rule-based shadow run ─────────────────────────────────────────
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

        # ── Record ───────────────────────────────────────────────────────
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
