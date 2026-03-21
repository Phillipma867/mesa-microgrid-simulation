# mesa-microgrid-simulation

A microgrid simulation I built using Mesa as part of my GSoC 2026 application for the Behavioral Framework project.

The basic idea is pretty simple: you have a wind turbine, a battery, and some loads all running as Mesa agents. Each step, the system has to decide whether to charge the battery, discharge it, or just buy from the grid. I wanted to see if a machine learning model could make better decisions than a simple rule, and also figure out where Mesa made this kind of behavioral modeling easy or hard.

---

## How it works

There are four agents in the model. `WindAgent` generates power based on a sine wave with some noise added — production floats roughly between 5 and 15 kW. Two `LoadAgent` instances consume power and cycle between 5 kW and 10 kW on slightly different schedules so the total demand actually varies. `BatteryAgent` stores energy with 95% round-trip efficiency and keeps SOC between 0 and 100%.

Every step, `MicrogridModel` looks at how much wind is being produced vs how much load is being consumed, checks the current electricity price (it alternates between $1/kWh and $3/kWh to simulate off-peak and on-peak), and asks a Random Forest classifier what the battery should do: charge, discharge, or idle.

The model also runs a simple rule-based controller in parallel — charge if there's surplus, discharge if there's deficit, otherwise buy from the grid. It doesn't actually change the battery, it just tracks what the cost would have been. This way I can compare the two controllers directly on the same simulation run.

---

## Why I built it this way

I'm studying electrical engineering and I've been looking at how machine learning can improve energy management in small grids. When I found the Mesa Behavioral Framework GSoC project, it matched exactly what I was already thinking about — how do you model agents that have internal goals and priorities, not just reactive rules?

The battery in this project is basically a simplified BDI agent. It has beliefs (what's the current SOC, what's the wind doing, what does electricity cost right now), it has a goal (keep costs low), and it has to pick an action each step. Building this showed me pretty concretely where Mesa helps and where it doesn't, which is exactly what the Behavioral Framework project is trying to understand.

---

## Setup

```bash
pip install mesa streamlit plotly pandas scikit-learn numpy
streamlit run app.py
```

Then go to `http://localhost:8501`. I'd recommend running at least 5000 steps to see the cost comparison charts become meaningful.

---

## Files

`model.py` has all the Mesa agents and the model class. `ml_model.py` trains the Random Forest on synthetic data and exposes a `predict()` function. `app.py` is the Streamlit dashboard. `BEHAVIORAL_NOTES.md` has my notes on what was easy and what was painful about using Mesa for this kind of behavioral model.

---

## License

Apache 2.0
