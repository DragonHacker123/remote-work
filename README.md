# flydrive

Drive a racing car with a connectome-constrained model of the *Drosophila*
brain, and let it get faster lap after lap.

## What this actually is

There are three versions of "make the fly brain play F1", and it is worth being
blunt about which one this is.

1. **Boot the connectome and it drives.** Not possible. The connectome is
   wiring: who connects to whom and with how many synapses. It contains no
   synaptic strengths, no time constants, no neuromodulatory state. A fly also
   has no concept of a steering wheel.
2. **The connectome as a fixed architectural prior, with the remaining free
   parameters learned.** ← this repo.
3. **A "fly-inspired" network with boxes labelled after brain regions.** Easy
   and uninteresting.

In (2) the connectivity matrix is **fixed and never trained**. What is learned
is small and biologically shaped: one output gain, one time constant and one
resting drive per *cell type*; a sensory encoder; and a linear readout from the
descending neurons. About 1,100 parameters for a 686-neuron driving network.

## Why the fly, specifically

This is not an arbitrary pairing. The fly brain already contains a steering
controller, and the project borrows it rather than inventing one.

| circuit | real function | used here as |
| --- | --- | --- |
| EPG ring (ellipsoid body) | allocentric heading, ring attractor | compass |
| FC2 ring (fan-shaped body) | goal direction | where the racing line goes |
| **PFL3 L/R** | read heading with equal and opposite anatomical shifts; their difference is a turn command | **steering error → steering** |
| LAL → DNa02 | premotor, drives turning | steering actuator |
| T4/T5 → HS/VS | optic flow | yaw rate, sideslip |
| LPLC2 | looming detection | "the corner is arriving fast" |
| Ascending neurons | self-motion from the VNC | speed, slip, proprioception |
| Mushroom body (KC → MBON, DAN-gated) | associative learning | **lap-time improvement** |

The steering readout is not fitted into place — it falls out of the wiring.
Feed the EPG ring a heading bump and the FC2 ring a goal bump, and
`PFL3R − PFL3L` is a clean sine of the heading error:

```
$ flydrive probe-cx
 heading error   PFL3R-PFL3L    steer
     -90 deg   +0.2577 |----------------------- | +0.091
     -30 deg   +0.1264 |-----------             | +0.045
      +0 deg   -0.0318 |                      --| -0.011
     +30 deg   -0.1649 |         ---------------| -0.057
     +90 deg   -0.2410 |  ----------------------| -0.084

corr(PFL3 difference, sin(heading error)) = -0.991
corr(steer, heading error) = -0.739 (negative means corrective)
```

The heading and goal go in as **two separate bumps**, never as a precomputed
error. Computing that difference is the circuit's job, and handing it the
answer would be the whole trick.

## Quick start

```bash
pip install -e ".[dev]"
pytest                                  # 45 tests, ~30 s

flydrive info                           # connectome and track summary
flydrive reference --track national      # classical driver's lap time
flydrive probe-cx                       # the steering circuit, as above
flydrive train --track national --generations 200 --out theta.npz
flydrive session --theta theta.npz       # multi-lap MB plasticity session
```

## How it is trained

Three stages, because ES cannot bootstrap from a policy that crashes in the
first second — every candidate scores identically and there is no gradient to
climb.

1. **Closed-form readout fit.** Run a classical reference driver, let the
   network watch under teacher forcing, and ridge-regress the descending-neuron
   rates onto the driver's controls. Instant, and its R² answers the question
   that decides whether any of this can work: *how much of a driving policy is
   linearly available in the fly's motor bus?*
2. **ES on imitation loss.** Dense signal at every timestep. Unlike stage 1 this
   can reshape the *sensory encoder*, which is what actually limits steering
   accuracy.
3. **ES on driving reward** over a fixed horizon. With the horizon fixed,
   maximising progress *is* lap-time optimisation. Only this stage can beat the
   teacher, because only here is the objective speed rather than similarity.

Stages 1–3 are phylogeny: they tune what the animal is born with. The mushroom
body is ontogeny — see below.

## Learning to improve lap times

`flydrive session` runs repeated laps with the inherited parameters **frozen**.
Everything that changes lives in the KC→MBON synapses:

- Projection neurons carry the context (where on the circuit, how fast).
- Kenyon cells expand it into a sparse code via the connectome's own PN→KC
  wiring; APL's global inhibition keeps only the top ~10 active, which is what
  stops one corner's memory overwriting the next one's.
- Dopamine is a prediction error: this sector's time against the best the car
  has managed *there*, so the baseline is its own history.
- The update is the mushroom body's three-factor rule — presynaptic KC activity
  × postsynaptic perturbation × dopamine.

## The control experiment

This is what separates a result from a stunt, so it ships in the box:

```bash
python scripts/run_experiment.py --controls none,within-type
```

`within-type` is the strong null model. It preserves **every neuron's exact in-
and out-degree** *and* the complete cell-type-by-cell-type connectivity matrix,
destroying only neuron-level wiring specificity. Weaker controls (`pairing`,
`signs`, `weights`) are also available. Degree preservation is asserted in the
test suite, not assumed.

If the real connectome does not beat these, you have an elaborately-sourced
sparse RNN with a good origin story. Report it either way.

## Using the real FlyWire connectome

The repo ships a **surrogate** connectome: a scaled-down brain with the same
architectural motifs (ring attractor with Delta7 offset inhibition, PFL3 shift
geometry, KC sparse expansion with APL gain control, contralateral LAL
projections). This is so the whole pipeline runs offline and in CI. It is a
model of the architecture, not a substitute for the data.

For the real thing, download a Codex snapshot (registration required, data is
CC-BY-4.0) and point the loader at it:

```python
from flydrive.connectome.flywire import load_flywire
conn = load_flywire("data/flywire_783")     # connections.csv + classification.csv
```

Everything downstream is identical — same ports, same operator — so a model
developed on the surrogate re-fits on real wiring without touching the training
code. Cell-type naming varies between snapshots; extend `PORT_PATTERNS` in
`flydrive/connectome/flywire.py` if the port check fails.

## Driving F1 25

```bash
flydrive bridge record --track-file silverstone.npz   # drive one clean lap
flydrive bridge drive --track-file silverstone.npz --theta theta.npz
```

The simulator and the bridge build the observation vector with **the same
function** (`flydrive.sim.obs.observe`), so "the network sees the same thing in
the game as in training" is a fact about the code rather than an aspiration.

The parser reads only **Motion (id 0)** and **Car Telemetry (id 6)**, whose
layouts have been stable for years, and derives everything else — yaw rate by
differentiating heading, sideslip by projecting world velocity onto the car's
forward vector, slip angles from the same bicycle-model relations the simulator
uses. Motion Ex (id 13) carries per-wheel slip directly but its layout changes
most often.

**Untested against the real game.** There is no F1 25 in this environment. The
packet layouts are built from the published spec and verified byte-exact
against synthetic packets in `tests/test_bridge.py`, and every parse asserts
its expected size so a spec change fails loudly rather than producing plausible
garbage. Before trusting it, check:

- `LATERAL_SIGN` in `flydrive/bridge/f1_udp.py` — drive a known left-hand corner
  and confirm the reported yaw rate is positive.
- That the game is set to UDP format **2025** (it can also emit 2024/2023).
- Grip and mass differ from this simulator. Expect to need domain
  randomisation over `VehicleParams` before a trained policy transfers.

Use offline Time Trial only. This reads the game's own supported telemetry
output and pushes back through a normal virtual controller — nothing is
injected into the process — but driving an online session with synthetic input
is not what anti-cheat expects, and this project does not need it.

## Three things that turned out to matter

Recorded because each cost real debugging time and each is easy to get wrong
again.

**PFL3 must sit near threshold.** Its steering signal is the difference of two
*rectified* population sums. Let every wedge float above threshold and the sum
becomes linear — at which point the difference between two bumps is a constant
and the signal vanishes completely. Measured on the surrogate, threshold bias
takes the slope of `PFL3R − PFL3L` with respect to heading error from 0.0007 to
0.32, a factor of 450. The circuit looked fine in a static sweep and reported
nothing in closed loop until this was fixed. `THRESHOLD_BIAS` has a test.

**Aerodynamic drag is not a tyre force.** Charging drag to the friction ellipse
costs the rear axle ~20% of its lateral grip at speed and produces
inexplicable snap oversteer at every corner entry. Braking is front-biased and
must not be charged wholly to the rear either. Both have regression tests.

**A circuit needs a spread of corner speeds.** With realistic downforce
anything above roughly a 150 m radius is flat out, so procedurally-generated
"interesting" layouts came out as one hard corner and 3 km of full throttle —
which is not a driving task. Tracks are built from explicit corner sequences in
the 25–120 m radius band instead.

## Layout

```
flydrive/
  connectome/   schema (signed sparse, per-type parameter sharing), FlyWire
                loader, surrogate builder, null models
  sim/          track geometry, dynamic bicycle model with Pacejka tyres and
                downforce, vectorised env, the shared observation contract
  agents/       classical reference driver, connectome-constrained rate network
  learn/        ES, distillation, three-stage curriculum, MB plasticity
  bridge/       F1 25 UDP telemetry, virtual gamepad, live loop
scripts/        run_experiment.py -- real connectome vs shuffled control
tests/          45 tests; the central-complex ones are the load-bearing ones
```

## Limitations

- The surrogate connectome is a model of fly architecture, not fly data. All
  claims about "the connectome" are claims about that architecture until you
  run the FlyWire loader.
- Rate-based, not spiking. Fine for behaviour, wrong for anything about timing.
- The sensory encoder is invented. A fly has no speedometer and no curvature
  preview; those channels are engineering, not biology, and they are where a
  sceptic should look first.
- The bridge has never seen the real game.
