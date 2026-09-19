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
descending neurons. About 2,400 parameters over a 846-neuron, 16k-synapse
driving network — roughly three parameters per neuron, none of them a synapse.

The circuit it drives is the real Spa-Francorchamps, with real terrain: 6,928 m
and 106 m of elevation change, from a surveyed centreline georeferenced against
OpenStreetMap so SRTM height data can be sampled along it. Gradient and vertical
curvature are in the physics, not the scenery — see *Elevation* below.

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
     -90 deg   +0.2577 |----------------------- | +0.070
     -30 deg   +0.1264 |-----------             | +0.035
      +0 deg   -0.0318 |                      --| -0.010
     +30 deg   -0.1649 |         ---------------| -0.048
     +90 deg   -0.2410 |  ----------------------| -0.070

corr(PFL3 difference, sin(heading error)) = -0.991
corr(steer, heading error) = -0.738 (negative means corrective)
```

The heading and goal go in as **two separate bumps**, never as a precomputed
error. Computing that difference is the circuit's job, and handing it the
answer would be the whole trick.

## Quick start

```bash
pip install -e ".[dev]"
pytest -m "not slow"                    # 67 tests, ~35 s
pytest                                  # adds 2 end-to-end runs, several minutes

flydrive info                           # connectome and track summary
flydrive reference --track spa          # classical driver's lap time
flydrive probe-cx                       # the steering circuit, as above
flydrive train --track spa --generations 900 --out theta.npz
flydrive session --theta theta.npz      # multi-lap MB plasticity session
```

Spa ships in the repo as `flydrive/sim/data/spa.npz`. To rebuild it from the
public sources:

```bash
$ python scripts/fetch_spa.py --out flydrive/sim/data/spa.npz
$ flydrive info | tail -4
  spa              6928 m  min radius    16 m  climb  106 m  gradient -11%..+15%
```

## The circuit

Three sources, none of which has everything:

| source | gives | missing |
| --- | --- | --- |
| [TUM racetrack-database](https://github.com/TUMFTM/racetrack-database) | surveyed centreline at ~5 m spacing, measured left and right track widths | no georeference at all |
| [bacinger/f1-circuits](https://github.com/bacinger/f1-circuits) | the circuit in WGS84, from OpenStreetMap | only ~150 points |
| AWS terrain tiles | SRTM 1-arcsecond elevation | needs lat/lon |

So `scripts/fetch_spa.py` aligns the TUM geometry onto the OSM one — a Umeyama
similarity fit, searched over cyclic shift and direction, which lands at 12 m
mean error — converts the aligned points back to lat/lon and samples SRTM there.
The result keeps TUM's resolution and widths and gains real height: 363–469 m,
+15.0% at its steepest (Raidillon), −10.7% at its steepest descent.

### Elevation

Not decoration. Two terms:

- **Gravity along the road**, `−m g sin θ`. The climb out of Eau Rouge costs
  about a tenth of the available drive force; the drop to Stavelot hands it
  back, and a driver who ignores it arrives at the corner too fast.
- **Vertical curvature**, which scales tyre load by `cos θ + v² κ_v / g`. The
  compression at the bottom of Eau Rouge is worth most of an extra g; the crest
  at the top of the Kemmel climb takes grip away exactly where the car is
  braking for Les Combes.

Both are in the observation vector *and* in the reference driver's backward
speed-profile pass, so the classical driver brakes earlier downhill without
being told to.

### The racing line

The surveyed centreline is not driveable. Its tightest radius is 10.0 m at the
Bus Stop and 13.8 m at La Source; a real driver opens both out across the full
width of the road. So the path everything is measured against is a
**minimum-curvature racing line** solved through the corridor, which lifts the
tightest radius on the lap to 15.8 m:

| corner | centreline | racing line |
| --- | --- | --- |
| La Source | 13.8 m | 21.3 m |
| Eau Rouge | 153.4 m | 200.6 m |
| Les Combes | 28.5 m | 53.9 m |
| Pouhon | 70.8 m | 82.9 m |
| Stavelot | 30.2 m | 55.0 m |
| Bus Stop | 10.0 m | 15.8 m |

Offsets along the track normal make the path's second difference affine in the
offset, so this is a convex box-constrained QP, solved with an active set
rather than by clipping an unconstrained solve. Two details decide whether it
produces a racing line or noise, and both are written up under *Things that
turned out to matter* below.

Room either side of that line is asymmetric — at an apex there are a few
centimetres on the inside and ten metres on the outside — so the track carries
`hw_left` and `hw_right`, and going off is measured in metres past the white
line rather than as a multiple of the room available.

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

Stage 3 runs in **segments of growing horizon** — 18 s for half the budget, then
36 s, then 63 s. A short window is a dense, cheap signal for learning to take a
corner at all, but it cannot teach a car to survive a 7 km lap: at 18 s out of
118 s, a car that crashes at 17 s scores almost as well as one that survives, so
nothing selects for stringing corners together. Measured over 400 generations at
a fixed 18 s on Spa, the crash penalty being charged started at 37 and was still
23 at the end — essentially every car was still crashing inside the window. The
shares are weighted against cost, because a generation costs what its horizon
costs, and at 63 s the *implicit* penalty for crashing (the distance not
covered) is an order of magnitude larger than the explicit one.

Two more things are bolted onto stage 3 so that "it completes a lap" is not
where training stops:

- A **true lap time** is measured from the start line every 25 generations and
  the fastest parameters are kept separately from the highest-fitness ones.
  Fixed-horizon distance stops separating policies once they all survive the
  window; lap time does not.
- A **silence penalty** charges the fitness for every descending population the
  policy leaves unused, because reward alone has no reason to keep the fly's
  motor bus alive. See *Things that turned out to matter*.

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

## Measured

Reference driver, for scale — these are the lap times the learned network is
judged against:

| circuit | length | reference lap | average |
| --- | --- | --- | --- |
| **spa** (surveyed, with elevation) | **6928 m** | **118.22 s** | **211 km/h** |
| oval | 2142 m | 29.56 s | 261 km/h |
| national | 2757 m | 54.50 s | 182 km/h |
| gp | 4457 m | 71.10 s | 226 km/h |
| technical | 2236 m | 55.62 s | 145 km/h |

A real Spa qualifying lap is about 101 s, so a controller built from nothing but
textbook vehicle dynamics is roughly 17% off the pace of a Formula 1 driver in a
car it is only approximately modelling. That is the right order of magnitude to
make the comparison mean something.

**Driving.** The numbers below are from the **`national`** run, before the move
to surveyed Spa. Spa is a harder circuit — two and a half times the length, a
15.8 m hairpin, 106 m of elevation — and the results of training on it are not
in this table yet.

After the three-stage curriculum (50 imitation + 180 reward generations, ~50 min
on 4 cores), the connectome brain completes laps of `national`:

| driver | laps completed | best lap | average speed |
| --- | --- | --- | --- |
| classical reference | 32/32 | 54.06 s | 184 km/h |
| connectome brain | **31/32** | **81.88 s** | 121 km/h |

So it drives the whole circuit, about 1.5x slower than a controller built from
explicit vehicle dynamics. It does not yet beat the teacher, which is the
honest state of it — stage 3 optimises speed, and 180 generations on one
machine is not many. See the control experiment below for what the same
pipeline does on shuffled wiring; the comparison is not the clean win it might
look like from this table alone.

**Central complex.** `PFL3R − PFL3L` versus heading error: r = −0.99 against a
sine, steepest at zero error, corrective sign, and invariant to absolute
heading. In closed loop while driving on `national`, the PFL3 difference tracks
heading error at r = 0.88.

**Mushroom body.** 16 laps of `national` with inherited parameters frozen:

```
plasticity OFF  54.45  54.06  54.06  54.06 ... 54.06  54.06   (flat)
plasticity ON   54.89  54.34  54.24  54.29 ... 53.76  53.70   (-1.19 s)
```

All 24 cars survive; the entire improvement is in KC→MBON synapses.

**Descending-neuron readout.** How much of a competent driving policy is
linearly available in the fly's motor bus, measured on `national`:

| stage | steer R² | longitudinal R² |
| --- | --- | --- |
| closed-form fit, random encoder | 0.74 | 0.77 |
| after ES on imitation loss | **0.89** | 0.60 |

The jump in steering is the point: the closed-form fit takes the sensory
encoder as given, and only stage 2 can reshape it. That identifies the encoder,
not the 60-neuron readout, as what was limiting steering accuracy. Longitudinal
R² falls because the refit is done on states the *student* reaches rather than
the teacher's line, which is a harder and more honest target.

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

### What it showed

Identical pipeline, identical budget, identical seed — only the wiring differs.
**Measured on `national`**, before the move to surveyed Spa; it has not been
repeated on the harder circuit:

| | real connectome | within-type shuffle |
| --- | --- | --- |
| laps completed | **31/32** | **1/32** |
| best lap | 81.88 s | 63.06 s |
| mean distance before retiring | 2686 m (97%) | 1077 m (39%) |
| steer R², closed-form fit | 0.742 | 0.748 |
| steer R², after imitation ES | 0.888 | 0.854 |
| training progress, 18 s horizon | 555 m (best 645) | 562 m (best 753) |

Read the bottom three rows before the top one. By every *training-time*
measure the shuffle is indistinguishable from the real connectome — it fits the
readout equally well and covers marginally more ground per 18-second horizon.
The difference appears only when a full lap has to be completed: real wiring
finishes 31 laps out of 32, the shuffle finishes one.

And the one lap the shuffle did finish was **18 seconds faster**. So the honest
statement is not "the connectome drives better". It is that the real wiring
produced a far more *robust* policy while the shuffle produced a faster and
much more brittle one, and that the fixed-horizon training objective could not
tell them apart.

**This is one seed per arm.** A 31-versus-1 split is a large effect, but it is a
single sample of a stochastic pipeline, and the natural failure mode of this
kind of experiment is reporting seed variance as a finding. Replicate before
believing it:

```bash
for s in 0 1 2 3 4; do
  python scripts/run_experiment.py --seed $s --out results/seed$s
done
```

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

## Watching it learn

```bash
python scripts/record_training.py --out training.json   # trains, then replays every checkpoint
python scripts/build_viz.py training.json viz/index.html
```

Produces a single self-contained page: the car's attempts getting further across
every checkpoint, with the central complex drawn live beside them — the EPG
compass ring, the FC2 goal ring, the two PFL3 populations reading the compass at
opposite shifts, and the premotor and descending neurons they drive. The circuit
is shaded by altitude and there is an elevation profile under it with the car's
position marked, because a plan view of Spa cannot show the 106 m that make it
Spa.

Population activity is normalised against its maximum across the **whole run**
rather than per checkpoint, so activity organising over training is visible
instead of being rescaled away. That is what surfaced the dead-motor-channel
finding below — and the page's closing paragraph is now computed from the trace
rather than written by hand, so it reports what this run did instead of what the
last one did.

## Things that turned out to matter

Recorded because each cost real debugging time and each is easy to get wrong
again.

**A ridge term can delete the answer while looking harmless.** The
minimum-curvature solver regularises a second-difference operator whose small
eigenvalues go as `(2π/λ)⁴` for a feature of wavelength λ. At 1 m sampling a
ridge of `1e-6` suppresses everything longer than about 130 m — which is the
entire length scale a racing line works on. The first version returned offsets
of a few centimetres and looked, plausibly, like "the centreline is already
optimal". The ridge now exists only to make the factorisation safe and is
`1e-9`.

**Minimising curvature in index space rewards cheating.** Summing squared second
differences over the *centreline index* is `∫ κ² h³ ds` for a local spacing `h`.
Pulling the line onto the inside of a corner shrinks `h`, so the solver could
lower its objective while making the corner physically tighter — La Source went
from 14.6 m to 9.4 m, the exact opposite of the intent. Reweighting by `1/h³`
from the previous iterate restores the real integral. And `∫κ²` is a lap-length
average that will spend a hairpin to buy back a little on the sweepers, so the
objective is raised to the fourth power by iteratively reweighted least squares.
That reweighting is a fixed point rather than a descent and can cycle, so every
iterate is scored on peak curvature and the best is kept — with the centreline
in the running, which means the routine can decline to produce a line at all.

**A training horizon is a claim about what you are selecting for.** Stage 3
scored a policy on distance covered in a fixed 18-second window, which on a
2.7 km layout is 22% of a lap and worked fine. On Spa it is 15% of one, and it
quietly stops selecting for survival: a car that crashes at 17 s scores almost
as well as a car that gets round. The symptom was not obvious — fitness and
distance both climbed for 400 generations — and it only showed up by subtracting
fitness from the progress reward to recover what the crash penalty was actually
charging. It never fell below 23 against a penalty of 30. More generations would
not have fixed it, because the objective could not see the difference.

**Pedal travel is not a force.** The reference driver's friction ellipse
compared brake *pedal* against a grip *fraction*. Full brake is 32 kN, about
twice what the tyres can transmit at 35 m/s, so `brake ≤ 0.78` still permitted a
demand well past the limit. Worse, it checked the car as a whole, when what
actually lets go on corner entry is the rear axle by itself: it carries the
smaller share of the weight and braking transfers load *off* it. Because the
available load depends on the force being solved for, the rear ellipse is a
quadratic in brake force. Before this fix the reference driver retired at La
Source on every single lap; after it, 117.30 s and a lap everywhere else too.

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

**Training will abandon the biological motor readout unless you pay it not
to.** The decoder is initialised to the fly's own convention — the left–right
difference across DNa02 steers, DNa01 sets speed, DNp09 stops. An earlier run
ended with DNa02 and DNa01 **never firing at all**: ES drove them below a hard
rectifier and steered the car entirely through DNp09, nominally the *stop*
channel, plus a constant offset. Twelve of sixty descending neurons carried
everything. The central complex upstream still computed heading error
correctly, so the circuit that matters survived, but the motor bus collapsed to
one channel because nothing in the objective valued using it. Two changes: a
softplus rate law so a population below threshold still has a gradient, and an
explicit penalty on descending populations the policy never drives. The
visualiser now reports each channel's peak rate at the first checkpoint versus
the best one and says which verdict the data supports, rather than restating
last run's.

**A circuit needs a spread of corner speeds.** With realistic downforce
anything above roughly a 150 m radius is flat out, so procedurally-generated
"interesting" layouts came out as one hard corner and 3 km of full throttle —
which is not a driving task. Synthetic tracks are built from explicit corner
sequences in the 25–120 m radius band instead. Real Spa needs no such help: it
spans a 15.8 m hairpin and a 2 km flat-out straight.

## Layout

```
flydrive/
  connectome/   schema (signed sparse, per-type parameter sharing), FlyWire
                loader, surrogate builder, null models
  sim/          track geometry with elevation and the racing-line solver,
                dynamic bicycle model with Pacejka tyres, downforce and road
                gradient, vectorised env, the shared observation contract
  sim/data/     spa.npz -- the surveyed circuit, built by scripts/fetch_spa.py
  agents/       classical reference driver, connectome-constrained rate network
  learn/        ES, distillation, three-stage curriculum, MB plasticity
  bridge/       F1 25 UDP telemetry, virtual gamepad, live loop
scripts/        fetch_spa.py       -- build the circuit from public sources
                run_experiment.py  -- real connectome vs shuffled control
                record_training.py -- train, then replay every checkpoint
                build_viz.py       -- inline a trace into the visualiser
tests/          67 tests; the central-complex ones are the load-bearing ones
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
- The car is an F1 car in the way a bicycle model with one Pacejka curve per
  axle is: understeer, load transfer, downforce and a friction ellipse are
  there; tyre temperature, suspension, differential and DRS are not.
- Spa's elevation is SRTM at 30 m horizontal resolution, smoothed along the
  lap. It gets Eau Rouge and the Kemmel climb right; it is not a survey of the
  kerbs, and camber is not modelled at all.
- The racing line is minimum-curvature, not minimum-lap-time. A real optimal
  line trades curvature against where the car can use its power, which needs
  the vehicle model inside the optimisation.
- The brain laps slower than a classical controller. That gap is the headline
  open problem, and the training budget here is small enough that it is not yet
  evidence of a ceiling.
- Mushroom-body sessions run many cars sharing one set of KC→MBON weights.
  That multiplies the learning signal per lap and is a convenience, not a
  claim about flies.
