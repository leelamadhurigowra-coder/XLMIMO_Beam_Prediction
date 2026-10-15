
import json
import numpy as np
from pathlib import Path
from scipy.io import loadmat

# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

RESULTS_DIR.mkdir(exist_ok=True)

CONFIG_FILE = DATA_DIR / "beam_config.json"

# ============================================================
# 2. LOAD CONFIGURATION
# ============================================================

with open(CONFIG_FILE, "r") as f:
    config = json.load(f)

FC = config["carrier_frequency_hz"]
BW = config["bandwidth_hz"]
N_SC = config["num_subcarriers"]

NX = config["num_tx_antennas_x"]
NY = config["num_tx_antennas_y"]

N_ANT = NX * NY
C = config["speed_of_light_mps"]

D_LAMBDA = config["antenna_spacing_wavelengths"]

N_BEAMS_X = config["num_beams_x"]
N_BEAMS_Y = config["num_beams_y"]

N_BEAMS = N_BEAMS_X * N_BEAMS_Y

BATCH_SIZE = config["batch_size"]

print("=" * 65)
print("OPTIMAL BEAM LABEL GENERATION")
print("=" * 65)

print("Carrier frequency:", FC / 1e9, "GHz")
print("Bandwidth:", BW / 1e6, "MHz")
print("Subcarriers:", N_SC)
print("Transmit antennas:", N_ANT)
print("Candidate beams:", N_BEAMS)

# ============================================================
# 3. LOAD ORIGINAL DEEPMIMO DATA
# ============================================================

def load_variable(prefix):

    filename = DATA_DIR / (
        f"{prefix}_t001_tx000_r000.mat"
    )

    if not filename.exists():
        raise FileNotFoundError(
            f"Missing file: {filename}"
        )

    mat = loadmat(filename)

    if prefix not in mat:
        raise KeyError(
            f"Variable '{prefix}' not found in {filename}"
        )

    return np.asarray(mat[prefix], dtype=np.float64)


power_db = load_variable("power")
phase_deg = load_variable("phase")
delay = load_variable("delay")

aod_az_deg = load_variable("aod_az")
aod_el_deg = load_variable("aod_el")

print("\nRaw data shape:", power_db.shape)

# ============================================================
# 4. VALID PATH MASK
# ============================================================

valid = (
    np.isfinite(power_db)
    & np.isfinite(phase_deg)
    & np.isfinite(delay)
    & np.isfinite(aod_az_deg)
    & np.isfinite(aod_el_deg)
    & (delay >= 0)
)

valid_receiver = np.any(valid, axis=1)

receiver_indices = np.flatnonzero(valid_receiver)

print("Total receivers:", len(valid_receiver))
print("Usable receivers:", len(receiver_indices))

# ============================================================
# 5. FREQUENCY GRID
# ============================================================

# Center the subcarriers around the carrier frequency.
# The spacing is bandwidth / number of subcarriers.

delta_f = BW / N_SC

frequencies = (
    FC
    + (
        np.arange(N_SC) - (N_SC - 1) / 2
    ) * delta_f
)

print("Subcarrier spacing:", delta_f / 1e6, "MHz")
print(
    "Frequency range:",
    frequencies[0] / 1e9,
    "to",
    frequencies[-1] / 1e9,
    "GHz"
)

# ============================================================
# 6. TRANSMIT UPA ELEMENT POSITIONS
# ============================================================

# Half-wavelength spacing at carrier frequency.

wavelength = C / FC
d = D_LAMBDA * wavelength

x = np.arange(NX) * d
y = np.arange(NY) * d

xx, yy = np.meshgrid(x, y, indexing="ij")

antenna_positions = np.column_stack(
    (
        xx.ravel(),
        yy.ravel(),
        np.zeros(N_ANT)
    )
)

# ============================================================
# 7. CREATE 64-BEAM CODEBOOK
# ============================================================

# Eight azimuth directions and eight polar-angle directions.
# The polar angle is measured from the positive z-axis.
# These are candidate steering directions, not measured beams.

azimuth_grid = np.linspace(
    -157.5,
    157.5,
    N_BEAMS_X
)

polar_grid = np.linspace(
    22.5,
    157.5,
    N_BEAMS_Y
)

beam_directions = []

for theta_deg in polar_grid:
    for phi_deg in azimuth_grid:

        theta = np.deg2rad(theta_deg)
        phi = np.deg2rad(phi_deg)

        ux = np.sin(theta) * np.cos(phi)
        uy = np.sin(theta) * np.sin(phi)
        uz = np.cos(theta)

        beam_directions.append(
            [ux, uy, uz]
        )

beam_directions = np.asarray(
    beam_directions
)

# Carrier-frequency steering vectors.
# Each row corresponds to one candidate beam.

k0 = 2 * np.pi * FC / C

phase_codebook = (
    k0
    * beam_directions
    @ antenna_positions.T
)

codebook = (
    np.exp(1j * phase_codebook)
    / np.sqrt(N_ANT)
)

print("Codebook shape:", codebook.shape)

# ============================================================
# 8. CONVERT PATH ANGLES TO DIRECTION COSINES
# ============================================================

# Assumption:
# DeepMIMO elevation values are interpreted here as polar
# angles measured from the positive z-axis.
#
# Verify this convention against the scenario documentation
# before treating results as final research measurements.

azimuth = np.deg2rad(aod_az_deg)
polar = np.deg2rad(aod_el_deg)

ux = np.sin(polar) * np.cos(azimuth)
uy = np.sin(polar) * np.sin(azimuth)
uz = np.cos(polar)

# ============================================================
# 9. CONVERT PATH POWER AND PHASE TO COMPLEX GAINS
# ============================================================

# Assumption:
# power values are in dB and represent path power.
# Therefore amplitude = 10^(power_dB / 20).

amplitude = 10 ** (power_db / 20)

phase_rad = np.deg2rad(phase_deg)

path_gain = (
    amplitude
    * np.exp(1j * phase_rad)
)

# Invalid paths contribute zero.

path_gain = np.where(
    valid,
    path_gain,
    0.0 + 0.0j
)

# ============================================================
# 10. GENERATE OPTIMAL BEAM LABELS
# ============================================================

N_VALID = len(receiver_indices)

# One label per receiver per subcarrier.
labels = np.zeros(
    (N_VALID, N_SC),
    dtype=np.int16
)

# Store best beam gain for evaluation.
best_gain = np.zeros(
    (N_VALID, N_SC),
    dtype=np.float32
)

print("\nStarting channel reconstruction...")

for start in range(0, N_VALID, BATCH_SIZE):

    end = min(
        start + BATCH_SIZE,
        N_VALID
    )

    original_rows = receiver_indices[start:end]

    batch_size = len(original_rows)

    batch_labels = np.zeros(
        (batch_size, N_SC),
        dtype=np.int16
    )

    batch_gain = np.zeros(
        (batch_size, N_SC),
        dtype=np.float32
    )

    for sc, frequency in enumerate(frequencies):

        k = 2 * np.pi * frequency / C

        # Frequency-dependent propagation channel.
        H = np.zeros(
            (batch_size, N_ANT),
            dtype=np.complex128
        )

        for p in range(power_db.shape[1]):

            alpha = path_gain[
                original_rows, p
            ]

            # Propagation delay phase.
            delay_phase = np.exp(
                -1j * 2 * np.pi
                * frequency
                * delay[original_rows, p]
            )

            # Spatial phase across the physical array.
            spatial_dot = (
                ux[original_rows, p, None]
                * antenna_positions[None, :, 0]
                +
                uy[original_rows, p, None]
                * antenna_positions[None, :, 1]
                +
                uz[original_rows, p, None]
                * antenna_positions[None, :, 2]
            )

            spatial_phase = np.exp(
                1j * k * spatial_dot
            )

            H += (
                alpha[:, None]
                * delay_phase[:, None]
                * spatial_phase
            )

        # Evaluate all candidate beams.
        #
        # H:        (batch, antennas)
        # codebook: (beams, antennas)
        # response: (batch, beams)

        response = (
            H @ codebook.conj().T
        )

        gain = np.abs(response) ** 2

        # Select strongest beam for this subcarrier.
        optimal_beam = np.argmax(
            gain,
            axis=1
        )

        optimal_gain = np.max(
            gain,
            axis=1
        )

        batch_labels[:, sc] = optimal_beam
        batch_gain[:, sc] = optimal_gain

    labels[start:end] = batch_labels
    best_gain[start:end] = batch_gain

    print(
        f"Processed {end}/{N_VALID} receivers"
    )

# ============================================================
# 11. SAVE LABELS AND METADATA
# ============================================================

label_file = RESULTS_DIR / "beam_labels.npy"

np.save(label_file, labels)

np.save(
    RESULTS_DIR / "beam_label_receiver_indices.npy",
    receiver_indices
)

np.save(
    RESULTS_DIR / "beam_label_best_gain.npy",
    best_gain
)

np.save(
    RESULTS_DIR / "beam_frequencies_hz.npy",
    frequencies
)

report = {
    "total_receivers": int(len(valid_receiver)),
    "usable_receivers": int(N_VALID),
    "num_subcarriers": int(N_SC),
    "num_tx_antennas": int(N_ANT),
    "num_candidate_beams": int(N_BEAMS),
    "carrier_frequency_hz": float(FC),
    "bandwidth_hz": float(BW),
    "subcarrier_spacing_hz": float(delta_f),
    "label_shape": list(labels.shape),
    "angle_convention": (
        "Azimuth plus polar angle from positive z-axis; "
        "verify against original scenario documentation."
    ),
    "power_assumption": (
        "Path power treated as dB; amplitude = 10^(power_dB/20)."
    ),
    "array_model": (
        "8x8 planar array, half-wavelength spacing at carrier."
    ),
    "beam_codebook": (
        "64 candidate steering directions; "
        "carrier-frequency phase-shifter codebook."
    )
}

with open(
    RESULTS_DIR / "beam_label_report.json",
    "w"
) as f:

    json.dump(
        report,
        f,
        indent=4
    )

print("\n" + "=" * 65)
print("BEAM LABEL GENERATION COMPLETED")
print("=" * 65)

print("Labels shape:", labels.shape)
print("Saved:", label_file)
print("Report saved.")

print(
    "Unique beam labels:",
    np.unique(labels)
)

print(
    "Label range:",
    labels.min(),
    "to",
    labels.max()
)

print("\nNext step: Train the machine learning models.")