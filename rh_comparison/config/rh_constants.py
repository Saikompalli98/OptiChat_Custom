"""
RH Comparison Framework — Session State Keys and Path Constants.

All state keys follow the RH_ prefix convention to avoid any collision
with OptiChat's state namespace. Separate ADK app name ('optichat_rh')
guarantees namespace isolation at the session level.
"""

# ---------------------------------------------------------------------------
# ADK App Name
# ---------------------------------------------------------------------------
RH_APP_NAME = "optichat_rh"

# ---------------------------------------------------------------------------
# On-Disk Paths
# ---------------------------------------------------------------------------
RH_TMP_ROOT = "tmp/rh_epochs"
RH_REGISTRY_PATH = "tmp/rh_epochs/registry.json"
RH_COMPARISONS_DIR = "tmp/rh_epochs/comparisons"

# Comparison file names (relative to comparisons/{epoch_a}_vs_{epoch_b}/)
QT1_FILENAME = "qt1_structural.json"
QT2_FILENAME = "qt2_solution.json"
QT3_FILENAME = "qt3_backward.json"
QT4_FILENAME = "qt4_attribution.json"

# ---------------------------------------------------------------------------
# Session State Keys — Persistent (survive across queries in a session)
# ---------------------------------------------------------------------------
RH_SESSION_INITIALIZED = "RH_SESSION_INITIALIZED"

RH_EPOCH_A_ID   = "RH_EPOCH_A_ID"       # str: epoch_id for epoch A
RH_EPOCH_B_ID   = "RH_EPOCH_B_ID"       # str: epoch_id for epoch B
RH_EPOCH_A_META = "RH_EPOCH_A_META"     # dict: EpochMeta for epoch A
RH_EPOCH_B_META = "RH_EPOCH_B_META"     # dict: EpochMeta for epoch B

RH_DESCRIPTION_A = "RH_DESCRIPTION_A"  # str: natural language description of epoch A
RH_DESCRIPTION_B = "RH_DESCRIPTION_B"  # str: natural language description of epoch B

# QT results — None until first computed, then cached as dicts
RH_QT1_RESULT = "RH_QT1_RESULT"        # Computed eagerly at session init
RH_QT2_RESULT = "RH_QT2_RESULT"        # Computed on first user demand
RH_QT3_RESULT = "RH_QT3_RESULT"        # Computed on first user demand
RH_QT4_RESULT = "RH_QT4_RESULT"        # Computed on first user demand

# ---------------------------------------------------------------------------
# Session State Keys — Temporary (reset each query)
# ---------------------------------------------------------------------------
RH_QUERY_TYPE = "RH_QUERY_TYPE"        # Current query classification tag

# ---------------------------------------------------------------------------
# Default State Values
# ---------------------------------------------------------------------------
RH_PERSISTENT_STATES: dict = {
    RH_SESSION_INITIALIZED: False,
    RH_EPOCH_A_ID:          "",
    RH_EPOCH_B_ID:          "",
    RH_EPOCH_A_META:        {},
    RH_EPOCH_B_META:        {},
    RH_DESCRIPTION_A:       "",
    RH_DESCRIPTION_B:       "",
    RH_QT1_RESULT:          None,
    RH_QT2_RESULT:          None,
    RH_QT3_RESULT:          None,
    RH_QT4_RESULT:          None,
}

RH_TEMPORARY_STATES: dict = {
    RH_QUERY_TYPE: "GENERAL",
}

# ---------------------------------------------------------------------------
# Query Type Tags (used by root agent and callbacks)
# ---------------------------------------------------------------------------
class QueryType:
    GENERAL           = "GENERAL"
    RETRIEVAL         = "RETRIEVAL"
    MODEL_DESCRIPTION = "MODEL_DESCRIPTION"
    STRUCTURAL_CHANGE = "STRUCTURAL_CHANGE"
    SOLUTION_DIFF     = "SOLUTION_DIFF"
    BACKWARD_COMPAT   = "BACKWARD_COMPAT"
    ATTRIBUTION       = "ATTRIBUTION"

# ---------------------------------------------------------------------------
# Analytics Thresholds
# ---------------------------------------------------------------------------
class Thresholds:
    # Parameter change magnitude buckets (absolute relative change)
    MAGNITUDE_SMALL    = 0.10   # |Δ/v| < 10%
    MAGNITUDE_MODERATE = 0.30   # 10% ≤ |Δ/v| < 30%
    # |Δ/v| ≥ 30% → "large"

    # Minimum relative change to count as "changed" (numerical noise filter)
    PARAM_CHANGE_TOL = 1e-6

    # Variable change tolerance (solution comparison)
    VAR_CHANGE_TOL = 1e-5

    # Constraint binding tolerance (same as extract_tool.py)
    BINDING_TOL = 1e-5

    # Shadow price absolute significance threshold for QT4 bottleneck detection
    # Any |dual| > this value is considered significant (binding constraint)
    DUAL_SIGNIFICANCE_TOL = 1e-4

    # Minimum relative param shift to flag as a QT4 trigger
    TRIGGER_THRESHOLD = 0.10

# ---------------------------------------------------------------------------
# Upload Modes (config format field)
# ---------------------------------------------------------------------------
class UploadMode:
    SEPARATE = "separate"   # model .py + separate data .json
    EMBEDDED = "embedded"   # model .py with data already inside
