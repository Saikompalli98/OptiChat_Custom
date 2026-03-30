import streamlit as st
import requests
import json
import uuid
import base64
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="OptiChat")

API_BASE_URL = "http://localhost:8000"
OPTICHAT_APP   = "optichat"
RH_APP         = "optichat_rh"
RH_UPLOADS_DIR = "tmp/rh_uploads"

# ---------------------------------------------------------------------------
# Shared: user ID (persists for the browser session)
# ---------------------------------------------------------------------------
if "user_id" not in st.session_state:
    st.session_state.user_id = f"user_{uuid.uuid4().hex[:8]}"

# ---------------------------------------------------------------------------
# Mode selector — top of sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("OptiChat")
mode = st.sidebar.radio(
    "Mode",
    ["OptiChat", "RH Comparison"],
    index=0,
    key="app_mode",
)
st.sidebar.divider()

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _post_run(app_name: str, user_id: str, session_id: str, parts: list) -> list:
    """POST /run and return the list of events, or [] on failure."""
    try:
        resp = requests.post(
            f"{API_BASE_URL}/run",
            json={
                "app_name":   app_name,
                "user_id":    user_id,
                "session_id": session_id,
                "new_message": {"role": "user", "parts": parts},
            },
            timeout=300,
        )
        if resp.status_code == 200:
            return resp.json()
        st.error(f"API error {resp.status_code}: {resp.text}")
    except Exception as exc:
        st.error(f"Failed to reach API server: {exc}")
    return []


def _extract_text(events: list) -> str:
    """Pull meaningful text parts out of an ADK event list.

    Skips intermediate tool-response events (e.g. 'ok' confirmations from
    set_query_type) so they don't leak into the user-visible chat.
    """
    # Short tool confirmations that should never be shown to the user.
    _SKIP = {"ok", "ok.", "done", "done."}

    parts = []
    for event in events:
        content = event.get("content") or {}
        for part in content.get("parts", []):
            text = (part.get("text") or "").strip()
            if not text:
                continue
            # Skip single-word tool confirmations
            if text.lower() in _SKIP:
                continue
            # Skip legacy "Query type registered: ..." lines
            if text.lower().startswith("query type registered"):
                continue
            parts.append(text)
    return "\n".join(parts)


def _create_session(app_name: str, user_id: str, session_id: str) -> bool:
    """Create an ADK session. Returns True on success."""
    try:
        resp = requests.post(
            f"{API_BASE_URL}/apps/{app_name}/users/{user_id}/sessions",
            json={"session_id": session_id},
            timeout=30,
        )
        return resp.status_code == 200
    except Exception as exc:
        st.error(f"Failed to create session: {exc}")
        return False


def _save_upload(uploaded_file, dest_path: str) -> str:
    """Write a Streamlit UploadedFile to dest_path and return the absolute path."""
    Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
    with open(dest_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return os.path.abspath(dest_path)


# ===========================================================================
#  MODE A — OptiChat (existing functionality, unchanged)
# ===========================================================================
if mode == "OptiChat":

    st.title("OptiChat: Talk to your Optimization Model")

    # --- Sidebar: model selector ---
    METADATA_PATH = "tmp/model_objects/metadata.json"

    def load_metadata():
        if os.path.exists(METADATA_PATH):
            try:
                with open(METADATA_PATH, "r") as f:
                    return json.load(f)
            except Exception as e:
                st.sidebar.error(f"Error loading metadata: {e}")
        return {}

    metadata = load_metadata()
    selected_models = []

    if metadata:
        st.sidebar.subheader("Select Model")
        all_dates = sorted(list(metadata.keys()), reverse=True)
        for date in all_dates:
            with st.sidebar.expander(f"📅 {date}", expanded=False):
                base_models_data = metadata[date]
                for base_model_name in sorted(base_models_data.keys()):
                    st.markdown(f"**📓 {base_model_name}**")
                    base_key = f"chk_{date}_{base_model_name}_base"
                    if st.checkbox(f"{base_model_name} (Base)", key=base_key):
                        selected_models.append(base_model_name)
                    model_info = base_models_data[base_model_name]
                    if "modified_models" in model_info:
                        for mod_name in sorted(model_info["modified_models"].keys()):
                            mod_key = f"chk_{date}_{base_model_name}_{mod_name}"
                            if st.checkbox(mod_name, key=mod_key):
                                selected_models.append(mod_name)
                    st.divider()
        st.session_state.selected_models = selected_models
    else:
        st.sidebar.info("No models found. Upload a config to get started.")

    st.sidebar.subheader("Load Model Config")
    uploaded_json = st.sidebar.file_uploader("Upload JSON Config", type=["json"])

    show_model_representation = st.sidebar.checkbox("Show Model Representation", False)
    model_representation_placeholder = st.empty()

    show_code = st.sidebar.checkbox("Show Code", False)
    code_placeholder = st.empty()

    show_tech_feedback = st.sidebar.checkbox("Show Technical Feedback", False)

    if "messages" in st.session_state:
        chat_history_text = "\n\n".join(
            [f"{m['role']}: {m['content']}" for m in st.session_state.messages]
        )
        st.sidebar.download_button(
            label="Export Chat History",
            data=chat_history_text,
            file_name="chat_history.txt",
            mime="text/plain",
        )

    # --- Session setup ---
    if "session_id" not in st.session_state:
        st.session_state.session_id = f"session_{uuid.uuid4().hex[:8]}"
        if not _create_session(OPTICHAT_APP, st.session_state.user_id, st.session_state.session_id):
            st.error("Failed to create OptiChat session.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # --- Helper to fetch session state ---
    def fetch_session_state():
        try:
            url = (
                f"{API_BASE_URL}/apps/{OPTICHAT_APP}"
                f"/users/{st.session_state.user_id}"
                f"/sessions/{st.session_state.session_id}"
            )
            resp = requests.get(url)
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            pass
        return None

    # --- JSON Upload Handling ---
    if uploaded_json is not None:
        last_key = "last_uploaded_json"
        if last_key not in st.session_state or st.session_state[last_key] != uploaded_json.name:
            st.session_state[last_key] = uploaded_json.name
            json_content = uploaded_json.read()
            json_base64 = base64.b64encode(json_content).decode("utf-8")
            parts = [
                {
                    "text": "Initialize session with config",
                    "inline_data": {"mime_type": "application/json", "data": json_base64},
                }
            ]
            with st.spinner("Initializing session with uploaded config..."):
                events = _post_run(
                    OPTICHAT_APP,
                    st.session_state.user_id,
                    st.session_state.session_id,
                    parts,
                )
            if events:
                st.sidebar.success(f"Successfully uploaded {uploaded_json.name}")
                text = _extract_text(events)
                if text:
                    st.session_state.messages.append({"role": "assistant", "content": text})
            else:
                st.sidebar.error("Failed to initialize session with config.")

    # --- Visualization Rendering ---
    session_data = fetch_session_state()
    if session_data and "state" in session_data:
        state = session_data["state"]
        if show_model_representation:
            if "MODELS_DICTIONARY" in state:
                with model_representation_placeholder.container():
                    st.json(state["MODELS_DICTIONARY"])
        if show_code:
            if "CFG" in state and "models_code" in state["CFG"]:
                try:
                    paths = state["CFG"]["models_code"].get("local_resources", [])
                    code_content = ""
                    for path in paths:
                        if os.path.exists(path):
                            with open(path, "r") as f:
                                code_content += f"# File: {path}\n{f.read()}\n\n"
                    if code_content:
                        with code_placeholder.container():
                            st.code(code_content)
                except Exception:
                    pass
    else:
        if show_model_representation:
            model_representation_placeholder.info("No model data available yet.")
        if show_code:
            code_placeholder.info("No code available yet.")

    # --- Chat ---
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Enter your query here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if "selected_models" in st.session_state and st.session_state.selected_models:
            model_list_str = ", ".join(st.session_state.selected_models)
            context_message = f"[Using model: {model_list_str}] {prompt}"
        else:
            context_message = prompt

        with st.spinner("Thinking..."):
            events = _post_run(
                OPTICHAT_APP,
                st.session_state.user_id,
                st.session_state.session_id,
                [{"text": context_message}],
            )

        full_response = _extract_text(events)
        if full_response:
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            with st.chat_message("assistant"):
                st.markdown(full_response)


# ===========================================================================
#  MODE B — RH Comparison
# ===========================================================================
elif mode == "RH Comparison":

    st.title("Rolling Horizon Comparison")
    st.caption(
        "Compare two optimization model epochs side-by-side — structural changes, "
        "solution differences, backward compatibility, and attribution analysis."
    )

    # ------------------------------------------------------------------
    # Sidebar: epoch file uploaders
    # ------------------------------------------------------------------
    st.sidebar.subheader("Epoch A — Baseline")
    rh_label_a  = st.sidebar.text_input("Label", value="Epoch A", key="rh_label_a")
    rh_model_a  = st.sidebar.file_uploader("Model file (.py)", type=["py"],   key="rh_model_a")
    rh_data_a   = st.sidebar.file_uploader("Data file (.json, optional)",
                                            type=["json"], key="rh_data_a")

    st.sidebar.divider()

    st.sidebar.subheader("Epoch B — Updated")
    rh_label_b  = st.sidebar.text_input("Label", value="Epoch B", key="rh_label_b")
    rh_model_b  = st.sidebar.file_uploader("Model file (.py)", type=["py"],   key="rh_model_b")
    rh_data_b   = st.sidebar.file_uploader("Data file (.json, optional)",
                                            type=["json"], key="rh_data_b")

    st.sidebar.divider()

    both_uploaded = rh_model_a is not None and rh_model_b is not None
    init_btn = st.sidebar.button(
        "⚙️ Initialize Comparison Session",
        disabled=not both_uploaded,
        help="Upload both model files above to enable.",
    )

    if "rh_messages" not in st.session_state:
        st.session_state.rh_messages = []
    if "rh_initialized" not in st.session_state:
        st.session_state.rh_initialized = False
    if "rh_session_id" not in st.session_state:
        st.session_state.rh_session_id = None

    # Export button (only when chat has content)
    if st.session_state.rh_messages:
        rh_chat_text = "\n\n".join(
            [f"{m['role']}: {m['content']}" for m in st.session_state.rh_messages]
        )
        st.sidebar.download_button(
            label="Export Chat History",
            data=rh_chat_text,
            file_name="rh_chat_history.txt",
            mime="text/plain",
        )

    # ------------------------------------------------------------------
    # Initialize session when button clicked
    # ------------------------------------------------------------------
    if init_btn and both_uploaded:
        # Create a fresh session ID (allows re-initialization)
        rh_session_id = f"rh_session_{uuid.uuid4().hex[:8]}"

        # Save uploaded files to disk (use session ID for uniqueness)
        Path(RH_UPLOADS_DIR).mkdir(parents=True, exist_ok=True)
        base = os.path.join(RH_UPLOADS_DIR, rh_session_id)

        model_a_path = _save_upload(rh_model_a, f"{base}_epoch_a.py")
        model_b_path = _save_upload(rh_model_b, f"{base}_epoch_b.py")
        data_a_path  = _save_upload(rh_data_a,  f"{base}_epoch_a_data.json") if rh_data_a  else None
        data_b_path  = _save_upload(rh_data_b,  f"{base}_epoch_b_data.json") if rh_data_b  else None

        # Build JSON config for rh_initialize_session callback
        rh_config = {
            "epoch_a": {
                "epoch_id":   f"epoch_a_{rh_session_id}",
                "label":      rh_label_a,
                "model_path": model_a_path,
                "data_path":  data_a_path,
            },
            "epoch_b": {
                "epoch_id":   f"epoch_b_{rh_session_id}",
                "label":      rh_label_b,
                "model_path": model_b_path,
                "data_path":  data_b_path,
            },
        }

        config_bytes   = json.dumps(rh_config).encode("utf-8")
        config_base64  = base64.b64encode(config_bytes).decode("utf-8")

        # Create ADK session
        created = _create_session(RH_APP, st.session_state.user_id, rh_session_id)
        if not created:
            st.error("Failed to create RH session on the API server.")
            st.stop()

        # Send config to trigger rh_initialize_session callback
        with st.spinner(
            f"Loading '{rh_label_a}' and '{rh_label_b}', solving models, "
            "computing structural diff… (this may take a moment)"
        ):
            events = _post_run(
                RH_APP,
                st.session_state.user_id,
                rh_session_id,
                [
                    {
                        "text": "Initialize Rolling Horizon Comparison session",
                        "inline_data": {
                            "mime_type": "application/json",
                            "data": config_base64,
                        },
                    }
                ],
            )

        response_text = _extract_text(events)

        # Determine success: the callback returns None on success (no text from agent)
        # or an error/instruction message if config parsing failed.
        init_failed = (
            "failed" in response_text.lower()
            or "not yet configured" in response_text.lower()
            or "please provide" in response_text.lower()
        )

        if init_failed:
            st.error(f"Initialization error:\n\n{response_text}")
        else:
            st.session_state.rh_session_id  = rh_session_id
            st.session_state.rh_initialized = True
            st.session_state.rh_messages    = []   # fresh chat for this session

            # Show a welcome message
            welcome = (
                f"✅ **Session initialized.**\n\n"
                f"- **Epoch A**: {rh_label_a} (`{os.path.basename(model_a_path)}`)\n"
                f"- **Epoch B**: {rh_label_b} (`{os.path.basename(model_b_path)}`)\n\n"
                "Both models have been loaded and solved. You can now ask broad comparison "
                "questions, model-understanding questions, direct value-retrieval questions, "
                "or the four RH comparison analyses."
            )
            if response_text:
                welcome += f"\n\n---\n{response_text}"

            st.session_state.rh_messages.append({"role": "assistant", "content": welcome})
            st.sidebar.success("Session initialized successfully.")
            st.rerun()

    # ------------------------------------------------------------------
    # Main area: instructions or chat
    # ------------------------------------------------------------------
    if not st.session_state.rh_initialized:
        # Show instructions / getting-started guide
        st.info("Upload both model files in the sidebar and click **Initialize Comparison Session** to begin.")

        with st.expander("ℹ️ How to use Rolling Horizon Comparison", expanded=True):
            st.markdown(
                """
**What it does**

The RH Comparison mode lets you compare two versions of an optimization model
(two "epochs" from a rolling-horizon workflow). It runs four analytics modules
automatically:

| Analysis | What it answers |
|---|---|
| **QT1 — Structural Diff** | What variables, constraints, parameters, or index sets were added or removed? |
| **QT2 — Solution Diff** | How did the objective and variable values change? Which constraints changed binding status? |
| **QT3 — Backward Compat** | If you applied Epoch A's solution to Epoch B, would it still be feasible? How suboptimal would it be? |
| **QT4 — Attribution** | Which parameter shifts are associated with which solution changes? |

**Example questions you can ask**

- *"What structural changes happened between the two epochs?"*
- *"Which variables changed the most?"*
- *"Is the Epoch A solution still feasible in Epoch B?"*
- *"Why did the objective value change?"*
- *"What are the key parameter differences?"*
- *"What is the current value of x[tofu] in the later plan?"*
- *"How many food options are available in each run?"*
- *"Explain what this planning model is doing in plain English."*

**File format**

Upload standard Pyomo `.py` model files. Optionally upload a `.json` data file
if your model uses a separate data file (Mode A). Both epochs must be solvable
with Gurobi.
                """
            )
    else:
        # Show chat history
        for message in st.session_state.rh_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Accept input
        if rh_prompt := st.chat_input("Ask about the comparison…"):
            st.session_state.rh_messages.append({"role": "user", "content": rh_prompt})
            with st.chat_message("user"):
                st.markdown(rh_prompt)

            with st.spinner("Analyzing…"):
                events = _post_run(
                    RH_APP,
                    st.session_state.user_id,
                    st.session_state.rh_session_id,
                    [{"text": rh_prompt}],
                )

            full_response = _extract_text(events)
            if full_response:
                st.session_state.rh_messages.append(
                    {"role": "assistant", "content": full_response}
                )
                with st.chat_message("assistant"):
                    st.markdown(full_response)
            else:
                st.warning("No response received. Check the API server logs.")
