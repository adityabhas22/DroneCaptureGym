/* ============================================================
 * DroneCaptureOps · Mission Console — UI logic
 * ============================================================ */

const ROUTES = {
  live: {
    sessions: "/live/sessions",
    session: (id) => `/live/sessions/${encodeURIComponent(id)}`,
    reset: (id) => `/live/sessions/${encodeURIComponent(id)}/reset`,
    step: (id) => `/live/sessions/${encodeURIComponent(id)}/step`,
    run: (id) => `/live/sessions/${encodeURIComponent(id)}/run`,
    runModel: (id) => `/live/sessions/${encodeURIComponent(id)}/run_model`,
    replay: (id) => `/live/sessions/${encodeURIComponent(id)}/replay`,
    events: (id) => `/live/sessions/${encodeURIComponent(id)}/events`,
    eventsStream: (id) => `/live/sessions/${encodeURIComponent(id)}/events/stream`,
    compare: "/live/compare",
    tasks: "/live/tasks",
    suites: "/live/suites",
  },
  openenv: {
    reset: "/reset",
    step: "/step",
    state: "/state",
    metadata: "/metadata",
    schema: "/schema",
  },
};

const ACTION_PRESETS = {
  get_mission_checklist: {},
  get_site_map: {},
  get_telemetry: {},
  list_assets: {},
  takeoff: { altitude_m: 18 },
  fly_to_viewpoint: { x: 30, y: 24, z: 22, yaw_deg: -90, speed_mps: 4 },
  move_to_asset: { asset_id: "row_B6", standoff_bucket: "mid", speed_mps: 4 },
  set_gimbal: { pitch_deg: -56, yaw_deg: -90 },
  set_camera_source: { source: "thermal" },
  capture_thermal: { label: "thermal-overview" },
  capture_rgb: { label: "rgb-context" },
  inspect_capture: { photo_id: "photo_001" },
  return_home: {},
  land: {},
};

const MISSION_PHASES = [
  { key: "preflight", label: "Preflight" },
  { key: "takeoff", label: "Takeoff" },
  { key: "survey", label: "Survey" },
  { key: "anomaly_capture", label: "Anomaly capture" },
  { key: "return", label: "Return" },
  { key: "land", label: "Land" },
  { key: "report", label: "Report" },
];

const REWARD_COMPONENTS = [
  { key: "evidence_success", label: "Evidence", color: "#a8e6a3" },
  { key: "required_coverage", label: "Coverage", color: "#8ad6c1" },
  { key: "issue_capture", label: "Issues", color: "#f4b860" },
  { key: "operational_efficiency", label: "Efficiency", color: "#c08acb" },
  { key: "grounded_report", label: "Report", color: "#f49551" },
  { key: "process_reward", label: "Process", color: "#ddd2bd" },
];

const REWARD_MICRO = [
  "capture_quality",
  "checklist_completion",
  "battery_management",
  "safety_compliance",
];

const MODEL_PRESETS = {
  hf: {
    model: "Qwen/Qwen3-4B-Instruct-2507",
    apiBase: "https://router.huggingface.co/v1",
    maxSteps: "12",
  },
  openai: {
    model: "gpt-5.4-mini",
    apiBase: "",
    maxSteps: "8",
  },
  anthropic: {
    model: "claude-haiku-4-5-20251001",
    apiBase: "",
    maxSteps: "8",
  },
  scripted: {
    model: "",
    apiBase: "",
    maxSteps: "12",
  },
  random: {
    model: "",
    apiBase: "",
    maxSteps: "12",
  },
};

const PALETTE = {
  cream: "#f4ecdc",
  creamSoft: "#ddd2bd",
  muted: "#93a09b",
  dim: "#5d6863",
  amber: "#f4b860",
  amberWarm: "#f49551",
  sage: "#8ad6c1",
  sageDeep: "#5fb29b",
  coral: "#ff8a6b",
  coralDeep: "#d3573b",
  lime: "#b8e8a8",
  plum: "#c08acb",
  ink: "#0b1311",
  inkSoft: "#10181a",
  ground: "rgba(244, 236, 220, 0.08)",
};

const state = {
  apiMode: "openenv",
  connection: "offline",
  sessionId: null,
  observation: null,
  scene: null,
  routeTrail: [],
  actionLog: [],
  liveSessions: [],
  comparison: null,
  lastRaw: null,
  sessionStartedAt: null,
  camera: {
    zoom: 1,
    panX: 0,
    panY: 0,
    lock: null,
  },
  projector: null,
};

const el = {
  apiMode: document.querySelector("#apiMode"),
  sessionId: document.querySelector("#sessionId"),
  connectionState: document.querySelector("#connectionState"),
  missionTitle: document.querySelector("#missionTitle"),
  missionClock: document.querySelector("#missionClock"),

  sessionForm: document.querySelector("#sessionForm"),
  refreshSessions: document.querySelector("#refreshSessions"),
  taskInput: document.querySelector("#taskInput"),
  seedInput: document.querySelector("#seedInput"),
  backendInput: document.querySelector("#backendInput"),
  familyInput: document.querySelector("#familyInput"),

  objectivePrompt: document.querySelector("#objectivePrompt"),
  modelPolicy: document.querySelector("#modelPolicy"),
  modelMaxSteps: document.querySelector("#modelMaxSteps"),
  modelName: document.querySelector("#modelName"),
  modelApiBase: document.querySelector("#modelApiBase"),
  modelApiKey: document.querySelector("#modelApiKey"),
  modelAdapterPath: document.querySelector("#modelAdapterPath"),
  runModelButton: document.querySelector("#runModelButton"),
  replaySourcePath: document.querySelector("#replaySourcePath"),
  replayRecordIndex: document.querySelector("#replayRecordIndex"),
  replayButton: document.querySelector("#replayButton"),

  toolSelect: document.querySelector("#toolSelect"),
  argsInput: document.querySelector("#argsInput"),
  quickActions: document.querySelector("#quickActions"),
  stepButton: document.querySelector("#stepButton"),
  clearLog: document.querySelector("#clearLog"),

  actionLog: document.querySelector("#actionLog"),
  actionCount: document.querySelector("#actionCount"),

  captureList: document.querySelector("#captureList"),
  artifactCount: document.querySelector("#artifactCount"),

  rewardComposition: document.querySelector("#rewardComposition"),
  rewardHeadline: document.querySelector("#rewardHeadline"),
  rewardMode: document.querySelector("#rewardMode"),
  rewardMicro: document.querySelector("#rewardMicro"),

  batteryHeadline: document.querySelector("#batteryHeadline"),
  batteryReserve: document.querySelector("#batteryReserve"),
  batteryRingFill: document.querySelector("#batteryRingFill"),
  batteryReserveTick: document.querySelector("#batteryReserveTick"),

  altitudeHeadline: document.querySelector("#altitudeHeadline"),
  altitudeBand: document.querySelector("#altitudeBand"),
  altMarker: document.querySelector("#altMarker"),

  speedHeadline: document.querySelector("#speedHeadline"),
  speedSub: document.querySelector("#speedSub"),
  flightMicro: document.querySelector("#flightMicro"),

  hudPose: document.querySelector("#hudPose"),
  hudHeading: document.querySelector("#hudHeading"),
  hudSensor: document.querySelector("#hudSensor"),
  hudGimbal: document.querySelector("#hudGimbal"),
  compassNeedle: document.querySelector("#compassNeedle"),
  scaleBarFill: document.querySelector("#scaleBarFill"),
  scaleLabel: document.querySelector("#scaleLabel"),

  failureBanner: document.querySelector("#failureBanner"),
  canvas: document.querySelector("#sceneCanvas"),
  zoomLevel: document.querySelector("#zoomLevel"),
  cameraButtons: document.querySelectorAll(".scene-controls__btn"),
  phaseStrip: document.querySelector("#phaseStrip"),
  briefingFacts: document.querySelector("#briefingFacts"),
  missionInstruction: document.querySelector("#missionInstruction"),
  suggestedTools: document.querySelector("#suggestedTools"),
  blockersList: document.querySelector("#blockersList"),

  compareSpecs: document.querySelector("#compareSpecs"),
  compareButton: document.querySelector("#compareButton"),
  overlayButton: document.querySelector("#overlayButton"),
  compareResults: document.querySelector("#compareResults"),
};

const ctx = el.canvas.getContext("2d");

/* ===========================================================
 * Connection / API helpers
 * =========================================================== */

function setConnection(status) {
  state.connection = status;
  el.connectionState.textContent = status;
  el.connectionState.classList.toggle("status-pill--online", status === "online");
  el.connectionState.classList.toggle("status-pill--hot", status !== "online");
}

async function requestJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
    body: options.body === undefined ? undefined : JSON.stringify(options.body),
  });
  const text = await response.text();
  const data = text ? safeJson(text) : null;
  if (!response.ok) {
    const error = new Error(data?.detail || data?.message || `${response.status} ${response.statusText}`);
    error.status = response.status;
    error.data = data;
    throw error;
  }
  return data;
}

function safeJson(text) {
  try {
    return JSON.parse(text);
  } catch {
    return { text };
  }
}

function normalizeEnvelope(payload) {
  const rawObservation =
    payload?.observation ||
    payload?.next_observation ||
    payload?.scene?.observation ||
    payload?.state?.observation ||
    payload?.scene ||
    payload ||
    {};
  const reward =
    payload?.reward ??
    payload?.event?.reward ??
    rawObservation?.reward ??
    rawObservation?.reward_breakdown?.total ??
    0;
  const done = payload?.done ?? rawObservation?.done ?? false;
  const scene = payload?.scene || rawObservation?.scene || null;
  const observation = { ...rawObservation, scene, reward, done };
  const sessionId =
    payload?.session_id ||
    payload?.id ||
    payload?.session?.id ||
    payload?.session?.session_id ||
    state.sessionId;
  return { observation, scene, reward, done, sessionId, raw: payload };
}

function buildResetPayload() {
  const payload = {
    seed: Number.parseInt(el.seedInput.value || "0", 10),
    backend: el.backendInput.value || "geometry",
  };
  const task = el.taskInput.value.trim();
  const family = el.familyInput.value.trim();
  if (task) {
    payload.task = task;
    payload.task_id = task;
  }
  if (family) {
    payload.scenario_family = family;
  }
  return payload;
}

async function startSession(event) {
  event?.preventDefault();
  setConnection("connecting");
  const resetPayload = buildResetPayload();
  let envelope;

  try {
    const payload = await requestJson(ROUTES.live.sessions, { method: "POST", body: resetPayload });
    envelope = normalizeEnvelope(payload);
    state.apiMode = "live";
    state.sessionId = envelope.sessionId;
  } catch (error) {
    if (error.status && ![404, 405].includes(error.status)) {
      logAction("session.create", false, error.message);
    }
    const payload = await requestJson(ROUTES.openenv.reset, { method: "POST", body: resetPayload });
    envelope = normalizeEnvelope(payload);
    state.apiMode = "openenv";
    state.sessionId = envelope.sessionId || "singleton";
  }

  state.routeTrail = [];
  state.sessionStartedAt = Date.now();
  resetCamera();
  applyEnvelope(envelope);
  logAction("session.reset", true, missionTitle(envelope.observation) || "Mission initialized", resetPayload);
  setConnection("online");
}

async function executeStep() {
  const toolName = el.toolSelect.value.trim();
  if (!toolName) {
    logAction("manual.step", false, "Select a tool first.");
    return;
  }

  let args;
  try {
    args = el.argsInput.value.trim() ? JSON.parse(el.argsInput.value) : {};
  } catch (error) {
    logAction(toolName, false, `Arguments are not valid JSON: ${error.message}`);
    return;
  }

  const action = { tool_name: toolName, arguments: args };
  setConnection("stepping");
  try {
    let payload;
    if (state.apiMode === "live" && state.sessionId) {
      try {
        payload = await requestJson(ROUTES.live.step(state.sessionId), { method: "POST", body: { action } });
      } catch (error) {
        if (![404, 405].includes(error.status)) {
          throw error;
        }
        state.apiMode = "openenv";
        payload = await requestJson(ROUTES.openenv.step, { method: "POST", body: { action } });
      }
    } else {
      payload = await requestJson(ROUTES.openenv.step, { method: "POST", body: { action } });
    }

    const envelope = normalizeEnvelope(payload);
    applyEnvelope(envelope);
    const failed = Boolean(envelope.observation?.error || envelope.observation?.action_result?.error);
    logAction(
      toolName,
      !failed,
      envelope.observation?.system_message || (failed ? "Action rejected" : "Action accepted"),
      args,
    );
    setConnection("online");
  } catch (error) {
    logAction(toolName, false, error.message, args);
    setConnection("online");
  }
}

async function refreshSessions() {
  setConnection("scanning");
  try {
    const payload = await requestJson(ROUTES.live.sessions);
    state.liveSessions = Array.isArray(payload) ? payload : payload?.sessions || [];
    state.apiMode = "live";
    logAction("live.scan", true, `${state.liveSessions.length} live session(s) reported.`);
  } catch {
    try {
      const [tasks, suites] = await Promise.all([
        requestJson(`${ROUTES.live.tasks}?limit=5`),
        requestJson(ROUTES.live.suites),
      ]);
      state.liveSessions = [];
      state.apiMode = "live";
      logAction(
        "live.scan",
        true,
        `Catalog OK: ${(tasks?.tasks || []).length} task sample(s), ${(suites?.suites || []).length} suite(s).`,
      );
    } catch {
      state.liveSessions = [];
      state.apiMode = "openenv";
      logAction("live.scan", false, "No live API detected; using OpenEnv singleton.");
    }
  } finally {
    setConnection(state.observation ? "online" : "offline");
    render();
  }
}

async function ensureLiveSession() {
  if (state.apiMode === "live" && state.sessionId) {
    return state.sessionId;
  }
  await startSession();
  if (state.apiMode !== "live" || !state.sessionId) {
    throw new Error("Live API session is required for model runs and replays.");
  }
  return state.sessionId;
}

function buildModelSpec() {
  const policy = el.modelPolicy.value;
  const name = `${policy}-live`;
  const model = el.modelName.value.trim();
  const apiBase = el.modelApiBase.value.trim();
  const apiKey = el.modelApiKey.value.trim();
  const adapterPath = el.modelAdapterPath.value.trim();
  const spec = { name, policy };

  if (model) {
    if (policy === "local_hf" || policy === "vllm") {
      spec.base_model = model;
    } else {
      spec.model = model;
    }
  }
  if (apiBase) spec.api_base_url = apiBase;
  if (apiKey) spec.api_key = apiKey;
  if (adapterPath) spec.adapter_path = adapterPath;
  return spec;
}

function applyModelPreset(policy) {
  const preset = MODEL_PRESETS[policy];
  if (!preset) return;
  el.modelName.value = preset.model;
  el.modelApiBase.value = preset.apiBase;
  if (preset.maxSteps) {
    el.modelMaxSteps.value = preset.maxSteps;
  }
  el.modelAdapterPath.value = "";
  el.modelApiKey.value = "";
}

async function runLiveModel() {
  setConnection("model");
  el.runModelButton.disabled = true;
  try {
    const sessionId = await ensureLiveSession();
    const spec = buildModelSpec();
    const requestedSteps = clamp(Number.parseInt(el.modelMaxSteps.value || "12", 10), 1, 100);
    const basePayload = {
      spec,
      task_id: el.taskInput.value.trim() || null,
      scenario_family: el.familyInput.value.trim() || null,
      seed: Number.parseInt(el.seedInput.value || "0", 10),
      user_instruction: el.objectivePrompt.value.trim() || null,
    };

    logAction("model.run", true, `Starting ${spec.name} for up to ${requestedSteps} step(s).`, {
      policy: spec.policy,
      model: spec.model || spec.base_model || "default",
      max_steps: requestedSteps,
    });

    let emitted = 0;
    let latest = null;
    for (let stepIndex = 1; stepIndex <= requestedSteps; stepIndex += 1) {
      setConnection(`model ${stepIndex}/${requestedSteps}`);
      const result = await requestJson(ROUTES.live.runModel(sessionId), {
        method: "POST",
        body: { ...basePayload, max_steps: 1 },
      });
      latest = result;
      const events = result.events || [];
      emitted += events.length;
      ingestLiveEvents(events, { source: spec.name });
      applyEnvelope(normalizeEnvelope(result));
      await nextFrame();
      if (result.observation?.done || events.some((event) => event.type === "model-error" || event.done)) {
        break;
      }
    }

    logAction("model.run", true, `${spec.name} emitted ${emitted} tool event(s).`, {
      policy: spec.policy,
      model: spec.model || spec.base_model || "default",
      max_steps: requestedSteps,
      final_step: latest?.observation?.metadata?.step_count,
    });
    setConnection("online");
  } catch (error) {
    logAction("model.run", false, error.message);
    setConnection(state.observation ? "online" : "offline");
  } finally {
    el.runModelButton.disabled = false;
  }
}

function nextFrame() {
  return new Promise((resolve) => requestAnimationFrame(() => resolve()));
}

async function replayTrajectoryRecord() {
  setConnection("replay");
  try {
    const sessionId = state.sessionId || "replay-demo";
    const sourcePath = el.replaySourcePath.value.trim();
    if (!sourcePath) {
      throw new Error("Provide a JSON or JSONL source path to replay.");
    }
    const payload = {
      source_path: sourcePath,
      record_index: Number.parseInt(el.replayRecordIndex.value || "0", 10),
    };
    const result = await requestJson(ROUTES.live.replay(sessionId), { method: "POST", body: payload });
    state.apiMode = "live";
    ingestLiveEvents(result.events || [], { source: "replay" });
    applyEnvelope(normalizeEnvelope(result));
    logAction("replay.record", true, `Replayed ${result.actions_replayed || 0} action(s).`, payload);
    setConnection("online");
  } catch (error) {
    logAction("replay.record", false, error.message);
    setConnection(state.observation ? "online" : "offline");
  }
}

function ingestLiveEvents(events, options = {}) {
  for (const event of events || []) {
    if (!event?.action?.tool_name) continue;
    const failed = Boolean(event.parse_error || event.action_result?.error);
    logAction(
      `${options.source ? `${options.source}:` : ""}${event.action.tool_name}`,
      !failed,
      event.message || event.parse_error || event.action_result?.error || event.type || "tool call",
      event.action.arguments || {},
    );
  }
}

async function runComparison() {
  let specs;
  try {
    specs = JSON.parse(el.compareSpecs.value || "[]");
  } catch (error) {
    logAction("compare", false, `Model specs are not valid JSON: ${error.message}`);
    return;
  }
  if (!Array.isArray(specs) || specs.length === 0) {
    logAction("compare", false, "Provide at least one model spec.");
    return;
  }

  const payload = {
    specs,
    seed: Number.parseInt(el.seedInput.value || "0", 10),
    task_id: el.taskInput.value.trim() || null,
    scenario_family: el.familyInput.value.trim() || null,
    user_instruction: el.objectivePrompt.value.trim() || null,
    max_steps: 40,
    include_rollouts: true,
    include_trace_artifacts: false,
  };
  setConnection("comparing");
  try {
    const result = await requestJson(ROUTES.live.compare, { method: "POST", body: payload });
    state.comparison = result;
    renderComparison();
    logAction("compare", true, `Compared ${result.summaries?.length || 0} policy run(s).`, {
      specs: specs.map((spec) => spec.name || spec.policy),
    });
    setConnection("online");
  } catch (error) {
    logAction("compare", false, error.message);
    setConnection(state.observation ? "online" : "offline");
  }
}

function overlayBestComparisonRoute() {
  const summaries = state.comparison?.summaries || [];
  if (!summaries.length) {
    logAction("overlay", false, "Run a comparison first.");
    return;
  }
  const best = [...summaries].sort((a, b) => (b.final_reward || 0) - (a.final_reward || 0))[0];
  const route = (best.rollout?.trajectory || [])
    .map((step) => step.next_observation?.telemetry?.pose)
    .filter(isPose);
  if (!route.length) {
    logAction("overlay", false, `No route found for ${best.name}.`);
    return;
  }
  state.routeTrail = route;
  renderScene();
  logAction("overlay", true, `Overlaying ${best.name} route (${route.length} points).`);
}

/* ===========================================================
 * Apply envelope / state mutations
 * =========================================================== */

function applyEnvelope(envelope) {
  state.observation = envelope.observation;
  state.scene = envelope.scene || envelope.observation?.scene || null;
  state.sessionId = envelope.sessionId || state.sessionId;
  state.lastRaw = envelope.raw;
  rememberPose(envelope.observation, envelope.scene);
  updateToolOptions(envelope.observation);
  render();
}

function rememberPose(observation, scene) {
  const pose = scene?.drone?.pose || observation?.telemetry?.pose;
  if (!isPose(pose)) {
    return;
  }
  const previous = state.routeTrail[state.routeTrail.length - 1];
  if (!previous || distance2(previous, pose) > 0.04) {
    state.routeTrail.push({
      x: pose.x,
      y: pose.y,
      z: pose.z || 0,
      yaw_deg: pose.yaw_deg || 0,
    });
  }
  if (state.routeTrail.length > 240) {
    state.routeTrail = state.routeTrail.slice(-240);
  }
}

function distance2(a, b) {
  return (a.x - b.x) ** 2 + (a.y - b.y) ** 2 + ((a.z || 0) - (b.z || 0)) ** 2;
}

function isPose(value) {
  return Number.isFinite(value?.x) && Number.isFinite(value?.y);
}

/* ===========================================================
 * Tool catalog & quick actions
 * =========================================================== */

function updateToolOptions(observation) {
  const available = observation?.available_tools || [];
  const catalogNames = (observation?.tool_catalog || [])
    .map((tool) => tool.name || tool.tool_name)
    .filter(Boolean);
  const names = [...new Set([...available, ...catalogNames, ...Object.keys(ACTION_PRESETS)])];
  const selected = el.toolSelect.value;
  el.toolSelect.innerHTML = names
    .map((name) => `<option value="${escapeHtml(name)}">${escapeHtml(name)}</option>`)
    .join("");
  el.toolSelect.value = names.includes(selected) ? selected : names[0] || "";
  if (!selected && el.toolSelect.value in ACTION_PRESETS) {
    el.argsInput.value = JSON.stringify(ACTION_PRESETS[el.toolSelect.value], null, 2);
  }
}

function renderQuickActions() {
  const buttons = [
    "takeoff",
    "fly_to_viewpoint",
    "set_gimbal",
    "capture_thermal",
    "capture_rgb",
    "return_home",
    "land",
  ];
  el.quickActions.innerHTML = buttons
    .map(
      (name) =>
        `<button type="button" data-tool="${escapeHtml(name)}">${escapeHtml(
          name.replaceAll("_", " "),
        )}</button>`,
    )
    .join("");
}

function selectPreset(toolName) {
  el.toolSelect.value = toolName;
  el.argsInput.value = JSON.stringify(ACTION_PRESETS[toolName] || {}, null, 2);
}

/* ===========================================================
 * Action log
 * =========================================================== */

function logAction(tool, success, message, args = null) {
  state.actionLog.unshift({ time: new Date(), tool, success, message, args });
  state.actionLog = state.actionLog.slice(0, 80);
  renderLog();
}

function renderLog() {
  el.actionCount.textContent = `${state.actionLog.length} call${state.actionLog.length === 1 ? "" : "s"}`;
  if (!state.actionLog.length) {
    el.actionLog.innerHTML = emptyMarkup("Awaiting first command", "Issue a tool call to begin the trace.");
    return;
  }
  el.actionLog.innerHTML = state.actionLog
    .map((item) => {
      const args = formatArgChips(item.args);
      const status = item.success ? "ok" : "fail";
      const cls = `action-log__entry ${item.success ? "" : "is-failed"}`;
      return `<li class="${cls}">
        <div class="action-log__time">${escapeHtml(formatClock(item.time))}<small>${escapeHtml(formatRelative(item.time))}</small></div>
        <div class="action-log__bullet"></div>
        <div class="action-log__body">
          <div class="action-log__head">
            <span class="action-log__tool">${escapeHtml(item.tool)}</span>
            <span class="action-log__status">${status}</span>
          </div>
          ${args ? `<div class="action-log__args">${args}</div>` : ""}
          ${item.message ? `<div class="action-log__message">${escapeHtml(item.message)}</div>` : ""}
        </div>
      </li>`;
    })
    .join("");
}

function formatArgChips(args) {
  if (!args || typeof args !== "object" || Array.isArray(args)) {
    return Array.isArray(args) ? `<span class="arg-chip">${escapeHtml(JSON.stringify(args))}</span>` : "";
  }
  const entries = Object.entries(args);
  if (!entries.length) return "";
  return entries
    .map(([key, value]) => {
      const formatted = typeof value === "object" ? JSON.stringify(value) : String(value);
      return `<span class="arg-chip"><strong>${escapeHtml(key)}</strong>${escapeHtml(formatted)}</span>`;
    })
    .join("");
}

function formatClock(date) {
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false });
}

function formatRelative(date) {
  const diffSec = Math.max(0, Math.round((Date.now() - date.getTime()) / 1000));
  if (diffSec < 60) return `${diffSec}s ago`;
  if (diffSec < 3600) return `${Math.round(diffSec / 60)}m ago`;
  return `${Math.round(diffSec / 3600)}h ago`;
}

/* ===========================================================
 * Comparison cards
 * =========================================================== */

function renderComparison() {
  const summaries = state.comparison?.summaries || [];
  if (!el.compareResults) return;
  if (!summaries.length) {
    el.compareResults.innerHTML = emptyMarkup("No comparison yet", "Specs will replay against the same task & seed.");
    return;
  }
  el.compareResults.innerHTML = summaries
    .map((summary) => {
      const cls = `compare-card ${summary.success ? "is-success" : ""}`;
      return `<article class="${cls}">
        <div class="compare-card__head">
          <p class="compare-card__title">${escapeHtml(summary.name)}</p>
          <span class="compare-card__meta">${escapeHtml(summary.policy)} · ${summary.success ? "success" : "incomplete"}</span>
        </div>
        <dl>
          <div><dt>reward</dt><dd>${formatNumber(summary.final_reward, 3)}</dd></div>
          <div><dt>steps</dt><dd>${summary.steps ?? "—"}</dd></div>
          <div><dt>captures</dt><dd>${summary.captures?.length || 0}</dd></div>
          <div><dt>parse</dt><dd>${summary.parse_errors?.length || 0}</dd></div>
        </dl>
        <p class="compare-card__seq">${escapeHtml((summary.action_sequence || []).slice(0, 8).join(" → ")) || "—"}</p>
      </article>`;
    })
    .join("");
}

/* ===========================================================
 * Top-level render
 * =========================================================== */

function render() {
  el.apiMode.textContent = `API · ${state.apiMode}`;
  el.sessionId.textContent = `session · ${state.sessionId || "none"}`;
  renderMissionTitle();
  renderBriefing();
  renderPhaseStrip();
  renderInstruments();
  renderRewardComposition();
  renderHud();
  renderCaptures();
  renderLog();
  renderComparison();
  renderScene();
}

function renderMissionTitle() {
  const title = missionTitle(state.observation);
  if (!title) {
    el.missionTitle.textContent = "Awaiting brief";
    el.missionTitle.classList.add("status-pill--ghost");
    return;
  }
  el.missionTitle.textContent = title;
  el.missionTitle.classList.add("status-pill--ghost");
}

/* ===========================================================
 * Mission briefing
 * =========================================================== */

function renderBriefing() {
  const obs = state.observation || {};
  const scene = state.scene || obs.scene || {};
  const checklist = scene.checklist || obs.inspection_affordances || {};

  const instruction =
    obs.mission?.instruction ||
    checklist.instruction ||
    obs.system_message ||
    "Reset a session to load mission objectives.";
  el.missionInstruction.textContent = instruction;

  const meta = obs.metadata || {};
  const taskId = meta.task_id || obs.mission?.task_id || obs.mission?.mission_id;
  const family = meta.scenario_family || "—";
  const seed = meta.scenario_seed ?? "—";
  const stepCount = meta.step_count ?? scene.metadata?.step_count ?? 0;
  const remaining = obs.state_summary?.remaining_steps ?? "—";

  const requiredRows = checklist.required_rows || obs.mission?.required_rows || [];
  const covered = checklist.thermal_rows_covered || obs.checklist_status?.thermal_rows_covered || [];
  const anomalies = checklist.anomalies_detected || obs.checklist_status?.anomalies_detected || [];

  const facts = [
    ["Task", taskId || "—"],
    ["Scenario", family],
    ["Seed", seed],
    ["Phase", checklist.mission_phase || obs.inspection_affordances?.mission_phase || "preflight"],
    ["Step", `${stepCount} · ${remaining} left`],
    ["Rows", `${covered.length}/${requiredRows.length || "—"}`],
    ["Anomalies", anomalies.length],
    ["Status", obs.done ? (checklist.complete ? "complete" : "ended") : "active"],
  ];
  el.briefingFacts.innerHTML = facts
    .map(
      ([label, value]) =>
        `<div><dt>${escapeHtml(label)}</dt><dd>${escapeHtml(String(value))}</dd></div>`,
    )
    .join("");

  const suggested = checklist.suggested_tools || obs.inspection_affordances?.suggested_tools || [];
  el.suggestedTools.innerHTML = suggested
    .slice(0, 6)
    .map((tool) => `<li>${escapeHtml(tool)}</li>`)
    .join("");

  const blockers = [
    ...(checklist.blockers || obs.inspection_affordances?.blockers || []),
    ...(obs.warnings || scene.warnings || []),
  ];
  el.blockersList.innerHTML = [...new Set(blockers)]
    .slice(0, 6)
    .map((blocker) => `<li>${escapeHtml(blocker)}</li>`)
    .join("");
}

/* ===========================================================
 * Phase strip
 * =========================================================== */

function renderPhaseStrip() {
  const obs = state.observation || {};
  const scene = state.scene || obs.scene || {};
  const phase =
    scene.checklist?.mission_phase ||
    obs.inspection_affordances?.mission_phase ||
    "preflight";
  const activeIndex = MISSION_PHASES.findIndex((step) => step.key === phase);
  el.phaseStrip.innerHTML = MISSION_PHASES.map((step, index) => {
    const cls =
      index < activeIndex ? "is-done" : index === activeIndex ? "is-active" : "";
    return `<div class="phase-strip__step ${cls}">
      <span class="num">0${index + 1}</span>
      <span class="name">${escapeHtml(step.label)}</span>
    </div>`;
  }).join("");
}

/* ===========================================================
 * Telemetry instruments
 * =========================================================== */

function renderInstruments() {
  const telemetry = scenarioTelemetry();
  const battery = telemetry.battery_pct;
  const reserve = telemetry.reserve_pct;
  const altitude = telemetry.altitude_m;
  const speed = telemetry.ground_speed_mps;
  const distance = telemetry.distance_flown_m;
  const elapsed = telemetry.elapsed_time_s;

  // Battery ring
  const ringCircumference = 2 * Math.PI * 58;
  const fillPct = Number.isFinite(battery) ? clamp(battery, 0, 100) : 0;
  const offset = ringCircumference * (1 - fillPct / 100);
  el.batteryRingFill.setAttribute("stroke-dasharray", String(ringCircumference));
  el.batteryRingFill.setAttribute("stroke-dashoffset", String(offset));
  el.batteryRingFill.classList.toggle("is-low", fillPct < 50 && fillPct >= 20);
  el.batteryRingFill.classList.toggle("is-critical", fillPct < 20);
  el.batteryHeadline.textContent = Number.isFinite(battery) ? `${formatNumber(battery, 0)}%` : "—%";
  el.batteryReserve.textContent = Number.isFinite(reserve) ? `reserve ${formatNumber(reserve, 0)}%` : "reserve —%";

  if (Number.isFinite(reserve)) {
    const angle = (reserve / 100) * 360 - 90;
    const rad = (angle * Math.PI) / 180;
    const cx = 70;
    const cy = 70;
    const r1 = 50;
    const r2 = 64;
    const x1 = cx + Math.cos(rad) * r1;
    const y1 = cy + Math.sin(rad) * r1;
    const x2 = cx + Math.cos(rad) * r2;
    const y2 = cy + Math.sin(rad) * r2;
    el.batteryReserveTick.setAttribute("x1", x1);
    el.batteryReserveTick.setAttribute("y1", y1);
    el.batteryReserveTick.setAttribute("x2", x2);
    el.batteryReserveTick.setAttribute("y2", y2);
  }

  // Altitude tape
  const altMax = 60;
  const altPct = Number.isFinite(altitude) ? clamp(altitude / altMax, 0, 1) : 0;
  el.altMarker.style.bottom = `${altPct * 100}%`;
  el.altMarker.setAttribute("data-label", Number.isFinite(altitude) ? `${formatNumber(altitude, 0)} m` : "— m");
  el.altitudeHeadline.textContent = Number.isFinite(altitude) ? `${formatNumber(altitude, 1)} m` : "— m";
  el.altitudeBand.textContent =
    altitude >= 35 ? "high cruise" : altitude >= 15 ? "survey band" : altitude > 0 ? "low approach" : "ground";

  // Flight micro
  el.speedHeadline.textContent = Number.isFinite(speed) ? `${formatNumber(speed, 1)} m/s` : "— m/s";
  el.speedSub.textContent = Number.isFinite(distance) ? `distance ${formatNumber(distance, 0)} m` : "distance — m";
  el.flightMicro.innerHTML = [
    ["Air spd", Number.isFinite(telemetry.air_speed_mps) ? `${formatNumber(telemetry.air_speed_mps, 1)} m/s` : "—"],
    ["Wx", telemetry.weather_band || "—"],
    ["GPS", `${telemetry.satellites_visible ?? "—"} sat`],
    ["Link", Number.isFinite(telemetry.link_rssi_pct) ? `${formatNumber(telemetry.link_rssi_pct, 0)}%` : "—"],
    ["Mode", telemetry.mode || "—"],
    ["Storage", `${telemetry.storage_remaining ?? "—"}`],
  ]
    .map(([label, value]) => `<li><span>${escapeHtml(label)}</span><strong>${escapeHtml(String(value))}</strong></li>`)
    .join("");

  // Mission clock
  if (state.sessionStartedAt) {
    const total = Math.max(0, Math.floor((Date.now() - state.sessionStartedAt) / 1000));
    const mins = String(Math.floor(total / 60)).padStart(2, "0");
    const secs = String(total % 60).padStart(2, "0");
    el.missionClock.textContent = `T+ ${mins}:${secs}`;
  } else {
    el.missionClock.textContent = "T+ 00:00";
  }
}

function scenarioTelemetry() {
  const obs = state.observation || {};
  const scene = state.scene || {};
  const sceneTele = scene.telemetry || {};
  const sceneDrone = scene.drone || {};
  const obsTele = obs.telemetry || {};

  return {
    battery_pct: pickFinite(sceneTele.battery_pct, obsTele.battery?.level_pct, sceneDrone.battery_pct),
    reserve_pct: pickFinite(sceneTele.reserve_pct, obsTele.battery?.reserve_pct, obs.mission?.min_battery_at_done_pct),
    altitude_m: pickFinite(sceneDrone.pose?.z, obsTele.pose?.z, sceneTele.altitude_m),
    ground_speed_mps: pickFinite(sceneTele.ground_speed_mps, obsTele.velocity?.ground_speed_mps),
    air_speed_mps: pickFinite(sceneTele.air_speed_mps, obsTele.velocity?.air_speed_mps),
    distance_flown_m: pickFinite(sceneTele.distance_flown_m, obsTele.distance_flown_m),
    elapsed_time_s: pickFinite(sceneTele.elapsed_time_s, obsTele.elapsed_time_s),
    weather_band: sceneTele.weather_band || obsTele.weather_band || null,
    gps_fix_type: sceneTele.gps_fix_type ?? obsTele.gps?.fix_type,
    satellites_visible: sceneTele.satellites_visible ?? obsTele.gps?.satellites_visible,
    link_rssi_pct: pickFinite(sceneTele.link_rssi_pct, obsTele.link?.rssi_pct),
    storage_remaining: sceneTele.storage_remaining ?? obsTele.camera?.storage_remaining,
    mode: sceneDrone.mode || obsTele.autopilot?.mode || null,
    camera_source: sceneDrone.camera_source || obsTele.camera?.active_source || null,
  };
}

function pickFinite(...values) {
  for (const value of values) {
    const num = Number(value);
    if (Number.isFinite(num)) return num;
  }
  return null;
}

/* ===========================================================
 * Reward composition
 * =========================================================== */

function renderRewardComposition() {
  const breakdown =
    state.scene?.reward ||
    state.observation?.reward_breakdown ||
    state.observation?.scene?.reward ||
    {};
  const total = Number(breakdown.total ?? state.observation?.reward ?? 0);

  const parts = REWARD_COMPONENTS.map((entry) => ({
    ...entry,
    value: clamp(Math.abs(Number(breakdown[entry.key] ?? 0)), 0, 1),
    raw: Number(breakdown[entry.key] ?? 0),
  }));
  const sum = parts.reduce((acc, part) => acc + part.value, 0) || 1;
  el.rewardComposition.innerHTML = parts
    .map((part) => {
      const width = (part.value / sum) * 100;
      return `<span style="background:${part.color};width:${width}%" title="${escapeHtml(part.label)} ${formatNumber(part.raw, 3)}"></span>`;
    })
    .join("");

  el.rewardHeadline.textContent = formatNumber(total, 3);
  const terminal = breakdown.debug?.terminal_submitted || state.observation?.done;
  el.rewardMode.textContent = terminal ? "terminal" : "non-terminal";

  const microRows = [
    ...REWARD_COMPONENTS.map((part) => [part.label, formatNumber(breakdown[part.key], 2)]),
    ...REWARD_MICRO.map((key) => [key.replaceAll("_", " "), formatNumber(breakdown[key], 2)]),
    ["safety", formatNumber(breakdown.safety_gate, 2)],
    ["integrity", formatNumber(breakdown.integrity_gate, 2)],
    ["penalty", formatNumber(breakdown.penalties, 2)],
  ];
  el.rewardMicro.innerHTML = microRows
    .map(([label, value]) => `<li><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></li>`)
    .join("");
}

/* ===========================================================
 * HUD overlays
 * =========================================================== */

function renderHud() {
  const tele = scenarioTelemetry();
  const obs = state.observation || {};
  const scene = state.scene || {};
  const pose = scene.drone?.pose || obs.telemetry?.pose || { x: 0, y: 0, z: 0, yaw_deg: 0 };
  const gimbal = scene.drone?.gimbal || obs.telemetry?.gimbal;

  el.hudPose.textContent = `x ${formatNumber(pose.x, 1)} · y ${formatNumber(pose.y, 1)} · z ${formatNumber(pose.z, 1)}`;
  el.hudHeading.textContent = `hdg ${formatNumber(pose.yaw_deg, 0)}°`;
  el.hudSensor.textContent = (tele.camera_source || "—").toUpperCase();
  el.hudGimbal.textContent = gimbal
    ? `gimbal pitch ${formatNumber(gimbal.pitch_deg, 0)}° · yaw ${formatNumber(gimbal.yaw_deg, 0)}°`
    : "gimbal — / —";

  const yawRad = (Number(pose.yaw_deg || 0) * Math.PI) / 180;
  el.compassNeedle.style.transform = `translate(-50%, -100%) rotate(${formatNumber(yawRad * 180 / Math.PI, 0)}deg)`;
}

/* ===========================================================
 * Scene rendering — polished isometric tactical view
 * =========================================================== */

function renderScene() {
  resizeCanvas();
  const width = el.canvas.clientWidth;
  const height = el.canvas.clientHeight;
  ctx.clearRect(0, 0, width, height);

  drawBackground(width, height);

  const obs = state.observation;
  if (!obs) {
    drawEmptyScene(width, height);
    return;
  }

  const scene = sceneFromObservation();
  const projector = makeProjector(scene, width, height);
  state.projector = projector;
  drawIsoGrid(projector, width, height);
  drawZones(projector, scene.zones);
  drawAssets(projector, scene.assets, scene);
  drawViewpoints(projector, scene.viewpoints, scene.assets);
  drawCapturePoints(projector, scene.captures);
  drawRoute(projector, scene.routeTrail);
  drawHome(projector, scene.home);
  drawDrone(projector, scene.pose, scene, obs);
  updateScale(projector);
  updateZoomReadout();

  const failure = obs.error || obs.action_result?.error || "";
  el.failureBanner.hidden = !failure;
  el.failureBanner.textContent = failure ? `Rejected — ${failure}` : "";
}

function updateZoomReadout() {
  if (!el.zoomLevel) return;
  el.zoomLevel.textContent = `${Math.round(state.camera.zoom * 100)}%`;
}

function sceneFromObservation() {
  const obs = state.observation || {};
  const scene = state.scene || obs.scene || {};
  const fallbackMap = obs.site_map || {};

  const assets = scene.assets?.length ? scene.assets : obs.visible_assets?.length ? obs.visible_assets : fallbackMap.assets || [];
  const zones = scene.airspace_zones?.length
    ? scene.airspace_zones
    : fallbackMap.airspace_zones || obs.zones || [];
  const viewpoints = scene.viewpoints?.length
    ? scene.viewpoints
    : fallbackMap.viewpoints || obs.viewpoints || [];
  const capturesSource =
    scene.capture_points?.length ? scene.capture_points :
    obs.capture_log?.length ? obs.capture_log :
    obs.evidence_artifacts || [];

  const home =
    scene.home_pad?.pose ||
    fallbackMap.home_pad?.pose ||
    fallbackMap.home ||
    { x: 0, y: 0, z: 0 };

  const pose =
    scene.drone?.pose ||
    obs.telemetry?.pose ||
    fallbackMap.drone?.pose ||
    state.routeTrail[state.routeTrail.length - 1] ||
    home;

  const routeFromScene = (scene.route_history || []).map((entry) => entry.pose || entry).filter(isPose);
  const routeTrail = routeFromScene.length ? routeFromScene : state.routeTrail;

  const checklist = scene.checklist || {};
  return {
    home,
    pose,
    drone: scene.drone || null,
    assets,
    zones,
    viewpoints,
    captures: capturesSource,
    routeTrail,
    coveredAssetIds: new Set([
      ...(checklist.thermal_rows_covered || []),
      ...(obs.checklist_status?.thermal_rows_covered || []),
      ...(obs.checklist_status?.targets_acknowledged || []),
    ]),
    pendingAssetIds: new Set(
      checklist.pending_asset_ids || obs.inspection_affordances?.pending_asset_ids || [],
    ),
    anomalyTargets: new Set(
      Object.values(checklist.anomaly_targets || obs.checklist_status?.anomaly_targets || {}),
    ),
    sensor: scene.drone?.camera_source || obs.telemetry?.camera?.active_source || "rgb",
    gimbal: scene.drone?.gimbal || obs.telemetry?.gimbal || null,
  };
}

function makeProjector(scene, width, height) {
  const cos = Math.cos(Math.PI / 6);
  const sin = Math.sin(Math.PI / 6);

  let baseParams;
  if (state.camera.lock) {
    baseParams = state.camera.lock;
  } else {
    baseParams = computeAutoFitParams(scene, width, height, cos, sin);
  }

  const scale = baseParams.baseScale * state.camera.zoom;
  const zScale = scale * 0.85;
  const cx = baseParams.baseCx + state.camera.panX;
  const cy = baseParams.baseCy + state.camera.panY;
  const centerX = baseParams.centerX;
  const centerY = baseParams.centerY;

  return {
    minX: baseParams.minX,
    maxX: baseParams.maxX,
    minY: baseParams.minY,
    maxY: baseParams.maxY,
    scale,
    zScale,
    centerX,
    centerY,
    cx,
    cy,
    cos,
    sin,
    project: (pose) => {
      const x = (Number(pose?.x) || 0) - centerX;
      const y = (Number(pose?.y) || 0) - centerY;
      const z = Number(pose?.z) || 0;
      return {
        x: cx + (x - y) * cos * scale,
        y: cy + (x + y) * sin * scale - z * zScale,
      };
    },
    projectGround: (pose) => {
      const x = (Number(pose?.x) || 0) - centerX;
      const y = (Number(pose?.y) || 0) - centerY;
      return {
        x: cx + (x - y) * cos * scale,
        y: cy + (x + y) * sin * scale,
      };
    },
    unproject: (sx, sy, z = 0) => {
      // Inverse of the iso projection at altitude z.
      const a = (sx - cx) / (cos * scale);
      const b = (sy - cy + z * zScale) / (sin * scale);
      const xRel = (a + b) / 2;
      const yRel = (b - a) / 2;
      return { x: xRel + centerX, y: yRel + centerY };
    },
  };
}

function computeAutoFitParams(scene, width, height, cos, sin) {
  const points = [scene.home, scene.pose].filter(isPose).map((pose) => ({ x: pose.x, y: pose.y }));
  for (const asset of scene.assets) {
    const g = assetGeometry(asset);
    const w = g.width_m;
    const h = g.height_m;
    points.push({ x: g.center_x - w / 2 - 4, y: g.center_y - h / 2 - 4 });
    points.push({ x: g.center_x + w / 2 + 4, y: g.center_y + h / 2 + 4 });
  }
  for (const zone of scene.zones) {
    for (const point of zonePolygon(zone)) {
      points.push(point);
    }
  }
  for (const viewpoint of scene.viewpoints) {
    const pose = viewpoint.pose || viewpoint;
    if (isPose(pose)) points.push({ x: pose.x, y: pose.y });
  }
  for (const capture of scene.captures) {
    const pose = capture.pose || capture;
    if (isPose(pose)) points.push({ x: pose.x, y: pose.y });
  }
  for (const pose of scene.routeTrail) {
    if (isPose(pose)) points.push({ x: pose.x, y: pose.y });
  }

  if (points.length === 0) {
    points.push({ x: 0, y: 0 }, { x: 40, y: 30 });
  }

  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);
  const minX = Math.min(...xs) - 6;
  const maxX = Math.max(...xs) + 6;
  const minY = Math.min(...ys) - 6;
  const maxY = Math.max(...ys) + 6;
  const centerX = (minX + maxX) / 2;
  const centerY = (minY + maxY) / 2;

  const spanX = maxX - minX;
  const spanY = maxY - minY;
  const projectedWidth = (spanX + spanY) * cos;
  const projectedHeight = (spanX + spanY) * sin + 60;

  // Slightly tighter auto-fit so default labels & drone read comfortably.
  const padding = 0.74;
  const baseScale = Math.min(
    (width * padding) / Math.max(projectedWidth, 40),
    (height * padding) / Math.max(projectedHeight, 30),
  );

  return {
    minX,
    maxX,
    minY,
    maxY,
    centerX,
    centerY,
    baseScale,
    baseCx: width / 2,
    baseCy: height * 0.56,
  };
}

function resizeCanvas() {
  const ratio = window.devicePixelRatio || 1;
  const width = Math.max(320, Math.floor(el.canvas.clientWidth));
  const height = Math.max(420, Math.floor(el.canvas.clientHeight));
  const targetWidth = Math.floor(width * ratio);
  const targetHeight = Math.floor(height * ratio);
  if (el.canvas.width !== targetWidth || el.canvas.height !== targetHeight) {
    el.canvas.width = targetWidth;
    el.canvas.height = targetHeight;
  }
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
}

function drawBackground(width, height) {
  // Layered gradient stage with subtle horizon band
  const sky = ctx.createLinearGradient(0, 0, 0, height);
  sky.addColorStop(0, "#0e1a18");
  sky.addColorStop(0.35, "#0a1413");
  sky.addColorStop(1, "#06100e");
  ctx.fillStyle = sky;
  ctx.fillRect(0, 0, width, height);

  // Warm sun glow at top-left to match overall theme
  const glow = ctx.createRadialGradient(width * 0.18, height * 0.05, 10, width * 0.18, height * 0.05, height * 0.65);
  glow.addColorStop(0, "rgba(244, 184, 96, 0.18)");
  glow.addColorStop(1, "rgba(244, 184, 96, 0)");
  ctx.fillStyle = glow;
  ctx.fillRect(0, 0, width, height);

  // Subtle horizon ground
  ctx.fillStyle = "rgba(244, 236, 220, 0.015)";
  ctx.fillRect(0, height * 0.55, width, height * 0.45);
}

function drawEmptyScene(width, height) {
  ctx.save();
  ctx.translate(width / 2, height * 0.52);

  // Concentric rings
  ctx.strokeStyle = "rgba(244, 184, 96, 0.32)";
  ctx.lineWidth = 1.2;
  for (let radius = 60; radius < 240; radius += 36) {
    ctx.beginPath();
    ctx.ellipse(0, 0, radius * 1.6, radius * 0.7, 0, 0, Math.PI * 2);
    ctx.stroke();
  }

  // Axes
  ctx.strokeStyle = "rgba(244, 236, 220, 0.18)";
  ctx.beginPath();
  ctx.moveTo(-280, 0);
  ctx.lineTo(280, 0);
  ctx.moveTo(0, -90);
  ctx.lineTo(0, 100);
  ctx.stroke();

  // Label
  ctx.fillStyle = "rgba(244, 236, 220, 0.65)";
  ctx.font = "italic 28px 'Instrument Serif', serif";
  ctx.textAlign = "center";
  ctx.fillText("Awaiting reset", 0, -16);

  ctx.font = "10px 'JetBrains Mono', monospace";
  ctx.fillStyle = "rgba(244, 236, 220, 0.4)";
  ctx.fillText("ISSUE  RESET  TO  LOAD  SCENE", 0, 12);
  ctx.restore();
}

function drawIsoGrid(projector, width, height) {
  ctx.save();
  ctx.lineWidth = 1;
  const step = 10;
  const padding = step * 2;

  for (let x = Math.floor((projector.minX - padding) / step) * step; x <= projector.maxX + padding; x += step) {
    const major = Math.abs(x) % 50 === 0;
    ctx.strokeStyle = major
      ? "rgba(244, 236, 220, 0.07)"
      : "rgba(244, 236, 220, 0.03)";
    const a = projector.projectGround({ x, y: projector.minY - padding });
    const b = projector.projectGround({ x, y: projector.maxY + padding });
    line(a, b);
  }
  for (let y = Math.floor((projector.minY - padding) / step) * step; y <= projector.maxY + padding; y += step) {
    const major = Math.abs(y) % 50 === 0;
    ctx.strokeStyle = major
      ? "rgba(244, 236, 220, 0.07)"
      : "rgba(244, 236, 220, 0.03)";
    const a = projector.projectGround({ x: projector.minX - padding, y });
    const b = projector.projectGround({ x: projector.maxX + padding, y });
    line(a, b);
  }

  // Origin axes
  ctx.strokeStyle = "rgba(244, 184, 96, 0.16)";
  ctx.lineWidth = 1.2;
  const ox = projector.projectGround({ x: 0, y: projector.minY - padding });
  const oxe = projector.projectGround({ x: 0, y: projector.maxY + padding });
  line(ox, oxe);
  const oy = projector.projectGround({ x: projector.minX - padding, y: 0 });
  const oye = projector.projectGround({ x: projector.maxX + padding, y: 0 });
  line(oy, oye);

  ctx.restore();
}

function drawZones(projector, zones) {
  for (const zone of zones) {
    const polygonPoints = zonePolygon(zone).map(projector.projectGround);
    if (!polygonPoints.length) continue;
    const restricted = (zone.zone_type || "").includes("restricted") || (zone.constraint_level || "") === "no_fly";
    const stroke = restricted ? "rgba(255, 138, 107, 0.78)" : "rgba(244, 184, 96, 0.7)";
    const fill = restricted ? "rgba(255, 138, 107, 0.12)" : "rgba(244, 184, 96, 0.1)";

    ctx.save();
    ctx.lineWidth = 1.4;
    ctx.fillStyle = fill;
    ctx.strokeStyle = stroke;
    polygon(polygonPoints, true);
    ctx.stroke();
    hatchInside(polygonPoints, restricted ? "rgba(255, 138, 107, 0.18)" : "rgba(244, 184, 96, 0.12)");
    labelAt(zone.label || zone.zone_id || "zone", centroid(polygonPoints), restricted ? PALETTE.coral : PALETTE.amber);
    ctx.restore();
  }
}

function drawAssets(projector, assets, scene) {
  // Sort by depth (further = drawn first)
  const sorted = [...assets].sort((a, b) => {
    const ga = assetGeometry(a);
    const gb = assetGeometry(b);
    return (ga.center_x + ga.center_y) - (gb.center_x + gb.center_y);
  });

  for (const asset of sorted) {
    const g = assetGeometry(asset);
    const isCovered = scene.coveredAssetIds.has(asset.asset_id);
    const isPending = scene.pendingAssetIds.has(asset.asset_id);
    const isAnomaly = scene.anomalyTargets.has(asset.asset_id);

    const accent = isAnomaly
      ? PALETTE.coral
      : isCovered
        ? PALETTE.lime
        : isPending
          ? PALETTE.amber
          : PALETTE.sage;
    const fill = isAnomaly
      ? "rgba(255, 138, 107, 0.22)"
      : isCovered
        ? "rgba(184, 232, 168, 0.2)"
        : isPending
          ? "rgba(244, 184, 96, 0.16)"
          : "rgba(138, 214, 193, 0.15)";

    drawExtrudedBox(projector, g, fill, accent);

    // Coverage ring
    if (isCovered) {
      drawCoverageRing(projector, g, accent);
    }

    // Asset label on top
    const top = projector.project({ x: g.center_x, y: g.center_y, z: (g.center_z || 0) + 1.5 });
    drawLabel(asset.asset_id || asset.label || "asset", top, accent);
  }
}

function drawExtrudedBox(projector, g, fill, accent) {
  const halfW = g.width_m / 2;
  const halfH = g.height_m / 2;
  const baseZ = 0;
  const topZ = (g.center_z || 0) + 0.6; // tilt suggestion

  const cornersGround = [
    { x: g.center_x - halfW, y: g.center_y - halfH, z: baseZ },
    { x: g.center_x + halfW, y: g.center_y - halfH, z: baseZ },
    { x: g.center_x + halfW, y: g.center_y + halfH, z: baseZ },
    { x: g.center_x - halfW, y: g.center_y + halfH, z: baseZ },
  ];
  const cornersTop = cornersGround.map((corner) => ({ ...corner, z: topZ }));

  // Drop shadow
  ctx.save();
  ctx.fillStyle = "rgba(0, 0, 0, 0.32)";
  const shadow = cornersGround.map(projector.projectGround).map((point) => ({ x: point.x + 6, y: point.y + 8 }));
  polygon(shadow, true);
  ctx.restore();

  // Side faces (gradient based on accent)
  ctx.save();
  ctx.lineWidth = 1.1;
  const sideAlphaFill = "rgba(8, 14, 13, 0.55)";
  const sides = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 0],
  ];
  for (const [a, b] of sides) {
    const ag = projector.projectGround(cornersGround[a]);
    const bg = projector.projectGround(cornersGround[b]);
    const at = projector.project(cornersTop[a]);
    const bt = projector.project(cornersTop[b]);

    const grad = ctx.createLinearGradient(ag.x, ag.y, at.x, at.y);
    grad.addColorStop(0, "rgba(8, 14, 13, 0.85)");
    grad.addColorStop(1, sideAlphaFill);
    ctx.fillStyle = grad;
    ctx.strokeStyle = hexToRgba(accent, 0.42);
    ctx.beginPath();
    ctx.moveTo(ag.x, ag.y);
    ctx.lineTo(bg.x, bg.y);
    ctx.lineTo(bt.x, bt.y);
    ctx.lineTo(at.x, at.y);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
  }
  ctx.restore();

  // Top face — subtle stripes for solar panel feel
  const topPoints = cornersTop.map(projector.project);
  ctx.save();
  ctx.fillStyle = fill;
  ctx.strokeStyle = accent;
  ctx.lineWidth = 1.6;
  polygon(topPoints, true);
  ctx.stroke();
  drawPanelStripes(projector, g, topZ, accent);
  ctx.restore();
}

function drawPanelStripes(projector, g, topZ, accent) {
  const stripes = 6;
  ctx.save();
  ctx.strokeStyle = hexToRgba(accent, 0.42);
  ctx.lineWidth = 1.1;
  for (let index = 1; index < stripes; index += 1) {
    const x = g.center_x - g.width_m / 2 + (g.width_m / stripes) * index;
    const a = projector.project({ x, y: g.center_y - g.height_m / 2, z: topZ });
    const b = projector.project({ x, y: g.center_y + g.height_m / 2, z: topZ });
    line(a, b);
  }
  ctx.restore();
}

function drawCoverageRing(projector, g, accent) {
  const center = projector.projectGround({ x: g.center_x, y: g.center_y });
  const radiusX = (Math.max(g.width_m, g.height_m) / 2 + 4.5) * Math.cos(Math.PI / 6) * projector.scale;
  const radiusY = (Math.max(g.width_m, g.height_m) / 2 + 4.5) * Math.sin(Math.PI / 6) * projector.scale;
  ctx.save();
  ctx.strokeStyle = hexToRgba(accent, 0.7);
  ctx.lineWidth = 1.6;
  ctx.setLineDash([4, 6]);
  ctx.beginPath();
  ctx.ellipse(center.x, center.y, radiusX, radiusY, 0, 0, Math.PI * 2);
  ctx.stroke();
  ctx.restore();
}

function drawViewpoints(projector, viewpoints, assets) {
  const lookup = new Map(assets.map((asset) => [asset.asset_id, assetGeometry(asset)]));
  for (const viewpoint of viewpoints) {
    const pose = viewpoint.pose || viewpoint;
    if (!isPose(pose)) continue;
    const air = projector.project(pose);
    const ground = projector.projectGround(pose);

    ctx.save();
    // Altitude pole
    ctx.strokeStyle = "rgba(244, 184, 96, 0.35)";
    ctx.lineWidth = 1;
    ctx.setLineDash([2, 3]);
    line(ground, air);
    ctx.setLineDash([]);

    // Marker
    ctx.fillStyle = "rgba(244, 184, 96, 0.2)";
    ctx.strokeStyle = PALETTE.amber;
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    ctx.moveTo(air.x, air.y - 10);
    ctx.lineTo(air.x + 9, air.y);
    ctx.lineTo(air.x, air.y + 10);
    ctx.lineTo(air.x - 9, air.y);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();

    // Sight lines
    ctx.strokeStyle = "rgba(244, 184, 96, 0.18)";
    ctx.lineWidth = 0.9;
    for (const assetId of viewpoint.asset_ids || []) {
      const target = lookup.get(assetId);
      if (!target) continue;
      const end = projector.project({
        x: target.center_x,
        y: target.center_y,
        z: (target.center_z || 0) + 1,
      });
      line(air, end);
    }
    ctx.restore();
  }
}

function drawCapturePoints(projector, captures) {
  for (const capture of captures) {
    const pose = capture.pose || capture;
    if (!isPose(pose)) continue;
    const air = projector.project(pose);
    const ground = projector.projectGround(pose);
    const sensor = capture.sensor || capture.camera_source || "rgb";
    const accent = sensor === "thermal" ? PALETTE.coral : PALETTE.sage;

    ctx.save();
    // Pole to ground
    ctx.strokeStyle = hexToRgba(accent, 0.45);
    ctx.setLineDash([2, 3]);
    line(ground, air);
    ctx.setLineDash([]);

    // Frustum cone (subtle)
    if (capture.frustum) {
      drawFrustum(projector, capture.frustum, accent);
    }

    // Marker
    ctx.fillStyle = hexToRgba(accent, 0.95);
    ctx.strokeStyle = "rgba(11, 19, 17, 0.9)";
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    ctx.arc(air.x, air.y, 6, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    // Halo
    ctx.strokeStyle = hexToRgba(accent, 0.45);
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.arc(air.x, air.y, 12, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();
  }
}

function drawFrustum(projector, frustum, accent) {
  const origin = projector.project(frustum.origin);
  const yaw = (Number(frustum.yaw_deg || 0) * Math.PI) / 180;
  const pitch = (Number(frustum.pitch_deg || 0) * Math.PI) / 180;
  const range = Number(frustum.range_m || 30);
  const hfov = (Number(frustum.hfov_deg || 60) * Math.PI) / 180;

  // Project the central forward vector to the ground in iso projection.
  const cos = Math.cos(Math.PI / 6);
  const sin = Math.sin(Math.PI / 6);
  const dxScreen = (Math.cos(yaw) - Math.sin(yaw)) * cos * projector.scale;
  const dyScreen = (Math.cos(yaw) + Math.sin(yaw)) * sin * projector.scale;
  const verticalDrop = -Math.sin(pitch) * range * projector.zScale;

  const tipX = origin.x + Math.cos(yaw) * range * cos * projector.scale - Math.sin(yaw) * range * cos * projector.scale;
  const tipY =
    origin.y +
    (Math.cos(yaw) + Math.sin(yaw)) * range * sin * projector.scale +
    verticalDrop;

  const halfHfov = hfov / 2;
  const leftYaw = yaw - halfHfov;
  const rightYaw = yaw + halfHfov;
  const leftTipX = origin.x + (Math.cos(leftYaw) - Math.sin(leftYaw)) * range * cos * projector.scale;
  const leftTipY = origin.y + (Math.cos(leftYaw) + Math.sin(leftYaw)) * range * sin * projector.scale + verticalDrop;
  const rightTipX = origin.x + (Math.cos(rightYaw) - Math.sin(rightYaw)) * range * cos * projector.scale;
  const rightTipY = origin.y + (Math.cos(rightYaw) + Math.sin(rightYaw)) * range * sin * projector.scale + verticalDrop;

  ctx.save();
  ctx.fillStyle = hexToRgba(accent, 0.08);
  ctx.strokeStyle = hexToRgba(accent, 0.42);
  ctx.lineWidth = 0.9;
  ctx.beginPath();
  ctx.moveTo(origin.x, origin.y);
  ctx.lineTo(leftTipX, leftTipY);
  ctx.lineTo(tipX, tipY);
  ctx.lineTo(rightTipX, rightTipY);
  ctx.closePath();
  ctx.fill();
  ctx.stroke();
  ctx.restore();
}

function drawRoute(projector, routeTrail) {
  const points = (routeTrail || []).filter(isPose);
  if (points.length < 2) return;

  const projected = points.map(projector.project);

  ctx.save();
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.lineWidth = 2.8;

  // Faded older segments
  for (let i = 1; i < projected.length; i += 1) {
    const t = i / (projected.length - 1);
    const alpha = 0.22 + t * 0.72;
    ctx.strokeStyle = hexToRgba(PALETTE.cream, alpha);
    ctx.beginPath();
    ctx.moveTo(projected[i - 1].x, projected[i - 1].y);
    ctx.lineTo(projected[i].x, projected[i].y);
    ctx.stroke();
  }

  // Vertices
  for (let i = 0; i < projected.length; i += 1) {
    const t = i / Math.max(1, projected.length - 1);
    const alpha = 0.22 + t * 0.6;
    ctx.fillStyle = hexToRgba(PALETTE.cream, alpha);
    ctx.beginPath();
    ctx.arc(projected[i].x, projected[i].y, 2.2, 0, Math.PI * 2);
    ctx.fill();
  }

  // Ground projection of route (subtle shadow)
  ctx.strokeStyle = "rgba(0, 0, 0, 0.4)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  points.forEach((pose, idx) => {
    const ground = projector.projectGround(pose);
    if (idx === 0) ctx.moveTo(ground.x, ground.y);
    else ctx.lineTo(ground.x, ground.y);
  });
  ctx.stroke();

  ctx.restore();
}

function drawHome(projector, home) {
  const ground = projector.projectGround(home);
  ctx.save();
  ctx.translate(ground.x, ground.y);

  // Octagon pad
  ctx.strokeStyle = PALETTE.cream;
  ctx.fillStyle = "rgba(244, 236, 220, 0.08)";
  ctx.lineWidth = 1.4;
  ctx.beginPath();
  for (let i = 0; i < 8; i += 1) {
    const angle = (Math.PI * 2 * i) / 8 + Math.PI / 8;
    const x = Math.cos(angle) * 22 * Math.cos(Math.PI / 6);
    const y = Math.sin(angle) * 22 * Math.sin(Math.PI / 6);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.closePath();
  ctx.fill();
  ctx.stroke();

  // Cross-hair
  ctx.strokeStyle = "rgba(244, 236, 220, 0.45)";
  ctx.lineWidth = 0.8;
  ctx.beginPath();
  ctx.moveTo(-12, 0);
  ctx.lineTo(12, 0);
  ctx.moveTo(0, -7);
  ctx.lineTo(0, 7);
  ctx.stroke();

  ctx.fillStyle = PALETTE.cream;
  ctx.font = "italic 13px 'Instrument Serif', serif";
  ctx.textAlign = "center";
  ctx.fillText("home", 0, -22);
  ctx.restore();
}

function drawDrone(projector, pose, scene, obs) {
  const air = projector.project(pose);
  const ground = projector.projectGround(pose);
  const sensor = scene.sensor || "rgb";
  const accent = sensor === "thermal" ? PALETTE.coral : PALETTE.sage;
  const failed = Boolean(obs.error || obs.action_result?.error);

  // Ground shadow
  ctx.save();
  const shadowR = 18;
  const shadow = ctx.createRadialGradient(ground.x, ground.y, 1, ground.x, ground.y, shadowR);
  shadow.addColorStop(0, "rgba(0, 0, 0, 0.6)");
  shadow.addColorStop(1, "rgba(0, 0, 0, 0)");
  ctx.fillStyle = shadow;
  ctx.beginPath();
  ctx.ellipse(ground.x, ground.y, shadowR, shadowR * 0.42, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();

  // Altitude pillar
  ctx.save();
  const altGrad = ctx.createLinearGradient(0, ground.y, 0, air.y);
  altGrad.addColorStop(0, hexToRgba(accent, 0));
  altGrad.addColorStop(1, hexToRgba(accent, 0.7));
  ctx.strokeStyle = altGrad;
  ctx.lineWidth = 1.4;
  ctx.setLineDash([3, 4]);
  line(ground, air);
  ctx.setLineDash([]);
  ctx.restore();

  // Sensor frustum from the drone
  if (scene.gimbal) {
    const sensorRange = sensor === "thermal" ? 22 : 28;
    drawFrustum(
      projector,
      {
        origin: pose,
        yaw_deg: (pose.yaw_deg || 0) + (scene.gimbal.yaw_deg || 0),
        pitch_deg: scene.gimbal.pitch_deg || -20,
        range_m: sensorRange,
        hfov_deg: sensor === "thermal" ? 45 : 60,
      },
      accent,
    );
  }

  // Drone body
  ctx.save();
  ctx.translate(air.x, air.y);
  const yaw = ((pose.yaw_deg || 0) * Math.PI) / 180;
  ctx.rotate(yaw);

  if (failed) {
    ctx.strokeStyle = hexToRgba(PALETTE.coral, 0.7);
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    ctx.arc(0, 0, 34, 0, Math.PI * 2);
    ctx.stroke();
  }

  // Halo
  const halo = ctx.createRadialGradient(0, 0, 1, 0, 0, 24);
  halo.addColorStop(0, hexToRgba(accent, 0.6));
  halo.addColorStop(1, hexToRgba(accent, 0));
  ctx.fillStyle = halo;
  ctx.beginPath();
  ctx.arc(0, 0, 24, 0, Math.PI * 2);
  ctx.fill();

  // Arms
  ctx.strokeStyle = failed ? PALETTE.coral : PALETTE.cream;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(-16, -13);
  ctx.lineTo(16, 13);
  ctx.moveTo(-16, 13);
  ctx.lineTo(16, -13);
  ctx.stroke();

  // Rotors
  ctx.fillStyle = "rgba(11, 19, 17, 0.95)";
  ctx.strokeStyle = failed ? PALETTE.coral : accent;
  ctx.lineWidth = 1.4;
  for (const rotor of [
    [-17, -14],
    [17, 14],
    [-17, 14],
    [17, -14],
  ]) {
    ctx.beginPath();
    ctx.arc(rotor[0], rotor[1], 5.4, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  }

  // Body
  ctx.fillStyle = "rgba(11, 19, 17, 0.95)";
  ctx.strokeStyle = failed ? PALETTE.coral : PALETTE.cream;
  ctx.lineWidth = 1.6;
  ctx.beginPath();
  ctx.moveTo(0, -9);
  ctx.lineTo(11, 0);
  ctx.lineTo(0, 9);
  ctx.lineTo(-8, 0);
  ctx.closePath();
  ctx.fill();
  ctx.stroke();

  // Front nose
  ctx.strokeStyle = PALETTE.amber;
  ctx.lineWidth = 1.7;
  ctx.beginPath();
  ctx.moveTo(0, 0);
  ctx.lineTo(18, 0);
  ctx.stroke();

  ctx.restore();

  // Altitude readout above drone
  ctx.save();
  ctx.fillStyle = "rgba(11, 19, 17, 0.85)";
  ctx.strokeStyle = hexToRgba(accent, 0.6);
  ctx.lineWidth = 1;
  const labelText = `${formatNumber(pose.z || 0, 1)} m`;
  ctx.font = "12px 'JetBrains Mono', monospace";
  const tw = ctx.measureText(labelText).width + 14;
  const lx = air.x - tw / 2;
  const ly = air.y - 32;
  roundRect(lx, ly, tw, 18, 5);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = PALETTE.cream;
  ctx.textAlign = "center";
  ctx.fillText(labelText, air.x, ly + 12.5);
  ctx.restore();
}

function updateScale(projector) {
  const screenWidthFor10m = 10 * projector.scale;
  const totalScreenWidth = el.canvas.clientWidth;
  el.scaleBarFill.style.width = "100%";
  const meters = 10;
  el.scaleLabel.textContent = `${meters} m · ${formatNumber(screenWidthFor10m, 0)} px`;
}

/* ===========================================================
 * Captures (gallery with synthetic thumbnail previews)
 * =========================================================== */

function renderCaptures() {
  const obs = state.observation || {};
  const scene = state.scene || obs.scene || {};
  const captures =
    scene.capture_points?.length ? scene.capture_points :
    obs.capture_log?.length ? obs.capture_log :
    obs.evidence_artifacts || [];

  el.artifactCount.textContent = `${captures.length} artifact${captures.length === 1 ? "" : "s"}`;

  if (!captures.length) {
    el.captureList.innerHTML = emptyMarkup(
      "Evidence locker empty",
      "Thermal & RGB captures will appear here as you collect them.",
    );
    return;
  }

  el.captureList.innerHTML = [...captures]
    .reverse()
    .map((capture, index) => {
      const sensor = (capture.sensor || "rgb").toLowerCase();
      const quality = clamp(Number(capture.quality_score ?? 0), 0, 1);
      const coverage = clamp(Number(capture.coverage_pct ?? 0), 0, 1);
      const anomalies = capture.detected_anomalies || [];
      const targets = capture.targets_visible || capture.asset_ids || [];
      const id = `capture-thumb-${index}`;
      return `<article class="capture-card">
        <div class="capture-thumb">
          <span class="capture-thumb__sensor">${escapeHtml(sensor)}</span>
          <canvas id="${id}" width="176" height="176"></canvas>
        </div>
        <div class="capture-info">
          <p class="capture-info__title">${escapeHtml(capture.photo_id || "capture")}</p>
          <p class="capture-info__sub">${escapeHtml(capture.label || "—")}</p>
          <div class="capture-meta">
            <span class="chip chip--sage">q ${formatNumber(quality, 2)}</span>
            <span class="chip chip--amber">${formatNumber(coverage * 100, 0)}% cov</span>
            <span class="chip">${targets.length} tgt</span>
            <span class="chip ${anomalies.length ? "chip--coral" : ""}">${anomalies.length} anom</span>
          </div>
          <p class="capture-info__sub">${escapeHtml(targets.join(" · ") || "no visible targets")}</p>
        </div>
      </article>`;
    })
    .join("");

  // Paint synthetic thumbnails after DOM is updated
  [...captures]
    .reverse()
    .forEach((capture, index) => {
      const canvas = document.getElementById(`capture-thumb-${index}`);
      if (canvas) drawCaptureThumb(canvas, capture);
    });
}

function drawCaptureThumb(canvas, capture) {
  const c = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  const sensor = (capture.sensor || "rgb").toLowerCase();
  const quality = clamp(Number(capture.quality_score ?? 0), 0, 1);
  const coverage = clamp(Number(capture.coverage_pct ?? 0), 0, 1);
  const anomalies = (capture.detected_anomalies || []).length;

  c.clearRect(0, 0, w, h);

  // Background based on sensor
  const grad = c.createRadialGradient(w * 0.5, h * 0.5, 4, w * 0.5, h * 0.5, w * 0.7);
  if (sensor === "thermal") {
    grad.addColorStop(0, "#5a1a0a");
    grad.addColorStop(0.5, "#2a0d08");
    grad.addColorStop(1, "#0d0504");
  } else {
    grad.addColorStop(0, "#1f3833");
    grad.addColorStop(0.5, "#0d1a18");
    grad.addColorStop(1, "#06100e");
  }
  c.fillStyle = grad;
  c.fillRect(0, 0, w, h);

  // Frame
  c.strokeStyle = "rgba(244, 236, 220, 0.18)";
  c.lineWidth = 1;
  c.strokeRect(4, 4, w - 8, h - 8);

  // Pseudo content: scanlines + dots
  const seed = (capture.photo_id || "x").split("").reduce((acc, ch) => acc + ch.charCodeAt(0), 0);
  const rnd = mulberry32(seed);

  for (let i = 0; i < 8; i += 1) {
    c.strokeStyle = sensor === "thermal" ? `rgba(244, 184, 96, ${0.05 + i * 0.04})` : `rgba(138, 214, 193, ${0.04 + i * 0.04})`;
    c.lineWidth = 0.8;
    c.beginPath();
    const y = (i + 1) * (h / 9);
    c.moveTo(8, y);
    for (let x = 8; x < w - 8; x += 6) {
      const dy = (rnd() - 0.5) * 4;
      c.lineTo(x, y + dy);
    }
    c.stroke();
  }

  // "Target" — quality blob in center
  const blobSize = 40 + quality * 50;
  const blob = c.createRadialGradient(w / 2, h / 2, 4, w / 2, h / 2, blobSize);
  if (sensor === "thermal") {
    blob.addColorStop(0, `rgba(255, 220, 140, ${0.1 + quality * 0.5})`);
    blob.addColorStop(0.4, `rgba(244, 184, 96, ${0.08 + quality * 0.35})`);
    blob.addColorStop(1, "rgba(255, 138, 107, 0)");
  } else {
    blob.addColorStop(0, `rgba(184, 232, 168, ${0.12 + quality * 0.55})`);
    blob.addColorStop(0.5, `rgba(138, 214, 193, ${0.08 + quality * 0.4})`);
    blob.addColorStop(1, "rgba(95, 178, 155, 0)");
  }
  c.fillStyle = blob;
  c.beginPath();
  c.arc(w / 2, h / 2, blobSize, 0, Math.PI * 2);
  c.fill();

  // Anomaly hotspots
  for (let i = 0; i < anomalies; i += 1) {
    const ax = w * (0.3 + rnd() * 0.4);
    const ay = h * (0.3 + rnd() * 0.4);
    c.strokeStyle = "rgba(255, 138, 107, 0.85)";
    c.lineWidth = 1.4;
    c.beginPath();
    c.arc(ax, ay, 6 + rnd() * 6, 0, Math.PI * 2);
    c.stroke();
    c.strokeStyle = "rgba(255, 138, 107, 0.45)";
    c.lineWidth = 0.8;
    c.beginPath();
    c.moveTo(ax - 12, ay);
    c.lineTo(ax + 12, ay);
    c.moveTo(ax, ay - 12);
    c.lineTo(ax, ay + 12);
    c.stroke();
  }

  // Coverage gauge corner
  c.fillStyle = "rgba(11, 19, 17, 0.78)";
  c.fillRect(w - 36, h - 16, 32, 12);
  c.fillStyle = sensor === "thermal" ? "rgba(244, 184, 96, 0.85)" : "rgba(138, 214, 193, 0.85)";
  c.fillRect(w - 35, h - 15, 30 * coverage, 10);
  c.strokeStyle = "rgba(244, 236, 220, 0.3)";
  c.lineWidth = 0.5;
  c.strokeRect(w - 36, h - 16, 32, 12);

  // Crosshair
  c.strokeStyle = "rgba(244, 236, 220, 0.32)";
  c.lineWidth = 0.6;
  c.beginPath();
  c.moveTo(w / 2, 8);
  c.lineTo(w / 2, h - 8);
  c.moveTo(8, h / 2);
  c.lineTo(w - 8, h / 2);
  c.stroke();
}

function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/* ===========================================================
 * Helpers
 * =========================================================== */

function rectPoints(zone) {
  return [
    { x: zone.min_x, y: zone.min_y, z: 0 },
    { x: zone.max_x, y: zone.min_y, z: 0 },
    { x: zone.max_x, y: zone.max_y, z: 0 },
    { x: zone.min_x, y: zone.max_y, z: 0 },
  ];
}

function zonePolygon(zone) {
  if (Array.isArray(zone.polygon_xy) && zone.polygon_xy.length) {
    return zone.polygon_xy.map((point) => ({ x: point.x, y: point.y, z: 0 }));
  }
  const bounds = zone.bounds || zone;
  return rectPoints(bounds);
}

function assetGeometry(asset) {
  const geometry = asset.geometry || asset;
  const center = asset.center || geometry.center || geometry;
  return {
    center_x: center.center_x ?? center.x ?? 0,
    center_y: center.center_y ?? center.y ?? 0,
    center_z: center.center_z ?? center.z ?? 0,
    width_m: geometry.width_m || asset.width_m || 10,
    height_m: geometry.height_m || asset.height_m || 4,
  };
}

function polygon(points, fill = false) {
  ctx.beginPath();
  points.forEach((point, index) => {
    if (index === 0) ctx.moveTo(point.x, point.y);
    else ctx.lineTo(point.x, point.y);
  });
  ctx.closePath();
  if (fill) ctx.fill();
  ctx.stroke();
}

function hatchInside(points, color) {
  if (!points.length) return;
  const box = points.reduce(
    (acc, point) => ({
      minX: Math.min(acc.minX, point.x),
      maxX: Math.max(acc.maxX, point.x),
      minY: Math.min(acc.minY, point.y),
      maxY: Math.max(acc.maxY, point.y),
    }),
    { minX: Infinity, maxX: -Infinity, minY: Infinity, maxY: -Infinity },
  );
  ctx.save();
  ctx.beginPath();
  points.forEach((point, index) => {
    if (index === 0) ctx.moveTo(point.x, point.y);
    else ctx.lineTo(point.x, point.y);
  });
  ctx.closePath();
  ctx.clip();
  ctx.strokeStyle = color || "rgba(255, 138, 107, 0.16)";
  ctx.lineWidth = 0.8;
  for (let x = box.minX - 40; x < box.maxX + 40; x += 8) {
    line({ x, y: box.minY - 12 }, { x: x + 30, y: box.maxY + 12 });
  }
  ctx.restore();
}

function line(a, b) {
  ctx.beginPath();
  ctx.moveTo(a.x, a.y);
  ctx.lineTo(b.x, b.y);
  ctx.stroke();
}

function centroid(points) {
  return {
    x: points.reduce((sum, point) => sum + point.x, 0) / points.length,
    y: points.reduce((sum, point) => sum + point.y, 0) / points.length,
  };
}

function drawLabel(text, point, accent) {
  ctx.save();
  ctx.font = "italic 14px 'Instrument Serif', serif";
  ctx.textAlign = "center";
  const label = String(text);
  const width = ctx.measureText(label).width + 16;
  const lx = point.x - width / 2;
  const ly = point.y - 20;
  ctx.fillStyle = "rgba(11, 19, 17, 0.82)";
  ctx.strokeStyle = hexToRgba(accent, 0.55);
  ctx.lineWidth = 0.9;
  roundRect(lx, ly, width, 18, 5);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = accent;
  ctx.fillText(label, point.x, ly + 13);
  ctx.restore();
}

function labelAt(text, point, color) {
  drawLabel(text, point, color);
}

function roundRect(x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}

function missionTitle(obs) {
  return obs?.mission?.task_name || obs?.mission?.name || obs?.mission?.task_id || obs?.mission?.mission_id || "";
}

function emptyMarkup(headline, message) {
  return `<div class="empty-state">
    <em>${escapeHtml(headline)}</em>
    <span>${escapeHtml(message || "")}</span>
  </div>`;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatNumber(value, digits = 2) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(digits) : "—";
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, Number(value) || 0));
}

function hexToRgba(hex, alpha) {
  const clean = hex.replace("#", "");
  if (clean.length === 3) {
    const r = Number.parseInt(clean[0] + clean[0], 16);
    const g = Number.parseInt(clean[1] + clean[1], 16);
    const b = Number.parseInt(clean[2] + clean[2], 16);
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
  }
  const int = Number.parseInt(clean, 16);
  const r = (int >> 16) & 255;
  const g = (int >> 8) & 255;
  const b = int & 255;
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

/* ===========================================================
 * Camera controls — pan, zoom, keyboard, touch
 * =========================================================== */

function resetCamera() {
  state.camera = { zoom: 1, panX: 0, panY: 0, lock: null };
  updateZoomReadout();
}

function fitScene() {
  resetCamera();
  renderScene();
}

function lockCameraIfNeeded() {
  if (state.camera.lock || !state.projector) return;
  // Freeze the auto-fit values when the user starts navigating so subsequent
  // step()s don't re-center the world under their feet.
  const p = state.projector;
  state.camera.lock = {
    centerX: p.centerX,
    centerY: p.centerY,
    minX: p.minX,
    maxX: p.maxX,
    minY: p.minY,
    maxY: p.maxY,
    baseScale: p.scale / state.camera.zoom,
    baseCx: p.cx - state.camera.panX,
    baseCy: p.cy - state.camera.panY,
  };
}

function zoomBy(factor, cursorX = null, cursorY = null) {
  if (!state.projector) {
    state.camera.zoom = clamp(state.camera.zoom * factor, 0.25, 8);
    renderScene();
    return;
  }
  lockCameraIfNeeded();
  const px = cursorX ?? el.canvas.clientWidth / 2;
  const py = cursorY ?? el.canvas.clientHeight / 2;
  const before = state.projector.unproject(px, py);
  state.camera.zoom = clamp(state.camera.zoom * factor, 0.25, 8);
  renderScene();
  if (state.projector) {
    const after = state.projector.project({ x: before.x, y: before.y, z: 0 });
    state.camera.panX += px - after.x;
    state.camera.panY += py - after.y;
    renderScene();
  }
}

function panBy(dx, dy) {
  lockCameraIfNeeded();
  state.camera.panX += dx;
  state.camera.panY += dy;
  renderScene();
}

function attachCameraControls() {
  const canvas = el.canvas;
  let isPanning = false;
  let lastX = 0;
  let lastY = 0;
  let pinchDistance = null;
  let pinchCenter = null;

  canvas.addEventListener("mousedown", (event) => {
    if (event.button !== 0) return;
    event.preventDefault();
    isPanning = true;
    lastX = event.clientX;
    lastY = event.clientY;
    canvas.classList.add("is-grabbing");
  });

  window.addEventListener("mousemove", (event) => {
    if (!isPanning) return;
    event.preventDefault();
    const dx = event.clientX - lastX;
    const dy = event.clientY - lastY;
    lastX = event.clientX;
    lastY = event.clientY;
    panBy(dx, dy);
  });

  window.addEventListener("mouseup", () => {
    if (!isPanning) return;
    isPanning = false;
    canvas.classList.remove("is-grabbing");
  });

  canvas.addEventListener("mouseleave", () => {
    canvas.classList.remove("is-grabbing");
  });

  canvas.addEventListener("wheel", (event) => {
    event.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const cursorX = event.clientX - rect.left;
    const cursorY = event.clientY - rect.top;
    const factor = Math.exp(-event.deltaY * 0.0015);
    zoomBy(factor, cursorX, cursorY);
  }, { passive: false });

  canvas.addEventListener("touchstart", (event) => {
    if (event.touches.length === 1) {
      isPanning = true;
      lastX = event.touches[0].clientX;
      lastY = event.touches[0].clientY;
    } else if (event.touches.length === 2) {
      isPanning = false;
      pinchDistance = touchDistance(event.touches);
      pinchCenter = touchCenter(event.touches);
    }
    event.preventDefault();
  }, { passive: false });

  canvas.addEventListener("touchmove", (event) => {
    if (event.touches.length === 1 && isPanning) {
      const dx = event.touches[0].clientX - lastX;
      const dy = event.touches[0].clientY - lastY;
      lastX = event.touches[0].clientX;
      lastY = event.touches[0].clientY;
      panBy(dx, dy);
    } else if (event.touches.length === 2 && pinchDistance) {
      const newDistance = touchDistance(event.touches);
      const newCenter = touchCenter(event.touches);
      const factor = newDistance / pinchDistance;
      const rect = canvas.getBoundingClientRect();
      zoomBy(factor, newCenter.x - rect.left, newCenter.y - rect.top);
      panBy(newCenter.x - pinchCenter.x, newCenter.y - pinchCenter.y);
      pinchDistance = newDistance;
      pinchCenter = newCenter;
    }
    event.preventDefault();
  }, { passive: false });

  canvas.addEventListener("touchend", (event) => {
    if (event.touches.length < 2) {
      pinchDistance = null;
      pinchCenter = null;
    }
    if (event.touches.length === 0) {
      isPanning = false;
    } else if (event.touches.length === 1) {
      lastX = event.touches[0].clientX;
      lastY = event.touches[0].clientY;
      isPanning = true;
    }
  });

  window.addEventListener("keydown", (event) => {
    if (isTextInputFocused()) return;
    const step = event.shiftKey ? 100 : 48;
    let handled = true;
    if (event.key === "ArrowLeft") panBy(step, 0);
    else if (event.key === "ArrowRight") panBy(-step, 0);
    else if (event.key === "ArrowUp") panBy(0, step);
    else if (event.key === "ArrowDown") panBy(0, -step);
    else if (event.key === "+" || event.key === "=") zoomBy(1.2);
    else if (event.key === "-" || event.key === "_") zoomBy(1 / 1.2);
    else if (event.key === "0") {
      state.camera.zoom = 1;
      state.camera.panX = 0;
      state.camera.panY = 0;
      renderScene();
    } else if (event.key === "f" || event.key === "F") {
      fitScene();
    } else {
      handled = false;
    }
    if (handled) event.preventDefault();
  });

  el.cameraButtons.forEach((button) => {
    button.addEventListener("click", () => {
      const action = button.dataset.camera;
      const rect = canvas.getBoundingClientRect();
      const cx = rect.width / 2;
      const cy = rect.height / 2;
      if (action === "zoom-in") zoomBy(1.25, cx, cy);
      else if (action === "zoom-out") zoomBy(1 / 1.25, cx, cy);
      else if (action === "fit") fitScene();
    });
  });
}

function isTextInputFocused() {
  const active = document.activeElement;
  if (!active) return false;
  const tag = active.tagName;
  return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || active.isContentEditable;
}

function touchDistance(touches) {
  const dx = touches[0].clientX - touches[1].clientX;
  const dy = touches[0].clientY - touches[1].clientY;
  return Math.hypot(dx, dy);
}

function touchCenter(touches) {
  return {
    x: (touches[0].clientX + touches[1].clientX) / 2,
    y: (touches[0].clientY + touches[1].clientY) / 2,
  };
}

/* ===========================================================
 * Bootstrap
 * =========================================================== */

el.sessionForm.addEventListener("submit", startSession);
el.stepButton.addEventListener("click", executeStep);
el.refreshSessions.addEventListener("click", refreshSessions);
el.runModelButton.addEventListener("click", runLiveModel);
el.replayButton.addEventListener("click", replayTrajectoryRecord);
el.compareButton.addEventListener("click", runComparison);
el.overlayButton.addEventListener("click", overlayBestComparisonRoute);
el.modelPolicy.addEventListener("change", () => applyModelPreset(el.modelPolicy.value));
el.clearLog.addEventListener("click", () => {
  state.actionLog = [];
  renderLog();
});
el.toolSelect.addEventListener("change", () => selectPreset(el.toolSelect.value));
el.quickActions.addEventListener("click", (event) => {
  const button = event.target.closest("button[data-tool]");
  if (button) selectPreset(button.dataset.tool);
});
window.addEventListener("resize", () => {
  renderScene();
});

attachCameraControls();
renderQuickActions();
updateToolOptions({ available_tools: Object.keys(ACTION_PRESETS) });
render();

setInterval(() => {
  // Tick the mission clock and refresh relative timestamps in the log
  if (state.sessionStartedAt) {
    renderInstruments();
    renderLog();
  }
}, 1000);
