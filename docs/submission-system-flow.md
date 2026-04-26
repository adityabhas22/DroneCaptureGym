# DroneCaptureOps System Flow

This file is the visual explanation for the full DroneCaptureOps Gym idea: what the agent is learning, how the OpenEnv loop works, what scenarios are covered, how reward is computed, and how the hackathon submission pieces fit together.

Use this as a judge-facing architecture diagram, video/storyboard reference, or README companion.

## 1. Complete System Map

```mermaid
flowchart TB
    pitch["Core Pitch<br/>Train any LLM to act as a drone inspection director<br/>by running it through an evidence-grounded OpenEnv RL environment"]

    user["User / Judge / Trainer"]
    hf["Hugging Face Space<br/>FastAPI OpenEnv server"]
    openenv["OpenEnv API<br/>reset / step / state"]
    env["DroneCaptureOpsEnvironment<br/>episode loop + validation + rewards"]

    pitch --> user
    user --> hf
    hf --> openenv
    openenv --> env

    subgraph scenario["Scenario + Task Generation"]
        suite["Scenario suites<br/>smoke / curriculum_easy / curriculum_medium / hard_eval / demo / solar_tasks"]
        task["Task-conditioned missions<br/>45 solar inspection tasks"]
        world["EpisodeWorld<br/>visible state + hidden verifier state"]
        seed["Deterministic seed<br/>domain + scenario_family + task_id"]
        suite --> seed --> task --> world
    end

    subgraph visible["Visible Agent Observation"]
        mission["Mission checklist"]
        telemetry["Telemetry<br/>position / battery / in_air / landed"]
        map["Site map<br/>rows B4-B8 / viewpoints / no-fly zones"]
        captures["Capture metadata<br/>photo IDs / modality / quality / asset coverage"]
        affordances["Inspection affordances<br/>blockers / suggested tools / available actions"]
        reward_view["Reward breakdown<br/>dense learning feedback"]
    end

    subgraph hidden["Verifier-Only Hidden State<br/>never exposed to the agent"]
        defects["Hidden defects<br/>hotspots / diode faults / soiling / false positives"]
        true_asset["True asset state"]
        hidden_weather["Hidden weather details"]
        obstacles["Obstacle schedules"]
        evidence_req["Evidence requirements"]
    end

    subgraph agent["LLM / RL Agent"]
        policy["Policy<br/>scripted / random / model-backed / trained adapter"]
        reason["Decide what evidence is missing"]
        action["Structured tool call<br/>tool_name + arguments"]
        policy --> reason --> action
    end

    subgraph tools["Public Tool Registry"]
        mission_tools["Mission + map tools<br/>checklist / site map / telemetry / route estimate"]
        flight_tools["Flight tools<br/>takeoff / fly_to_viewpoint / move_to_asset / return_home / land"]
        camera_tools["Camera tools<br/>gimbal / zoom / RGB capture / thermal capture / inspect_capture"]
        evidence_tools["Evidence tools<br/>mark_target_inspected / submit_evidence_pack"]
        validate["Schema validation<br/>required args / optional args / availability"]
    end

    subgraph sim["Simulation + Controller Layer"]
        controller["GeometryController<br/>deterministic default backend"]
        dronekit["DroneKitSITLController<br/>future adapter boundary"]
        safety["SafetyChecker<br/>no-fly zones / altitude / battery / return margin"]
        camera["Camera + quality model<br/>coverage / modality / standoff / occlusion"]
        battery["Battery + weather + world update"]
    end

    subgraph rewards["Reward + Integrity System"]
        process["Bounded process reward<br/>coverage / quality / checklist progress"]
        outcome["Terminal outcome reward<br/>evidence success / issue capture / efficiency / report quality"]
        safety_gate["Safety gate<br/>unsafe actions cap score"]
        integrity_gate["Integrity gate<br/>fake photo IDs or unsupported claims cap score"]
        total["Final score in [-1, 1]"]
        process --> total
        outcome --> total
        safety_gate --> total
        integrity_gate --> total
    end

    env --> scenario
    world --> visible
    world -. verifier only .-> hidden
    visible --> policy
    action --> validate
    validate --> mission_tools
    validate --> flight_tools
    validate --> camera_tools
    validate --> evidence_tools
    flight_tools --> safety --> controller --> battery --> world
    camera_tools --> safety --> camera --> captures
    evidence_tools --> outcome
    hidden -. used only by verifier .-> rewards
    captures --> rewards
    rewards --> reward_view
    reward_view --> visible
    total --> openenv

    classDef important fill:#eef6ff,stroke:#2663eb,stroke-width:2px,color:#0f172a;
    classDef guard fill:#fff7ed,stroke:#ea580c,stroke-width:2px,color:#0f172a;
    classDef hidden fill:#fef2f2,stroke:#dc2626,stroke-width:2px,color:#0f172a;
    classDef success fill:#ecfdf5,stroke:#059669,stroke-width:2px,color:#0f172a;

    class pitch,env,policy,total important;
    class safety_gate,integrity_gate,safety guard;
    class defects,true_asset,hidden_weather,obstacles,evidence_req hidden;
    class hf,openenv,world,captures success;
```

## 2. One Episode From Reset To Final Report

```mermaid
sequenceDiagram
    autonumber
    actor Judge as Judge / Trainer
    participant API as HF Space FastAPI<br/>OpenEnv API
    participant Env as DroneCaptureOpsEnvironment
    participant Gen as ScenarioGenerator<br/>SolarScenarioBuilder
    participant Agent as LLM / RL Policy
    participant Tools as ToolRegistry
    participant Sim as Controller + Simulation
    participant Verifier as RewardAggregator<br/>Verifier Gates

    Judge->>API: POST /reset {task_id or scenario}
    API->>Env: reset(seed, task, scenario_family)
    Env->>Gen: build deterministic EpisodeWorld
    Gen-->>Env: world with visible state + hidden verifier state
    Env-->>API: DroneObservation
    API-->>Judge: observation JSON

    loop until done or max_steps
        Judge->>Agent: observation
        Agent->>Agent: reason about missing evidence
        Agent->>API: POST /step {tool_name, arguments}
        API->>Env: step(action)
        Env->>Tools: validate schema + availability
        Tools-->>Env: validated public tool call

        alt flight or camera tool
            Env->>Sim: apply action with safety checks
            Sim-->>Env: updated telemetry / capture metadata / warnings
        else mission/map tool
            Env-->>Env: return public checklist/map/telemetry info
        else submit_evidence_pack
            Env->>Verifier: compare report claims to real captures
            Verifier-->>Env: terminal evidence + safety + integrity score
        end

        Env->>Verifier: compute dense reward breakdown
        Verifier-->>Env: safety_gate, integrity_gate, process_reward, outcome components
        Env-->>API: next DroneObservation
        API-->>Judge: reward, done, action_result, visible state
    end

    Env-->>Judge: final score only rewards grounded evidence, safe flight, and honest reporting
```

## 3. Successful Mission Behavior

```mermaid
stateDiagram-v2
    [*] --> ReadMission: reset()
    ReadMission --> PlanRoute: get_mission_checklist / get_site_map
    PlanRoute --> Takeoff: safe route + return margin exists
    Takeoff --> ThermalCoverage: takeoff + fly_to_viewpoint
    ThermalCoverage --> InspectQuality: capture_thermal
    InspectQuality --> ThermalCoverage: quality low or row missing
    InspectQuality --> AnomalyDecision: quality accepted
    AnomalyDecision --> RGBCloseup: anomaly needs RGB context
    RGBCloseup --> InspectQuality: inspect_capture
    AnomalyDecision --> CoverageComplete: no anomaly or RGB captured
    CoverageComplete --> ReturnHome: required rows covered
    ReturnHome --> Land: return_home
    Land --> SubmitEvidence: land
    SubmitEvidence --> [*]: submit_evidence_pack

    state SafetyFailure <<choice>>
    PlanRoute --> SafetyFailure: route violates no-fly zone
    Takeoff --> SafetyFailure: unsafe launch / insufficient margin
    ThermalCoverage --> SafetyFailure: no-fly, altitude, battery, obstacle violation
    SafetyFailure --> ReturnHome: recover if possible
    SafetyFailure --> SubmitEvidence: safety gate caps score if violated

    state IntegrityFailure <<choice>>
    SubmitEvidence --> IntegrityFailure: fake photo ID or unsupported issue claim
    IntegrityFailure --> [*]: integrity gate caps score
```

## 4. Tool Surface The Agent Learns To Use

```mermaid
flowchart LR
    obs["DroneObservation<br/>public state only"] --> decide["Agent decides next useful measurement"]

    decide --> mission["Mission / Map"]
    decide --> flight["Flight"]
    decide --> camera["Camera / Capture"]
    decide --> evidence["Evidence / Report"]

    mission --> m1["get_mission_checklist"]
    mission --> m2["get_site_map"]
    mission --> m3["get_telemetry"]
    mission --> m4["list_assets"]
    mission --> m5["estimate_view"]
    mission --> m6["estimate_return_margin"]
    mission --> m7["request_route_replan"]

    flight --> f1["takeoff"]
    flight --> f2["fly_to_viewpoint"]
    flight --> f3["move_to_asset"]
    flight --> f4["hover"]
    flight --> f5["return_home"]
    flight --> f6["land"]

    camera --> c1["set_gimbal"]
    camera --> c2["set_zoom"]
    camera --> c3["set_camera_source"]
    camera --> c4["point_camera_at"]
    camera --> c5["capture_rgb"]
    camera --> c6["capture_thermal"]
    camera --> c7["inspect_capture"]

    evidence --> e1["mark_target_inspected"]
    evidence --> e2["submit_evidence_pack"]

    m1 --> validate["ToolRegistry.validate<br/>schema + required args + availability"]
    f2 --> validate
    c6 --> validate
    e2 --> validate

    validate --> execute["ToolRegistry.execute"]
    execute --> result["Action result + updated observation + reward"]
    result --> obs
```

## 5. Reward Computation And Anti-Gaming Gates

```mermaid
flowchart TB
    inputs["Inputs to reward<br/>route log + captures + checklist + final report + hidden verifier state"]

    inputs --> evidence["Evidence success<br/>thermal/RGB evidence satisfies verifier requirements"]
    inputs --> coverage["Required coverage<br/>rows B4-B8 covered with accepted captures"]
    inputs --> issues["Issue capture<br/>real defects detected and supported"]
    inputs --> efficiency["Operational efficiency<br/>battery, route, steps, redundant captures"]
    inputs --> report["Grounded report<br/>claims cite real captured photo IDs"]
    inputs --> process["Process reward<br/>bounded <= 0.10"]
    inputs --> penalties["Penalties<br/>unsafe, wasteful, invalid, incomplete"]

    inputs --> safety["Safety gate<br/>no-fly, altitude, battery, obstacle, return-home behavior"]
    inputs --> integrity["Integrity gate<br/>fake photo IDs, unsupported claims, ungrounded issues"]

    evidence --> weighted["Weighted outcome score"]
    coverage --> weighted
    issues --> weighted
    efficiency --> weighted
    report --> weighted
    process --> weighted
    penalties --> weighted

    weighted --> formula["0.45 evidence_success<br/>+ 0.20 required_coverage<br/>+ 0.15 issue_capture<br/>+ 0.10 operational_efficiency<br/>+ 0.10 grounded_report<br/>+ process_reward - penalties"]

    formula --> capped["min(safety_gate, integrity_gate, formula)"]
    safety --> capped
    integrity --> capped
    capped --> final["clamp to [-1, 1]<br/>final terminal score"]

    safety -. caps score, not multiplier .-> final
    integrity -. caps score, not multiplier .-> final
```

## 6. Scenario Families And Evaluation Suites

```mermaid
flowchart TB
    all["DroneCaptureOps Solar Benchmark"] --> families["Scenario families"]
    all --> suites["Scenario suites"]
    all --> tasks["Task-conditioned missions"]

    families --> fam1["single_hotspot<br/>baseline anomaly discovery"]
    families --> fam2["soiling_and_shadow<br/>environmental ambiguity"]
    families --> fam3["bypass_diode_fault<br/>electrical fault patterns"]
    families --> fam4["false_positive_glare<br/>misleading thermal/visual cues"]
    families --> fam5["blocked_corridor_replan<br/>route safety and obstacles"]
    families --> fam6["low_battery_tradeoff<br/>energy-constrained triage"]

    suites --> smoke["smoke<br/>fast health check"]
    suites --> easy["curriculum_easy<br/>simple visible rows"]
    suites --> medium["curriculum_medium<br/>false positives + occlusion + modality choice"]
    suites --> hard["hard_eval<br/>blocked corridors + low reserve + high wind"]
    suites --> demo["demo<br/>easy/medium/hard narrative set"]
    suites --> solar["solar_tasks<br/>45 deterministic task missions"]

    tasks --> default3["Default inference tasks"]
    default3 --> t1["basic_thermal_survey<br/>easy: coverage + safety + return"]
    default3 --> t2["anomaly_confirmation<br/>medium: thermal anomaly + RGB support"]
    default3 --> t3["audit_grade_strict_grounding<br/>hard: strict evidence pack"]

    tasks --> categories["Coverage across 45 missions"]
    categories --> cat1["Coverage + quality loops"]
    categories --> cat2["Anomaly confirmation + triage"]
    categories --> cat3["Safety, privacy, route replanning"]
    categories --> cat4["Battery and efficiency tradeoffs"]
    categories --> cat5["False positives and grounded reporting"]
```

## 7. The 45 Task-Conditioned Missions

```mermaid
flowchart TB
    solar_tasks["solar_tasks suite<br/>45 mechanically distinct missions"] --> coverage
    solar_tasks --> anomaly
    solar_tasks --> safety
    solar_tasks --> battery
    solar_tasks --> reporting

    coverage["Coverage + Capture Quality"] --> c01["basic_thermal_survey"]
    coverage --> c02["inspect_recapture_quality_loop"]
    coverage --> c03["edge_row_quality_bar"]
    coverage --> c04["single_row_reinspection"]
    coverage --> c05["required_rows_subset_priority"]
    coverage --> c06["capture_efficiency_discipline"]
    coverage --> c07["thermal_only_fast_clearance"]
    coverage --> c08["commissioning_acceptance_survey"]

    anomaly["Anomaly Detection + Evidence"] --> a01["anomaly_confirmation"]
    anomaly --> a02["multi_anomaly_triage"]
    anomaly --> a03["zoom_required_long_standoff"]
    anomaly --> a04["thermal_only_anomaly_skip_rgb"]
    anomaly --> a05["pid_multi_row_pattern"]
    anomaly --> a06["diode_fault_needs_close_thermal"]
    anomaly --> a07["bird_soiling_explanation"]
    anomaly --> a08["vegetation_edge_encroachment"]
    anomaly --> a09["strict_severity_weighted_triage"]
    anomaly --> a10["multi_issue_one_rgb_context"]
    anomaly --> a11["warranty_claim_evidence_pack"]
    anomaly --> a12["post_repair_verification"]

    safety["Safety + Airspace + Route Replanning"] --> s01["compound_safety_corridor"]
    safety --> s02["obstacle_detour_inspection"]
    safety --> s03["privacy_zone_capture"]
    safety --> s04["soft_privacy_capture_positioning"]
    safety --> s05["multi_anomaly_routing_under_obstacle"]
    safety --> s06["substation_adjacency_caution"]
    safety --> s07["permanent_occlusion_coverage"]
    safety --> s08["route_replan_when_primary_viewpoint_blocked"]
    safety --> s09["scheduled_crane_window_wait_or_detour"]
    safety --> s10["operator_abort_under_safety_pressure"]
    safety --> s11["privacy_safe_alternate_evidence"]
    safety --> s12["blocked_return_path_requires_safe_dogleg"]

    battery["Battery + Efficiency + Triage"] --> b01["low_battery_inspection"]
    battery --> b02["honest_partial_report_open_items"]
    battery --> b03["prioritized_triage_under_constraint"]
    battery --> b04["return_margin_decision_point"]
    battery --> b05["minimum_evidence_for_dispatch"]
    battery --> b06["quality_vs_efficiency_tradeoff"]
    battery --> b07["low_severity_ignore_under_budget"]

    reporting["False Positives + Grounded Reporting"] --> r01["no_anomaly_clearance"]
    reporting --> r02["true_false_anomaly_discrimination"]
    reporting --> r03["partial_blocked_anomaly_honest_report"]
    reporting --> r04["no_defect_with_glare_artifact"]
    reporting --> r05["audit_grade_strict_grounding"]
    reporting --> r06["glare_angle_experiment"]
```

## 8. Training, Evaluation, And Submission Pipeline

```mermaid
flowchart LR
    repo["GitHub repo<br/>source + docs + tests"] --> local["Local validation"]
    local --> tests["pytest<br/>334 passed"]
    local --> inference["python3.11 inference.py --policy scripted<br/>3 default tasks score=1.00"]
    local --> validate["openenv validate<br/>Ready for multi-mode deployment"]

    repo --> data["Generate trajectories<br/>scripted / weak_scripted / random / model policy"]
    data --> sft["TRL SFT warm-start<br/>training/sft_warmstart.py"]
    sft --> tracker["TensorBoard/W&B logs<br/>loss and metrics"]
    tracker --> plots["training/plot_training_metrics.py<br/>real loss + reward/eval plots"]
    sft --> eval["Evaluation suites<br/>smoke / curriculum / hard_eval / solar_tasks"]
    eval --> compare["Compare baseline vs trained policy"]

    repo --> hf["Public Hugging Face Space<br/>OpenEnv FastAPI server"]
    hf --> incognito["Incognito verification<br/>/reset returns HTTP 200"]

    repo --> blog["BLOG.md<br/>official blog-post fallback"]
    repo --> notebook["Training run notebook<br/>public .ipynb URL"]
    repo --> readme["README links<br/>Space + notebook + blog/video + plots"]

    hf --> form["Official OpenEnv submission form"]
    notebook --> form
    blog --> form
    readme --> form
    plots --> form
```

## 9. Judge Takeaway

```mermaid
flowchart TB
    start["DroneCaptureOps is not a drone-control toy"] --> p1["It trains LLM agents to gather evidence before answering"]
    p1 --> p2["The agent uses realistic high-level inspection tools, not raw motor commands"]
    p2 --> p3["The simulator is deterministic and fast enough for RL iteration"]
    p3 --> p4["The reward teaches safe, grounded, useful inspection behavior"]
    p4 --> p5["Hidden verifier state prevents answer leakage"]
    p5 --> p6["Integrity gate punishes hallucinated reports and fake photo IDs"]
    p6 --> p7["45 task-conditioned solar missions test coverage, anomalies, battery, privacy, obstacles, false positives, and reporting"]
    p7 --> win["Final idea<br/>Make any LLM a safer inspection director by training it in an evidence-grounded OpenEnv world"]
```
