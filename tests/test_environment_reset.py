from dronecaptureops.core.environment import DroneCaptureOpsEnvironment


def test_reset_is_deterministic_for_same_seed():
    env_a = DroneCaptureOpsEnvironment()
    env_b = DroneCaptureOpsEnvironment()

    obs_a = env_a.reset(seed=11)
    obs_b = env_b.reset(seed=11)

    assert obs_a.mission == obs_b.mission
    assert obs_a.site_map == obs_b.site_map
    assert obs_a.telemetry == obs_b.telemetry
    assert env_a.state.scenario_seed == 11
    assert env_b.state.scenario_seed == 11
