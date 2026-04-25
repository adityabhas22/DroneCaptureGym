from dronecaptureops.core.environment import DroneCaptureOpsEnvironment


def test_observation_and_visible_state_do_not_leak_hidden_defects():
    env = DroneCaptureOpsEnvironment()
    obs = env.reset(seed=7)

    obs_json = obs.model_dump_json()
    state_json = env.state.model_dump_json()

    assert env.debug_world.hidden_defects
    for defect in env.debug_world.hidden_defects:
        assert defect.defect_id not in obs_json
        assert defect.defect_id not in state_json
        assert defect.defect_type not in obs_json
        assert defect.defect_type not in state_json
