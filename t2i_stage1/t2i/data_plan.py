"""Data-plan boundaries; the paper recipe and update path stay unchanged."""
CC12M_ONLY = 'cc12m-only'
PAPER = 'paper-two-stage'


def plan_name(cc12m_only):
    return CC12M_ONLY if cc12m_only else PAPER


def plan_steps(recipe, cc12m_only):
    return recipe.stage_a_steps + (0 if cc12m_only else recipe.stage_b_steps)


def valid_steps(steps, smoke, recipe, cc12m_only):
    maximum = plan_steps(recipe, cc12m_only)
    return 0 < steps <= maximum and (smoke or steps == maximum)


def stage_b_admission(parent, binding, step):
    """Only a completed A-only run may add B in a new output directory."""
    previous = parent['run_identity']['binding']
    same = {key for key in previous.keys() | binding.keys()
            if key not in ('data_plan', 'data')}
    checks = dict(parent_plan=previous.get('data_plan') == CC12M_ONLY,
                  destination_plan=binding.get('data_plan') == PAPER,
                  parent_step=step == 250000 and parent.get('step') == 250000,
                  parent_data=set(previous.get('data', {})) == {'A'},
                  destination_data=set(binding.get('data', {})) == {'A', 'B'},
                  a_manifest=previous.get('data', {}).get('A') == binding.get('data', {}).get('A'),
                  unchanged_binding=(set(previous) - {'data_plan', 'data'} ==
                                     set(binding) - {'data_plan', 'data'} and
                                     all(previous[key] == binding[key] for key in same)))
    return dict(gate='stage_b_continuation', parent_step=step,
                metadata_step=parent.get('step'), required_step=250000,
                parent_binding=previous, destination_binding=binding,
                checks=checks, passed=all(checks.values()))


def plan_mismatch_message(binding):
    value = '1' if binding.get('data_plan') == CC12M_ONLY else '0'
    return ('This run was started with CC12M_ONLY=' + value +
            '; set CC12M_ONLY=' + value + ' again. To continue completed Stage A '
            'into Stage B, use a new output directory (OUTPUT_DIR), CC12M_ONLY=0 and '
            'RESUME pointing to checkpoint_00250000.')
