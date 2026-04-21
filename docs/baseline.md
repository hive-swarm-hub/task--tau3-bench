# Baseline — gpt-5.2 + terminal_use + reasoning_effort=high

## Overall

| Metric | Value |
|---|---|
| Pass^1 (single trial) | **24/97 (24.7%)** |
| DB Match | 26/97 (26.8%) |
| Write actions matched | 91/479 (19.0%) |
| Cost | ~$97 |
| Avg per simulation | ~$1.00 |
| Runtime | ~45min at concurrency=8 |

## Full eval — passing tasks (24)

```
task_001, task_002, task_004, task_005, task_006, task_014, task_015,
task_016, task_017, task_019, task_023, task_024, task_025, task_028,
task_032, task_033, task_034, task_035, task_050, task_055, task_072,
task_073, task_089, task_101
```

## Lite eval — 10/20 passing

The curated lite subset (`EVAL_LITE=1`) covers 7 failure clusters. At baseline:

| Cluster | Pass count | Tasks |
|---|---|---|
| canary | 2/4 | task_001 ✓, task_004 ✓, task_007 ✗, task_076 ✗ |
| playbook_trap | 1/1 | task_033 ✓ |
| dispute_calculator | 1/5 | task_017 ✓, task_018 ✗, task_021 ✗, task_026 ✗, task_040 ✗ |
| execution_discipline | 0/3 | task_036 ✗, task_087 ✗, task_100 ✗ |
| variance_band | 2/3 | task_006 ✓, task_016 ✓, task_035 ✓ |
| escalation | 1/2 | task_005 ✓, task_091 ✗ |
| recently_flipped | 2/2 | task_019 ✓, task_024 ✓ |

(Note: `variance_band` passed 3/3 in this snapshot but the tagged tasks occasionally vary — variance_band tasks are specifically chosen because they flip between runs.)

## Failure class distribution (73 failing tasks)

| Class | Count |
|---|---|
| priority_1_verification_or_unlock | 69 |
| priority_4_execution_discipline | 2 |
| priority_2_wrong_arguments | 2 |

P1 dominates at 94%. See `docs/failure_patterns.md` for diagnostic signals per class.

## Termination reasons (failures)

| Reason | Count |
|---|---|
| user_stop | 55 |
| max_steps (budget exhausted) | 18 |

## Behavioral stats

- **Passing tasks:** avg 33 shell calls, 91 turns
- **Failing tasks:** avg 56 shell calls, 146 turns

Over-searching correlates with failure, not success.

## Reproduction

```bash
bash prepare.sh
bash eval/eval.sh > run.log 2>&1
grep "^accuracy:" run.log
```

Expected: `accuracy: ~0.23 to 0.27` (±2 task variance between runs on identical code is normal — OpenAI `system_fingerprint` drift at temp=0).
