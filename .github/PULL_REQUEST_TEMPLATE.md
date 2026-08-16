## What this changes

<!-- One or two sentences. What behaviour is different after this PR? -->

## Why

<!-- The problem being solved. Link an issue if there is one. -->

## Verification

- [ ] `make test` passes (all weightless gates)
- [ ] `make strict`, `make portable`, and `make asan` pass
- [ ] If kernels changed: tiny graph reports `maxdiff=0.000000`
- [ ] If tokenizer changed: the 9,384-sample parity suite passes
- [ ] If full output could change: the published 16-token oracle still matches exactly

## Numbers, if this is a performance change

<!-- Timing claims need a noise floor. Report at least 3 runs of each arm, or say
     explicitly that the effect was not measured against variance. Include
     TTFT, TPOT, memory plan, threads, storage and power state. See
     docs/PERFORMANCE_AUDIT.md. -->

| arm | run 1 | run 2 | run 3 | mean |
|-----|-------|-------|-------|------|
|     |       |       |       |      |

## Risk

<!-- What could this break that the tests would not catch? -->
