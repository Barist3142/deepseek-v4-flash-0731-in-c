# Runs the DeepSeek tokenizer roundtrip when the released tokenizer.json is available,
# and prints a line ctest recognises as a skip when it is not.
#
# tokenizer.json has 128,000 entries and ships with the checkpoint, not with this
# repository, so this gate cannot run on a clean checkout. It is registered anyway:
# a check that vanishes from the list when it cannot run reads as "one fewer test to
# care about", and ctest's count silently stops matching what `make test` runs.
#
# DSV4_MODEL_DIR is read here, at TEST time rather than configure time, so
# `DSV4_MODEL_DIR=/path/to/model ctest` works without reconfiguring the build.

set(MODEL_DIR "$ENV{DSV4_MODEL_DIR}")

if(MODEL_DIR STREQUAL "")
  message("SKIPPED: DSV4_MODEL_DIR is not set. The vocabulary ships with the checkpoint;"
          " set DSV4_MODEL_DIR to a model directory to run this gate.")
  return()
endif()

if(NOT EXISTS "${MODEL_DIR}/tokenizer.json")
  message("SKIPPED: no tokenizer.json in ${MODEL_DIR}")
  return()
endif()

execute_process(COMMAND "${TEST_TOK}" "${MODEL_DIR}"
                RESULT_VARIABLE rc)
if(NOT rc EQUAL 0)
  message(FATAL_ERROR "tokenizer test failed with status ${rc}")
endif()
