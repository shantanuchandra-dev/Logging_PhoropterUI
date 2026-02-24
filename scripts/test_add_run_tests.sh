#!/usr/bin/env bash
# Test whether run-tests accepts "add" in right_eye/left_eye.
# Run from repo root: bash scripts/test_add_run_tests.sh
# If the backend returns 200 and no error, ADD is supported in run-tests.

BASE_URL="${BASE_URL:-https://rajasthan-royals.preprod.lenskart.com}"
PHOROPTER_ID="${PHOROPTER_ID:-phoropter-1}"

echo "Testing run-tests with ADD in right_eye / left_eye..."
echo "URL: $BASE_URL/phoropter/$PHOROPTER_ID/run-tests"
echo ""

curl -s -w "\nHTTP_STATUS:%{http_code}\n" -X POST "$BASE_URL/phoropter/$PHOROPTER_ID/run-tests" \
  -H "Content-Type: application/json" \
  -d '{
    "test_cases": [
      {
        "case_id": 1,
        "aux_lens": "BINO",
        "right_eye": {"sph": -2.00, "cyl": -1.00, "axis": 90, "add": 1.25},
        "left_eye": {"sph": -1.75, "cyl": -1.00, "axis": 180, "add": 1.25}
      }
    ]
  }' | tee /tmp/add_test_response.txt

echo ""
if grep -q "HTTP_STATUS:200" /tmp/add_test_response.txt; then
  echo "Result: 200 OK – run-tests appears to accept ADD."
else
  echo "Result: Non-200 or error – check response above. ADD may not be supported in run-tests."
fi
