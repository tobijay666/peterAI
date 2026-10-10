"""Unit test for doc sync and drift checking (Phase I4)."""
import pytest
from scripts.gen_docs import main as gen_docs_main
from scripts.check_doc_drift import check_doc_drift

pytestmark = pytest.mark.docs


def test_doc_sync_no_drift():
    """Ensure generated reference docs match code and config."""
    # First generate fresh docs to ensure docs/generated exists
    gen_docs_main()
    
    # Assert no drift
    assert check_doc_drift() is True, "Generated docs in docs/generated/ must be in sync with code/config"
