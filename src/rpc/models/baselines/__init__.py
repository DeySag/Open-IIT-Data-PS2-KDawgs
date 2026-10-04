"""Baselines: incumbent rule, account GBM, contact-point GBM.

Leakage contract: NOTHING in this package may reference ground truth or the
policy log (enforced by tests/test_eval.py). Features come only from the
(eval-supplied) point-in-time feature frame.
"""

from src.rpc.eval.registry import register_scorer
from src.rpc.models.baselines.account_gbm import AccountGBMScorer
from src.rpc.models.baselines.contact_gbm import ContactGBMScorer
from src.rpc.models.baselines.incumbent import IncumbentRuleScorer

__all__ = ["IncumbentRuleScorer", "AccountGBMScorer", "ContactGBMScorer"]


def _register() -> None:
    register_scorer("incumbent", IncumbentRuleScorer)
    register_scorer("account_gbm", AccountGBMScorer)
    register_scorer("contact_gbm", ContactGBMScorer)


_register()
