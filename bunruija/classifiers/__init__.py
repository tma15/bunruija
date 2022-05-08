import lightgbm  # type: ignore
from sklearn.svm import LinearSVC, SVC  # type: ignore
from sklearn.ensemble import (  # type: ignore
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression  # type: ignore

from ..registry import BUNRUIJA_REGISTRY
from .lstm import LSTMClassifier
from .prado import PRADO
from .qrnn import QRNN
from .transformer import TransformerClassifier


BUNRUIJA_REGISTRY["lgbm"] = lightgbm.LGBMClassifier
BUNRUIJA_REGISTRY["linear_svm"] = LinearSVC
BUNRUIJA_REGISTRY["lr"] = LogisticRegression
BUNRUIJA_REGISTRY["lstm"] = LSTMClassifier
BUNRUIJA_REGISTRY["prado"] = PRADO
BUNRUIJA_REGISTRY["qrnn"] = QRNN
BUNRUIJA_REGISTRY["random_forest"] = RandomForestClassifier
BUNRUIJA_REGISTRY["svm"] = SVC
BUNRUIJA_REGISTRY["stacking"] = StackingClassifier
BUNRUIJA_REGISTRY["transformer"] = TransformerClassifier
BUNRUIJA_REGISTRY["voting"] = VotingClassifier
