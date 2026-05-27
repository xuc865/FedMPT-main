from model.dualcoop import DualCoop
from model.FedTPG import FedTPG
from model.custom_coop import CoOpCLIP
from model.custom_vlp import VLPCLIP
from model.positivecoop import PositiveCoop
from model.scpnet import SCPNet
try:
    from model.tcp import TCPCLIP
except ModuleNotFoundError as e:
    _tcp_err = e

    class TCPCLIP:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError(
                "TCPCLIP is unavailable because a required dependency is missing. "
                f"Original error: {_tcp_err}"
            )

try:
    from model.fedpgp import FedPGPCLIP
except ModuleNotFoundError as e:
    _fedpgp_err = e

    class FedPGPCLIP:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError(
                "FedPGPCLIP is unavailable because a required dependency is missing. "
                f"Original error: {_fedpgp_err}"
            )
from model.fedawa import FedAWACLIP
from model.fedmvp import FedMVPCLIP
from model.fedmpt import FedMPTCLIP
from model.fedram import FedRAMCLIP

# Maple depends on Dassl; make it optional so `import model` works
# even when Dassl isn't installed (useful for FedMPT-only runs).
try:
    from model.maple import MapleCLIP
except ModuleNotFoundError as e:
    _maple_err = e

    class MapleCLIP:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError(
                "MapleCLIP is unavailable because a required dependency is missing. "
                f"Original error: {_maple_err}"
            )