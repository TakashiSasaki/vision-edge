"""Require a pinned, completed baseline experiment; a FAIL baseline stays FAIL."""
from acquire_reference import CONFIG_IDENTITY
from verify_checkpoint import ARTIFACTS, REVISION


def validate(report):
    if report.get('completed') is not True or report.get('status') not in ('PASS','FAIL'):
        raise ValueError('incomplete baseline')
    if report.get('checkpoint_revision')!=REVISION:
        raise ValueError('baseline revision mismatch')
    for key,expected in [('checkpoint',ARTIFACTS['mirage.safetensors']),('cact',ARTIFACTS['mirage.cact']),('config',CONFIG_IDENTITY)]:
        if report.get(key)!=dict(zip(('size','sha256'),expected)):
            raise ValueError(f'baseline {key} identity mismatch')
    if report.get('weights_mode')!='cact-dequantized' or report.get('padding')!='symmetric' or report.get('provider')!=['CPUExecutionProvider']:
        raise ValueError('baseline weights/padding/provider mismatch')
    return report['onnx']
