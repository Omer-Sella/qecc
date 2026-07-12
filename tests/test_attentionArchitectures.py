import math
import torch
from qecc.attentionArchitectures import (
    TOKEN_FEATURE_SIZE, buildTokenFeatures, CodeEncoder, AttentionPool,
)

POSITIONAL_SLICE = slice(5, 12)  # 6 harmonics + linear position


def makeBits(l, m, batchSize=2):
    generator = torch.Generator().manual_seed(0)
    return torch.randint(0, 2, (batchSize, 2 * l + 2 * m), generator=generator).float()


def test_tokenShapeAndWidth():
    bits = makeBits(6, 6)
    tokens = buildTokenFeatures(bits, 6, 6)
    assert tokens.shape == (2, 24, TOKEN_FEATURE_SIZE)


def test_bitAndGroupColumns():
    l = m = 6
    bits = makeBits(l, m)
    tokens = buildTokenFeatures(bits, l, m)
    torch.testing.assert_close(tokens[:, :, 0], bits)          # column 0 = bit value
    # group one-hot: token 0 is aX (group 0), token l is aY (group 1),
    # token l+m is bX (group 2), token 2l+m is bY (group 3)
    assert tokens[0, 0, 1] == 1.0 and tokens[0, 0, 2] == 0.0
    assert tokens[0, l, 2] == 1.0
    assert tokens[0, l + m, 3] == 1.0
    assert tokens[0, 2 * l + m, 4] == 1.0


def test_cyclicPositionIsFractionOfPeriod():
    # slot 3 of aX at l=6 and slot 6 of aX at l=12 share angle pi -> identical positional features
    tokensSix = buildTokenFeatures(torch.zeros(1, 24), 6, 6)
    tokensTwelve = buildTokenFeatures(torch.zeros(1, 48), 12, 12)
    torch.testing.assert_close(tokensSix[0, 3, POSITIONAL_SLICE],
                               tokensTwelve[0, 6, POSITIONAL_SLICE])


def test_globalsColumns():
    # l != m so a log(l)/log(m) column swap cannot go unnoticed
    tokens = buildTokenFeatures(torch.zeros(1, 2 * 6 + 2 * 12), 6, 12)
    torch.testing.assert_close(tokens[0, 0, 12], torch.tensor(math.log(6.0)))
    torch.testing.assert_close(tokens[0, 0, 13], torch.tensor(math.log(12.0)))
    torch.testing.assert_close(tokens[0, 0, 14], torch.tensor(1.0))  # reserved/constant


def test_encoderAndPoolAreSizeAgnostic():
    encoder = CodeEncoder()
    pool = AttentionPool(64)
    for l, m in ((6, 6), (12, 12)):
        bits = makeBits(l, m)
        tokens = buildTokenFeatures(bits, l, m)
        pooled = pool(encoder(tokens))
        assert pooled.shape == (2, 64)


def test_stateDictLoadsStrictAcrossInstances():
    source, target = CodeEncoder(), CodeEncoder()
    target.load_state_dict(source.state_dict(), strict=True)
