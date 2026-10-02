# 🇩🇪 German Language Funding Drive

## Motivation

German is one of the most spoken languages in Europe:
- ~95 million native speakers
- ~85 million second-language speakers

Despite this high demand, there are no suitable openweight TTS alternatives with German support that run efficiently on CPU.

## Why Funding is Needed

Training a high-quality German voice requires:
- Carefully curated training data (Common Voice is not viable per developer requirements)
- Significant effort to gather and prepare appropriate audio samples
- GPU training resources

Rather than spending time gathering data inefficiently, funding allows @hexgrad to focus on producing quality training data and training the model properly.

## Comparison with Alternatives

| Model | Hardware | German Support | CPU Viability |
|-------|----------|----------------|---------------|
| Kokoro 82M v1.0 | ~tens of MB, CPU/GPU | ❌ No (in development) | ✅ High |
| Fish Speech 1.5 | 10-12 GB VRAM | ❌ Unclear | ⚠️ Low |
| Chatterbox | ~8 GB VRAM | ✅ Yes | ⚠️ Medium |
| XTTS v2 | 8+ GB VRAM | ✅ Yes | ⚠️ Low-Medium |

Most suitable alternatives either require expensive GPU hardware or have unclear German support.

## Current Progress

Multiple GitHub issues have been created requesting German support:
- [Issue #232](https://github.com/hexgrad/kokoro/issues/232)
- [Issue #186](https://github.com/hexgrad/kokoro/issues/186)

## How to Support

### Primary Method: Opire Funding Platform
Visit: **https://app.opire.dev/issues/01KCKZCP14P37WZQZQA9W6BGJJ**

1. Login with your GitHub account
2. Choose your payment method (Stripe, PayPal, etc.)
3. Complete your donation

### Payment Options
- **Stripe**: VISA, SEPA Direct Debit, PayPal
- **PayPal**: Direct payment without Opire account (contact developer)

⚠️ **Important**: Use a normal Stripe account, not Stripe Express (for business accounts).

### Contact
For questions about funding or German voice development, please comment on the relevant GitHub issues.

## Alternative Funding Platforms

If Opire doesn't work for you, other platforms considered:
- polar.sh
- Algora

You can suggest adding these in comments on the funding issue.
