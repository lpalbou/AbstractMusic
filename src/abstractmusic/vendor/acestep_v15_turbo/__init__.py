"""
Vendored ACE-Step v1.5 turbo Transformers model code.

Source:
- Hugging Face model repo: `ACE-Step/Ace-Step1.5`
- Revision: `19671f406d603126926c1b7e2adc169acbcade22`
- Files:
  - `acestep-v15-turbo/configuration_acestep_v15.py`
  - `acestep-v15-turbo/modeling_acestep_v15_turbo.py`

License:
- Apache-2.0 (see headers in the vendored files)

We vendor these files because Transformers `trust_remote_code` only searches the
repo root for custom modules, but ACE-Step ships its custom modules inside the
checkpoint subfolder. Vendoring avoids runtime remote-code execution and keeps
the integration self-contained.
"""

