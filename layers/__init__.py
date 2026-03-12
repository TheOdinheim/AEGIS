"""
AEGIS Defense Layers — Eight-layer immune system architecture.

ASSUMED-BREACH POSTURE: Each layer in this package operates under the
assumption that every other layer — upstream and downstream — has been
fully compromised. L2 Innate does not trust that L1 Barrier validated
the schema. L5 Output does not trust that L3 Adaptive cleared the request.
L6 Policy does not trust any individual layer's verdict. This is not
paranoia; it is the fundamental design contract of the immune system.

Layers:
    L1 barrier.py          — Skin / Mucous Membranes
    L2 innate/             — Pattern Recognition Receptors / NK Cells
    L3 adaptive/           — T-Cells / B-Cells / Clonal Selection
    L4 memory/             — Memory B-Cells / Antibody Library
    L5 output/             — Complement System Cascade
    L6 policy.py           — Regulatory T-Cells / Immune Tolerance
    L7 healing.py          — Wound Healing / Tissue Repair
    L8 (production only)   — Herd Immunity / Federated Intelligence
"""
