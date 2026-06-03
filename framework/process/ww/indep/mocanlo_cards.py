"""Write a MoCaNLO NLO-EW card set for one independent-WW grid point.

Templated from the validated card set in
``mocanlo/validate/ww_nloew/cards/`` (e+e- → 4f, NLO QCD⁰+EW¹, complex-mass
scheme, ``pdf_set=none`` — no beam ISR, so ``cms_beam_energy`` IS √ŝ).  One
4-run job (born/virt/real/idip) gives both σ̂_Born (run 1) and σ̂_NLO (Σ runs).

A grid point is fixed by (channel block, m_W, Γ_W, √ŝ, EW α-scheme).  Fixed SM
inputs and scales default to the validated values; the renormalisation /
factorisation / IR scales are pinned to a reference (m_W^ref = 80.379), NOT to
the varied m_W, so the POI variations don't drag a spurious scale dependence.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

from framework.process.ww.indep.channels import ChannelBlock


@dataclass(frozen=True)
class SMInputs:
    """Fixed SM inputs, synced to the WW production card
    (cards/ww_default.py PARAM_INPUTS) so the independent calc and the BFS
    chain share m_t / M_H / M_Z / G_F. NB the ISR β_e α is a separate, fit-time
    knob (isr_beta), intentionally α_Gμ and NOT set here."""
    mZ: float = 91.1876               # card M_Z
    gZ: float = 2.4952
    mt: float = 172.5                 # card m_t (OS pole)
    gTop: float = 1.3448
    mH: float = 125.25                # card M_H
    gH: float = 4.07e-3
    fermi_constant: float = 1.1663787e-5   # card G_F
    scheme_alpha: str = "gf"          # gf | alpha0 | alphaz | alphamsbar
    mu_ref: float = 80.379            # renorm / fact / IR scale [GeV]
    n_loops: int = 2


@dataclass(frozen=True)
class IntegrationSettings:
    """MC integration termination knobs."""
    n_target_accepted: int = 20000
    target_rel_precision_pct: float = 2.0   # MoCaNLO 'target_relative_precision' (%)
    time_wall: str = "0-02:00"              # D-HH:MM per run
    dipole_alpha: float = 1.0e-2


def _proc_card(block: ChannelBlock) -> str:
    return f"""<subprocesses>
\t<subprocess id="ww">
\t\t<partonic_process>
\t\t\t<incoming> e+ e- </incoming>
\t\t\t<outgoing> {block.outgoing} </outgoing>
\t\t\t<pdf1_codes> -2 </pdf1_codes>
\t\t\t<pdf2_codes>  2 </pdf2_codes>
\t\t\t<tree_qcd_order> 0 </tree_qcd_order>
\t\t\t<loop_qcd_order> 0 </loop_qcd_order>
\t\t</partonic_process>
\t\t<real_process>
\t\t\t<incoming> e+ e- </incoming>
\t\t\t<outgoing> {block.outgoing_real} </outgoing>
\t\t\t<pdf1_codes> -2 </pdf1_codes>
\t\t\t<pdf2_codes>  2 </pdf2_codes>
\t\t\t<tree_qcd_order> 0 </tree_qcd_order>
\t\t</real_process>
\t</subprocess>
</subprocesses>
<resonances id="none">
</resonances>
"""


_RUN_TYPES = (("1", "born"), ("2", "virt"), ("3", "real"), ("4", "idip"))


def _run_card(ecm: float, integ: IntegrationSettings) -> str:
    runs = []
    for rid, rtype in _RUN_TYPES:
        runs.append(f"""\t<run id="{rid}">
\t\t<type>            {rtype}     </type>
\t\t<subdirectory>    ww/{rtype}  </subdirectory>
\t\t<subprocess       id="ww"/>
\t\t<run_parameters   id="ww"/>
\t\t<model_parameters id="ww"/>
\t\t<cuts             id="nocut"/>
\t\t<recombinations   id="nocut"/>
\t\t<resonances       id="none"/>
\t\t<histograms       id="none"/>
\t</run>""")
    runs_block = "\n".join(runs)
    n_acc = f"{integ.n_target_accepted:,}".replace(",", " ")
    return f"""<runs directory="out">
{runs_block}
</runs>
"""


def _param_card(block, mW: float, gW: float, ecm: float,
                sm: SMInputs, integ: IntegrationSettings) -> str:
    n_acc = f"{integ.n_target_accepted:,}".replace(",", " ")
    return f"""<model_parameters id="ww">
\t<ons_z0_mass>    {sm.mZ:.6g}      </ons_z0_mass>
\t<ons_z0_width>   {sm.gZ:.6g}      </ons_z0_width>
\t<ons_w_mass>     {mW:.6f}      </ons_w_mass>
\t<ons_w_width>    {gW:.6f}       </ons_w_width>
\t<top_mass>       {sm.mt:.6g}       </top_mass>
\t<top_width>      {sm.gTop:.6g}      </top_width>
\t<higgs_mass>     {sm.mH:.6g}      </higgs_mass>
\t<higgs_width>    {sm.gH:.6g}    </higgs_width>
\t<fermi_constant> {sm.fermi_constant:.6g}  </fermi_constant>
\t<scheme_alpha>   {sm.scheme_alpha}          </scheme_alpha>
\t<renormalization_scale> {sm.mu_ref:.6g} </renormalization_scale>
\t<factorization_scale>   {sm.mu_ref:.6g} </factorization_scale>
\t<infrared_scale>        {sm.mu_ref:.6g} </infrared_scale>
\t<n_loops>        {sm.n_loops}           </n_loops>
</model_parameters>

<run_parameters id="ww">
\t<output_level> 4 </output_level>
\t<cms_beam_energy> {ecm:.6f}d0 </cms_beam_energy>
\t<helicity_ep> 0 </helicity_ep>
\t<helicity_em> 0 </helicity_em>
\t<ignore_e0_check> true </ignore_e0_check>
\t<scale>
\t    <dynamical_scale_type> 0 </dynamical_scale_type>
\t    <scale_factors_f> 1d0 </scale_factors_f>
\t    <scale_factors_r> 1d0 </scale_factors_r>
\t</scale>
\t<pdfs>
\t    <beam1> e- </beam1>
\t    <beam2> e+ </beam2>
\t    <pdf_set> none </pdf_set>
\t</pdfs>
\t<integration_termination>
\t    <n_target_accepted_events> {n_acc} </n_target_accepted_events>
\t    <target_relative_precision> {max(1, int(round(integ.target_rel_precision_pct)))} </target_relative_precision>
\t    <time_wall> {integ.time_wall} </time_wall>
\t</integration_termination>
\t<dipoles>
\t    <alpha_fs_fs> {integ.dipole_alpha:g} </alpha_fs_fs>
\t    <alpha_fs_is> {integ.dipole_alpha:g} </alpha_fs_is>
\t    <alpha_is_fs> {integ.dipole_alpha:g} </alpha_is_fs>
\t    <alpha_is_is> {integ.dipole_alpha:g} </alpha_is_is>
\t</dipoles>
</run_parameters>
"""


_CUT_CARD_NOCUT = "<cuts id=\"nocut\">\n</cuts>\n\n<recombinations id=\"nocut\">\n</recombinations>\n"
_PLOT_CARD = "<histograms id=\"none\">\n</histograms>\n"

#: MoCaNLO default final-state type codes for charged leptons (from the
#: validated e+e- examples: ee_tt uses mu+=16, ta-=19 with NO dressing; vbs-zz
#: uses e+=4, e-=5, mu+=16, mu-=17).  Bare leptons get these codes directly, so
#: a theta_angle cut can target them without any recombination step.
LEPTON_CODE = {"e-": 5, "e+": 4, "mu-": 17, "mu+": 16}


def _lepton_tokens(outgoing: str) -> list[str]:
    """Charged-lepton tokens present in a block's outgoing string."""
    return [tok for tok in outgoing.split() if tok in LEPTON_CODE]


#: Same-flavour opposite-sign charged-lepton pairs — the γ*→ℓℓ low-mass NC
#: pole lives in their invariant mass, so an m_ℓℓ cut is the physical tool.
_SF_OS_PAIRS = (("e-", "e+"), ("mu-", "mu+"))


def _sf_os_pairs(outgoing: str) -> list[tuple[str, str]]:
    """Same-flavour OS lepton pairs present in a block (e⁻e⁺, μ⁻μ⁺)."""
    toks = set(_lepton_tokens(outgoing))
    return [(a, b) for (a, b) in _SF_OS_PAIRS if a in toks and b in toks]


def _cut_card(block: ChannelBlock, cos_theta_max: float | None,
              pt_min: float | None = None, mll_min: float | None = None) -> str:
    """cut_card.xml.  Fiducial selection on the charged leptons:
    ``cos_theta_max`` → |cosθ_l| < cos_theta_max (removes forward t-channel
    single-W); ``pt_min`` → p_T,l > pt_min GeV (detector floor); ``mll_min`` →
    m_ℓℓ > mll_min GeV on every same-flavour OS pair (the physical tool for the
    γ*→ℓℓ low-mass NC pole — only the same-flavour leptonic blocks have such a
    pair).  No cut at all when all three are None."""
    if cos_theta_max is None and pt_min is None and mll_min is None:
        return _CUT_CARD_NOCUT
    cuts = []
    for tok in _lepton_tokens(block.outgoing):
        code = LEPTON_CODE[tok]
        if cos_theta_max is not None:
            theta_min = math.degrees(math.acos(cos_theta_max))   # 18.19° for 0.95
            cuts.append(f"""\t<cut type="theta_angle">
\t\t<name>{tok}_theta_cut</name>
\t\t<jet_type>{code}</jet_type>
\t\t<min_value>{theta_min:.4f}</min_value>
\t\t<max_value>{180.0 - theta_min:.4f}</max_value>
\t\t<n_required>1</n_required>
\t</cut>""")
        if pt_min is not None:
            cuts.append(f"""\t<cut type="transverse_momentum">
\t\t<name>{tok}_pt_cut</name>
\t\t<jet_type>{code}</jet_type>
\t\t<min_value>{pt_min:.4f}</min_value>
\t\t<n_required>1</n_required>
\t</cut>""")
    if mll_min is not None:
        for a, b in _sf_os_pairs(block.outgoing):
            ca, cb = LEPTON_CODE[a], LEPTON_CODE[b]
            cuts.append(f"""\t<cut type="invariant_mass">
\t\t<name>{a}{b}_mll_cut</name>
\t\t<jet_type>{ca} {cb}</jet_type>
\t\t<jet_tag>0 0</jet_tag>
\t\t<target>0</target>
\t\t<action>-1</action>
\t\t<min_value>{mll_min:.4f}</min_value>
\t\t<max_value>1000000.0000</max_value>
\t</cut>""")
    body = "\n".join(cuts)
    return (f'<cuts id="nocut">\n{body}\n</cuts>\n\n'
            f'<recombinations id="nocut">\n</recombinations>\n')


def write_cards(procdir: str, block: ChannelBlock, mW: float, gW: float,
                ecm: float, sm: SMInputs = SMInputs(),
                integ: IntegrationSettings = IntegrationSettings(),
                cos_theta_max: float | None = None,
                pt_min: float | None = None,
                mll_min: float | None = None) -> str:
    """Write the 5 MoCaNLO cards into ``<procdir>/cards/``; return ``procdir``.

    ``cos_theta_max`` (e.g. 0.95) + ``pt_min`` (e.g. 10.0) + ``mll_min`` (e.g.
    10.0, same-flavour OS pairs) enable the fiducial charged-lepton cuts; all
    ``None`` (default) is the inclusive configuration.
    """
    cards_dir = os.path.join(procdir, "cards")
    os.makedirs(cards_dir, exist_ok=True)
    files = {
        "proc_card.xml": _proc_card(block),
        "run_card.xml": _run_card(ecm, integ),
        "param_card.xml": _param_card(block, mW, gW, ecm, sm, integ),
        "cut_card.xml": _cut_card(block, cos_theta_max, pt_min, mll_min),
        "plot_card.xml": _PLOT_CARD,
    }
    for name, content in files.items():
        with open(os.path.join(cards_dir, name), "w") as fh:
            fh.write(content)
    return procdir
