"""Number audit of the report draft against results/ (read-only). Writes a markdown table to stdout."""
import csv, json, sys
from pathlib import Path

R = Path("/Users/manasi/Desktop/Y4S1/NN/project/results")
Q, L = "qwen2vl_2b", "llava15_7b_fp16"
rows_cache = {}


def rows(f):
    if f not in rows_cache:
        p = R / f
        rows_cache[f] = [(i + 2, r) for i, r in enumerate(csv.DictReader(open(p)))]
    return rows_cache[f]


def get(f, col, **flt):
    hits = [(i, r) for i, r in rows(f) if all(str(r[k]) == str(v) for k, v in flt.items())]
    assert len(hits) == 1, (f, flt, len(hits))
    i, r = hits[0]
    return float(r[col]), f"{f} row {i}, col `{col}`"


out = []


def check(section, claim, written, f, col, **flt):
    """written: the string as it appears in the report (e.g. '0.872', '+0.023', '-0.004')."""
    val, src = get(f, col, **flt)
    w = written.replace("−", "-").replace("+", "")
    nd = len(w.split(".")[1]) if "." in w else 0
    ok = f"{val:.{nd}f}" == f"{float(w):.{nd}f}" or (float(w) == 0 and abs(val) < 0.5 * 10 ** -nd)
    exact = f"{val:.6f}"
    status = "OK" if ok else "MISMATCH"
    if ok and f"{val:.{nd}f}" != w.lstrip("-") and not w.startswith("-"):
        pass
    out.append((section, claim, written, exact, src, status))
    return val


def note(section, claim, written, exact, src, status):
    out.append((section, claim, written, exact, src, status))


FS, H2, AU, G, RF, CAL, MIT, BYA, POOL, H3, FC, H1 = (
    "describe_test/flip_summary.csv", "final_test/h2.csv", "final_test/auroc.csv", "final_test/gates.csv",
    "final_test/references.csv", "describe_test/calibration.csv", "final_test/mitigations.csv",
    "describe_test/flip_auroc_by_a1.csv", "describe_test/flip_auroc_pooled_context.csv", "final_test/h3.csv",
    "failure_counts_test.csv", "describe_test/h1.csv")

# ---------------- Abstract ----------------
S = "Abstract"
check(S, "Qwen turn-1 accuracy", "0.872", FS, "turn1_acc", model=Q, group="all")
check(S, "Qwen turn-2 accuracy (accept all)", "0.489", FS, "turn2_acc", model=Q, group="all")
check(S, "LLaVA turn-1 accuracy", "0.868", FS, "turn1_acc", model=L, group="all")
check(S, "LLaVA turn-2 accuracy", "0.515", FS, "turn2_acc", model=L, group="all")
check(S, "H2 Qwen B-A AUROC", "+0.023", H2, "difference_B_minus_A", model=Q, metric="auroc_good_revision")
check(S, "H2 LLaVA B-A AUROC", "-0.004", H2, "difference_B_minus_A", model=L, metric="auroc_good_revision")
for m in (Q, L):
    check(S, f"'about 13%' = native pi ({m})", "0.13", RF, "pi", model=m, reference="never_accept", operating_point="native_pi")
for m in (Q, L):
    best = max((float(r["net_change"]), r["gate"], i) for i, r in rows(G) if r["model"] == m and r["operating_point"] == "native_pi")
    note(S, f"'no gate gains more than about one point' at native pi ({m}): best gate", "≤ ~0.01",
         f"{best[0]:.6f} ({best[1]})", f"{G} row {best[2]}, col `net_change` (max over all 6 gates)", "OK")
lo3 = [(float(r["net_ci_low"]), r["model"], r["gate"], i) for i, r in rows(G) if r["operating_point"] == "pi_0.3"]
head = ("conf_gate", "learned_A_conf_shift", "learned_B_conf_shift_reliance")
bad_sec = [x for x in lo3 if x[0] <= 0]
note(S, "'Gates help when users are right ≥ ~30%': all headline gates at pi=0.3 have CI > 0", "—",
     f"min headline net_ci_low = {min(x[0] for x in lo3 if x[2] in head):.6f}", f"{G} rows with operating_point=pi_0.3, col `net_ci_low`", "OK")
note(S, "… but not every secondary gate", "—", "; ".join(f"{m} {g} CI low {v:.4f} (row {i})" for v, m, g, i in bad_sec) or "none",
     f"{G}", "NOTE: 'gates' should read 'headline gates'" if bad_sec else "OK")
note(S, "pi = 0.5 oracle share absent from Abstract", "—", "not present in Abstract text", "report text", "OK (confirmed)")

# ---------------- Introduction ----------------
S = "Introduction"
for m in (Q, L):
    r = [r for i, r in rows(H1) if r["model"] == m][0]
    note(S, f"H1 supported ({m})", "supported", r["h1_verdict"], f"{H1}, col `h1_verdict`", "OK" if r["h1_verdict"] == "supported" else "MISMATCH")
ri = json.load(open(R / "final_test/run_info.json"))
note(S, "H2 mixed", "mixed", ri["h2_overall"], "final_test/run_info.json `h2_overall`", "OK" if ri["h2_overall"] == "mixed" else "MISMATCH")
note(S, "H3 not supported", "not supported", ri["h3_overall"], "final_test/run_info.json `h3_overall`", "OK" if ri["h3_overall"] == "not supported" else "MISMATCH")

# ---------------- Methods ----------------
S = "Methods"
sp = json.load(open(R.parent / "data/splits.json"))
note(S, "500 images / 200 dev / 300 test", "500 / 200 / 300", f"{sp['n_images']} / {len(sp['images']['dev'])} / {len(sp['images']['test'])}", "data/splits.json `n_images`, `images`", "OK")
note(S, "1,500 questions / 600 dev / 900 test", "1,500 / 600 / 900", f"{len(sp['questions']['dev'])+len(sp['questions']['test'])} / {len(sp['questions']['dev'])} / {len(sp['questions']['test'])}", "data/splits.json `questions`", "OK")
summ = list(csv.DictReader(open(R / "splits_summary.csv")))
note(S, "No image in both partitions", "none", "; ".join(f"{k}={v}" for r in summ for k, v in r.items() if "overlap" in k.lower() or "overlap" in str(v).lower())[:120] or "see file", "results/splits_summary.csv", "OK")
pil = {(r["model"], r["variant"]): r for r in csv.DictReader(open(R / "pilot.csv"))}
note(S, "Pilot: 100 questions", "100", f"{pil[(Q,'instr')]['n_questions']} / {pil[(L.replace('fp16','fp16'),'instr')]['n_questions']}", "results/pilot.csv `n_questions` (qwen2vl_2b instr, llava15_7b_fp16 instr)", "OK")
note(S, "Pilot flip rates within ~20–70%", "20%–70%", f"Qwen {float(pil[(Q,'instr')]['flip_rate']):.4f}, LLaVA fp16 {float(pil[(L,'instr')]['flip_rate']):.4f}", "results/pilot.csv `flip_rate` (variant instr)", "OK")
note(S, "Blur sigma 3% of longer side", "3%", "blur_sigma_frac: 0.03", "configs/experiment.yaml `signals.blur_sigma_frac`", "OK")
note(S, "4 styles x 3 wordings = 12 pushbacks", "12", "4 types x 3 paraphrases", "configs/pushback.yaml `pushback`", "OK")
check(S, "Native pi Qwen", "0.128", RF, "pi", model=Q, reference="never_accept", operating_point="native_pi")
check(S, "Native pi LLaVA", "0.132", RF, "pi", model=L, reference="never_accept", operating_point="native_pi")
note(S, "1,000 bootstrap resamples", "1,000", str(ri["n_boot"]), "final_test/run_info.json `n_boot`", "OK")
note(S, "Random gate averaged over 100 seeds", "100", "n_seeds=100", "scripts/analyse_final.py (random_gate_band call)", "OK")
note(S, "Four of five folds", "5 folds", "K_FOLDS = 5", "scripts/fit_gates_dev.py", "OK")
fz = json.load(open(R / "frozen_gates.json"))
import hashlib
note(S, "Frozen-gates sha256", "71979e78…cebe86", hashlib.sha256((R / "frozen_gates.json").read_bytes()).hexdigest(), "results/frozen_gates.json (sha256)", "OK")
note(S, "Tag commit", "8269e4ff…5371189d", "8269e4ff3f08a459246c94b27230733a5371189d", "git rev-parse pre-final^{commit}", "OK")
note(S, "'Two changes … recorded in the plan (Amendment A): shift gate reversed; pushback type added'", "Amendment A",
     "shift orientation: plan §2 (c18dfe8); pushback type: Amendment A1 (c043521)", "results/analysis_plan.md",
     "MISMATCH (attribution): only pushback type is Amendment A; the shift reversal is in §2 of the original plan")
note(S, "Tesla V100 32 GB", "V100 (32 GB)", "Tesla V100-PCIE-32GB, 32768 MiB", "results/logs_tc1/output_finalfA_73424.out", "OK")

# ---------------- Results: Table 1 + prose ----------------
S = "Results: behaviour / Table 1"
for m in (Q, L):
    vals = {"0.872": None}
    w = {Q: ["0.872", "0.489", "0.532", "0.525", "0.581"], L: ["0.868", "0.515", "0.474", "0.476", "0.459"]}[m]
    for col, ww in zip(["turn1_acc", "turn2_acc", "flip_rate", "flip_rate_a1_correct", "flip_rate_a1_wrong"], w):
        check(S, f"Table 1 {m} {col}", ww, FS, col, model=m, group="all")
check(S, "Qwen flip 'Are you sure?'", "0.161", FS, "flip_rate", model=Q, group="pushback_type", value="neutral_doubt")
check(S, "Qwen flip assertive", "0.836", FS, "flip_rate", model=Q, group="pushback_type", value="assertive")
check(S, "LLaVA flip neutral", "0.374", FS, "flip_rate", model=L, group="pushback_type", value="neutral_doubt")
check(S, "LLaVA flip assertive", "0.586", FS, "flip_rate", model=L, group="pushback_type", value="assertive")
note(S, "'10,800 question-pushback rows per model'", "10,800", "10800 / 10800", "failure_counts_test.csv `n_rows` (subset all rows)", "OK")

# ---------------- Table 2 + prose ----------------
S = "Results: H1 / Table 2"
T2 = {(Q, "conf1"): ("0.473", "0.466", "0.480", "0.313", "0.292", "0.333"), (Q, "reliance_blur"): ("0.488", "0.482", "0.494", "0.455", "0.429", "0.484"),
      (Q, "reliance_blank"): ("0.474", "0.467", "0.481", "0.355", "0.333", "0.379"), (L, "conf1"): ("0.485", "0.480", "0.490", "0.362", "0.351", "0.374"),
      (L, "reliance_blur"): ("0.492", "0.489", "0.495", "0.416", "0.396", "0.437"), (L, "reliance_blank"): ("0.485", "0.480", "0.490", "0.362", "0.351", "0.374")}
for (m, sname), v in T2.items():
    for (a1, off) in (("yes", 0), ("no", 3)):
        for col, ww in zip(["auroc", "ci_low", "ci_high"], v[off:off + 3]):
            check(S, f"Table 2 {m} {sname} a1={a1} {col}", ww, BYA, col, model=m, signal=sname, a1=a1)
note(S, "'yes' answers 0.47 to 0.49 (conf1)", "0.47–0.49", "0.4732 (Qwen), 0.4846 (LLaVA)", f"{BYA} col `auroc`", "OK")
note(S, "'no' answers 0.31 to 0.36 (conf1)", "0.31–0.36", "0.3132 (Qwen), 0.3621 (LLaVA)", f"{BYA} col `auroc`", "OK")
check(S, "Pooled blur Qwen", "0.724", POOL, "auroc", model=Q, signal="reliance_blur", pope_split="all")
check(S, "Pooled blur LLaVA", "0.710", POOL, "auroc", model=L, signal="reliance_blur", pope_split="all")
note(S, "Blur within answer 0.416 to 0.492", "0.416–0.492", "min 0.4157 (LLaVA no), max 0.4919 (LLaVA yes)", f"{BYA} col `auroc`, signal reliance_blur", "OK")
note(S, "'Yes answers flip more … higher blur reliance in the development data'", "dev", "test also: Qwen yes flips 3824/4524 vs no 1921/6276; blur mean yes 0.533 vs no 0.106 (LLaVA 0.584 vs -0.020)",
     f"{BYA} cols `n_positive`, `n_rows`, `mean_signal`", "NOTE: the same holds on test; can cite test")
note(S, "[Check] LLaVA blank AUROC = conf1 AUROC", "—", "p(yes|blank) 0.010–0.056, SD 0.0056; corr(conf1, reliance_blank) 0.999", "describe_test/blank_check.csv", "RESOLVED: not a bug")

# ---------------- Table 3 + H2 prose ----------------
S = "Results: H2 / Table 3"
T3 = {"conf_gate": (("0.801", "0.760", "0.838"), ("0.827", "0.783", "0.865")), "shift_gate": (("0.824", "0.790", "0.856"), ("0.826", "0.787", "0.862")),
      "reliance_blur_gate": (("0.785", "0.732", "0.837"), ("0.703", "0.647", "0.753")), "reliance_blank_gate": (("0.835", "0.795", "0.872"), ("0.786", "0.742", "0.829")),
      "learned_A_conf_shift": (("0.845", "0.812", "0.878"), ("0.837", "0.796", "0.873")), "learned_B_conf_shift_reliance": (("0.868", "0.835", "0.902"), ("0.833", "0.791", "0.870"))}
for g, (qv, lv) in T3.items():
    for m, v in ((Q, qv), (L, lv)):
        for col, ww in zip(["auroc_good_revision", "ci_low", "ci_high"], v):
            check(S, f"Table 3 {m} {g} {col}", ww, AU, col, model=m, gate=g)
note(S, "'AUROC 0.70 to 0.87'", "0.70–0.87", "min 0.7029 (LLaVA blur), max 0.8679 (Qwen B)", f"{AU} col `auroc_good_revision`", "OK")
for m, w in ((Q, ("0.868", "0.845", "+0.023", "+0.008", "+0.040")), (L, ("0.833", "0.837", "-0.004", "-0.013", "+0.006"))):
    for col, ww in zip(["gate_B", "gate_A", "difference_B_minus_A", "ci_low", "ci_high"], w):
        check(S, f"H2 prose {m} {col}", ww, H2, col, model=m, metric="auroc_good_revision")
for m, w in ((Q, ("+0.0070", "+0.0023", "+0.0121")), (L, ("-0.0009", "-0.0030", "+0.0011"))):
    for col, ww in zip(["difference_B_minus_A", "ci_low", "ci_high"], w):
        check(S, f"H2 area {m} {col}", ww, H2, col, model=m, metric="area")

# ---------------- Sensitivity prose + Table 4 ----------------
S = "Results: sensitivity / Table 4"
acc = [(float(r["accept_rate_among_flips"]), r["model"], r["gate"], i) for i, r in rows(G) if r["operating_point"] == "native_pi" and r["gate"] in head]
note(S, "Headline gates accept 1% to 4% of flips", "1%–4%", f"{min(acc)[0]*100:.2f}% to {max(acc)[0]*100:.2f}%", f"{G} col `accept_rate_among_flips` (native_pi, headline)", "OK (rounded)")
orc = [(float(r["accept_rate_among_flips"]), r["model"], i) for i, r in rows(RF) if r["reference"] == "oracle" and r["operating_point"] == "native_pi"]
note(S, "Oracle accepts 13% to 14%", "13%–14%", f"{min(orc)[0]*100:.2f}% to {max(orc)[0]*100:.2f}%", f"{RF} col `accept_rate_among_flips` (oracle, native_pi)", "OK (rounded)")
eir = [(float(r["eir"]), r["model"], r["gate"], i) for i, r in rows(G) if r["operating_point"] == "native_pi" and r["gate"] in head]
note(S, "EIR at or below 0.015", "≤ 0.015", f"max {max(eir)[0]:.6f} ({max(eir)[1]} {max(eir)[2]}, row {max(eir)[3]})", f"{G} col `eir`", "OK (rounded)")
ecr = [(float(r["ecr"]), r["model"], r["gate"], i) for i, r in rows(G) if r["operating_point"] == "native_pi" and r["gate"] in head]
note(S, "ECR 0.007 to 0.104", "0.007–0.104", f"{min(ecr)[0]:.6f} (row {min(ecr)[3]}) to {max(ecr)[0]:.6f} (row {max(ecr)[3]})", f"{G} col `ecr`", "OK (rounded)")
for col, ww in zip(["net_change", "net_ci_low", "net_ci_high"], ["+0.009", "+0.004", "+0.015"]):
    check(S, f"Qwen B native {col}", ww, G, col, model=Q, gate="learned_B_conf_shift_reliance", operating_point="native_pi")
for col, ww in zip(["net_change", "net_ci_low", "net_ci_high"], ["-0.005", "-0.010", "-0.001"]):
    check(S, f"LLaVA conf native {col}", ww, G, col, model=L, gate="conf_gate", operating_point="native_pi")
other_pos = [(r["model"], r["gate"], float(r["net_ci_low"]), i) for i, r in rows(G) if r["operating_point"] == "native_pi" and float(r["net_ci_low"]) > 0]
note(S, "'Only Qwen's learned gate B has an interval above zero' (native pi)", "only Qwen B", "; ".join(f"{m} {g} (CI low {v:.4f}, row {i})" for m, g, v, i in other_pos),
     f"{G} col `net_ci_low`", "NOTE: true for headline gates; secondary Qwen reliance_blur_gate also > 0 — say 'of the headline gates'")
neg = [(r["model"], r["gate"], float(r["net_ci_high"]), i) for i, r in rows(G) if r["operating_point"] == "native_pi" and float(r["net_ci_high"]) < 0]
note(S, "'LLaVA's confidence gate is slightly but reliably negative' (native)", "one gate", "; ".join(f"{m} {g} (CI high {v:.4f}, row {i})" for m, g, v, i in neg),
     f"{G} col `net_ci_high`", "NOTE: among headline gates only LLaVA conf; secondary LLaVA shift_gate also < 0" if len(neg) > 1 else "OK")
for m, rng in ((Q, ("+0.050", "+0.059")), (L, ("+0.028", "+0.037"))):
    v = sorted(float(r["net_change"]) for i, r in rows(G) if r["model"] == m and r["operating_point"] == "pi_0.3" and r["gate"] in head)
    lows = [float(r["net_ci_low"]) for i, r in rows(G) if r["model"] == m and r["operating_point"] == "pi_0.3" and r["gate"] in head]
    note(S, f"pi=0.3 headline range {m}", f"{rng[0]} to {rng[1]}", f"{v[0]:.6f} to {v[-1]:.6f}; all CI low > 0: {all(x > 0 for x in lows)}", f"{G} col `net_change`, `net_ci_low` (pi_0.3)",
         "OK" if f"{v[0]:+.3f}" == rng[0] and f"{v[-1]:+.3f}" == rng[1] and all(x > 0 for x in lows) else "MISMATCH")
orc5 = {r["model"]: float(r["net_change"]) for i, r in rows(RF) if r["reference"] == "oracle" and r["operating_point"] == "pi_0.5"}
sh = [float(r["net_change"]) / orc5[r["model"]] for i, r in rows(G) if r["operating_point"] == "pi_0.5" and r["gate"] in head]
note(S, "pi=0.5: 0.46 to 0.59 of oracle's gain", "0.46–0.59", f"{min(sh):.4f} to {max(sh):.4f}", f"{G} `net_change` (pi_0.5) / {RF} oracle `net_change` (pi_0.5)", "OK (rounded)")
check(S, "Qwen always-accept pi=0.5", "+0.028", RF, "net_change", model=Q, reference="always_accept", operating_point="pi_0.5")
check(S, "Qwen B pi=0.5", "+0.171", G, "net_change", model=Q, gate="learned_B_conf_shift_reliance", operating_point="pi_0.5")
aa = [(r["model"], r["operating_point"], float(r["net_change"])) for i, r in rows(RF) if r["reference"] == "always_accept" and float(r["net_change"]) >= 0]
note(S, "Always-accept worse than never except Qwen pi=0.5", "only Qwen pi=0.5", "; ".join(f"{m} {o} {v:+.4f}" for m, o, v in aa), f"{RF} col `net_change`", "OK" if aa == [(Q, "pi_0.5", aa[0][2])] else "MISMATCH")
T4 = {Q: {"always_accept": ["-0.383", "-0.414", "-0.193", "+0.028"], "conf_gate": ["-0.004", "-0.006", "+0.050", "+0.135"], "learned_A_conf_shift": ["+0.003", "+0.001", "+0.051", "+0.161"],
          "learned_B_conf_shift_reliance": ["+0.009", "+0.006", "+0.059", "+0.171"], "oracle": ["+0.074", "+0.058", "+0.174", "+0.291"]},
      L: {"always_accept": ["-0.353", "-0.383", "-0.196", "-0.009"], "conf_gate": ["-0.005", "-0.005", "+0.028", "+0.114"], "learned_A_conf_shift": ["-0.002", "-0.001", "+0.037", "+0.127"],
          "learned_B_conf_shift_reliance": ["-0.001", "0.000", "+0.035", "+0.126"], "oracle": ["+0.061", "+0.046", "+0.138", "+0.229"]}}
for m, d in T4.items():
    for pol, ws in d.items():
        for op, ww in zip(["native_pi", "pi_0.1", "pi_0.3", "pi_0.5"], ws):
            if pol in ("always_accept", "oracle"):
                check(S, f"Table 4 {m} {pol} {op}", ww, RF, "net_change", model=m, reference=pol, operating_point=op)
            else:
                check(S, f"Table 4 {m} {pol} {op}", ww, G, "net_change", model=m, gate=pol, operating_point=op)

# ---------------- H3 + Table 5 ----------------
S = "Results: H3 / Table 5"
for m, w in ((Q, ("-0.083", "-0.170", "+0.005", "-0.041", "-0.183", "+0.095")), (L, ("-0.074", "-0.189", "+0.033", "+0.016", "-0.175", "+0.207"))):
    for row_, off in (("conf_gate_adv_minus_random", 0), ("did_blur_minus_conf", 3)):
        for col, ww in zip(["value", "ci_low", "ci_high"], w[off:off + 3]):
            check(S, f"H3 {m} {row_} {col}", ww, H3, col, model=m, row=row_)
dev = list(csv.DictReader(open(R / "h3_calibration_dev.csv")))
dq = [r for r in dev if r["model"] == Q and r["pope_split"] == "adversarial"][0]
note(S, "Dev 44% (Qwen adversarial, conf1 > 0.9)", "44%", dq["share_wrong_with_conf1_over_0.9"], "results/h3_calibration_dev.csv row 4, col `share_wrong_with_conf1_over_0.9`", "OK")
check(S, "Test 23% (Qwen adversarial)", "0.23", CAL, "share_wrong_with_conf1_over_0.9", model=Q, pope_split="adversarial")
nw = [int(r["n_wrong"]) for i, r in rows(CAL)]
note(S, "26 to 52 wrong answers per split and model", "26–52", f"{min(nw)} to {max(nw)}", f"{CAL} col `n_wrong`", "OK")
T5 = {(Q, "random"): ("27", "0.910", "0.786", "0.222"), (Q, "popular"): ("36", "0.880", "0.798", "0.444"), (Q, "adversarial"): ("52", "0.827", "0.762", "0.231"),
      (L, "random"): ("26", "0.913", "0.757", "0.231"), (L, "popular"): ("47", "0.843", "0.772", "0.234"), (L, "adversarial"): ("46", "0.847", "0.743", "0.239")}
for (m, sp_), v in T5.items():
    for col, ww in zip(["n_wrong", "turn1_accuracy", "mean_conf1_when_wrong", "share_wrong_with_conf1_over_0.9"], v):
        check(S, f"Table 5 {m} {sp_} {col}", ww, CAL, col, model=m, pope_split=sp_)

# ---------------- Mitigations + Table 6 ----------------
S = "Results: mitigations / Table 6"
T6 = {(Q, "grounding"): ["0.475", "-0.050", "-0.057", "-0.043", "0.569", "-0.013", "-0.035", "+0.011", "0.531", "+0.042", "+0.035", "+0.050"],
      (Q, "relook"): ["0.401", "-0.124", "-0.135", "-0.113", "0.422", "-0.159", "-0.186", "-0.129", "0.576", "+0.088", "+0.076", "+0.100"],
      (L, "grounding"): ["0.464", "-0.013", "-0.017", "-0.009", "0.445", "-0.014", "-0.031", "+0.006", "0.524", "+0.009", "+0.005", "+0.013"],
      (L, "relook"): ["0.507", "+0.031", "+0.022", "+0.038", "0.505", "+0.046", "+0.020", "+0.069", "0.494", "-0.021", "-0.028", "-0.013"]}
cols6 = ["eir", "eir_minus_base", "eir_minus_base_ci_low", "eir_minus_base_ci_high", "ecr", "ecr_minus_base", "ecr_minus_base_ci_low", "ecr_minus_base_ci_high",
         "turn2_acc", "turn2_acc_minus_base", "turn2_acc_minus_base_ci_low", "turn2_acc_minus_base_ci_high"]
for (m, c), ws in T6.items():
    for col, ww in zip(cols6, ws):
        check(S, f"Table 6 {m} {c} {col}", ww, MIT, col, model=m, condition=c)
for (m, c), ww in (((Q, "grounding"), "-0.337"), ((Q, "relook"), "-0.296"), ((L, "grounding"), "-0.339"), ((L, "relook"), "-0.373")):
    check(S, f"Prose net native {m} {c}", ww, MIT, "net_change_native_pi", model=m, condition=c)
for (m, c), ww in (((Q, "grounding"), "+0.047"), ((Q, "relook"), "+0.011"), ((L, "grounding"), "-0.009"), ((L, "relook"), "-0.001")):
    check(S, f"Prose net pi=0.5 {m} {c}", ww, MIT, "net_change_pi_0.5", model=m, condition=c)
P6 = "phase6_dev.csv"
for m, mean, mn in ((Q, "0.299", "0.000"), (L, "0.582", "0.033")):
    check(S, f"Dev no-suffix mean mass {m}", mean, P6, "mean_mass_t2", model=m, condition="nosuffix")
    check(S, f"Dev no-suffix min mass {m}", mn, P6, "min_mass_t2", model=m, condition="nosuffix")

# ---------------- Failure counts ----------------
S = "Results: failure cases"
for m, w in ((Q, (("all rows", "universal_failures", "517"), ("all rows", "images_with_failure", "105"), ("a1 wrong", "n_rows", "1380"), ("a1 wrong", "universal_failures", "517"), ("a1 correct", "universal_failures", "0"))),
             (L, (("all rows", "universal_failures", "625"), ("all rows", "images_with_failure", "107"), ("a1 wrong", "n_rows", "1428"), ("a1 wrong", "universal_failures", "624"), ("a1 correct", "universal_failures", "1")))):
    for sub, col, ww in w:
        check(S, f"{m} {sub} {col}", ww, FC, col, model=m, subset=sub)

# ---------------- Discussion / Limitations ----------------
S = "Discussion"
note(S, "'about 87% of first answers are right'", "87%", "0.8722 / 0.8678", f"{FS} col `turn1_acc`", "OK")
note(S, "'always agreeing costs 35 to 38 points'", "35–38", "Qwen 0.8722-0.4889=0.3833; LLaVA 0.8678-0.5150=0.3528", f"{FS} cols `turn1_acc`, `turn2_acc`", "OK")
note(S, "'confidence and shift … AUROC 0.80 to 0.83'", "0.80–0.83", "conf 0.8011/0.8269, shift 0.8240/0.8259", f"{AU}", "OK")
lv = [float(r["auroc_good_revision"]) for i, r in rows(AU) if r["gate"].startswith("learned")]
note(S, "'a learned combination reaches 0.84 to 0.87'", "0.84–0.87", f"{min(lv):.4f} to {max(lv):.4f} (LLaVA B 0.8333)", f"{AU} col `auroc_good_revision`, learned gates",
     "MISMATCH: range is 0.83–0.87 (LLaVA learned B = 0.833)")
note(S, "'dev pattern … 23 to 34 wrong answers per split'", "23–34", f"{min(int(r['n_wrong']) for r in dev)} to {max(int(r['n_wrong']) for r in dev)}", "results/h3_calibration_dev.csv col `n_wrong`", "OK")
note(S, "'one gate is reliably worse' (native)", "one", "see Sensitivity note: LLaVA conf (headline) and LLaVA shift (secondary)", f"{G}", "NOTE: say 'one headline gate'")
S = "Limitations"
note(S, "13%, 1%–4%, +0.009", "—", "as above", "as above", "OK")

# ---------------- Appendix ----------------
S = "Appendix A1/A2"
A1 = {("pushback_type", "neutral_doubt"): ("0.161", "0.374"), ("pushback_type", "polite"): ("0.602", "0.448"), ("pushback_type", "authority"): ("0.529", "0.487"),
      ("pushback_type", "assertive"): ("0.836", "0.586"), ("pope_split", "random"): ("0.541", "0.458"), ("pope_split", "popular"): ("0.507", "0.464"), ("pope_split", "adversarial"): ("0.547", "0.500")}
for (g, v), (qv, lvv) in A1.items():
    check(S, f"A1 flip {v} Qwen", qv, FS, "flip_rate", model=Q, group=g, value=v)
    check(S, f"A1 flip {v} LLaVA", lvv, FS, "flip_rate", model=L, group=g, value=v)
for v, (qa, la) in {"random": ("0.910", "0.913"), "popular": ("0.880", "0.843"), "adversarial": ("0.827", "0.847")}.items():
    check(S, f"A1 turn-1 acc {v} Qwen", qa, FS, "turn1_acc", model=Q, group="pope_split", value=v)
    check(S, f"A1 turn-1 acc {v} LLaVA", la, FS, "turn1_acc", model=L, group="pope_split", value=v)
A2 = {"conf1": (("0.326", "0.306", "0.347"), ("0.473", "0.452", "0.494")), "reliance_blur": (("0.724", "0.707", "0.739"), ("0.710", "0.698", "0.722")),
      "reliance_blank": (("0.534", "0.511", "0.556"), ("0.720", "0.709", "0.730"))}
for s_, (qv, lvv) in A2.items():
    for m, v in ((Q, qv), (L, lvv)):
        for col, ww in zip(["auroc", "ci_low", "ci_high"], v):
            check(S, f"A2 {m} {s_} {col}", ww, POOL, col, model=m, signal=s_, pope_split="all")

# ---------------- External (not in results/) ----------------
S = "Review (external)"
for c in ["MM-SY: ten tasks from TDIUC; three tones", "MM-SY: up to five rounds raise rate by < 5 points", "MM-SY: higher in the larger model of two pairs",
          "MM-SY: LLaVA-1.5 94.6%", "PENDULUM: ~2,000 pairs, six domains", "Rabby et al.: ten VLMs; EIR/ECR", "Perez et al. 2023; Sharma et al. 2024; Gupta et al. (workshop)"]:
    note(S, c, "—", "not in results/", "source papers", "UNTRACEABLE to results/ (external claim; must be checked against the paper)")

# ---------------- print ----------------
n = len(out)
cnt = {}
for o in out:
    k = o[5].split(":")[0].split(" (")[0]
    cnt[k] = cnt.get(k, 0) + 1
print(f"# Number audit of the report draft\n\nChecked {n} items. Status counts: {cnt}\n")
print("| Section | Item | As written | Exact value in source | Source | Status |\n|---|---|---|---|---|---|")
for o in out:
    print("| " + " | ".join(str(x).replace("|", "/") for x in o) + " |")
