# Gene-Specific Interactive CRISPR Mechanism Visualizer

## 1. Overview & Purpose
The **CRISPR-Cas9 Targeting Mechanism Visualizer** is an interactive, data-driven vector graphics (SVG + HTML5 + CSS) visualization component integrated into the Multi-Objective CRISPR Guide RNA Optimization Platform.

Its primary purpose is to bridge the gap between computational multi-objective optimization (Pareto frontier selection balancing DeepSpCas9 activity and CFD off-target specificity) and real molecular biology, explaining the conceptual sequential biophysical process of CRISPR-Cas9 genome editing at the selected target locus.

---

## 2. Data Interface & Source of Truth
The visualizer strictly uses **actual, verified project data** retrieved through `src/pipeline_controller.py` and the dashboard API:
- **Zero hardcoding**: Sequences, PAM motifs, genomic loci, and metrics are read dynamically from the verified Top-5 selected guides or NCBI-resolved genomic profiles.
- **Biophysical parameters displayed**:
  - 20-nt Protospacer sequence
  - PAM motif (5'-NGG-3')
  - Genomic locus & Strand orientation (`+` Sense or `-` Antisense)
  - DeepSpCas9 on-target score (%)
  - CFD genome-wide specificity score (0–1 scale)
  - CFD off-target risk burden
  - GC Content (%), Duplex Melting Temperature ($T_m$), and Secondary Structure Folding Energy ($\text{MFE}$).

---

## 3. Sequential Animation States (1–8)

| State | Stage Name | Biological & Conceptual Process | Visual Representation |
|---|---|---|---|
| **1** | **Target DNA** | Sense (5'→3') and antisense (3'←5') genomic DNA duplex in target region. | Dual strand DNA ladder with nucleotide badges and intact hydrogen bonding. |
| **2** | **Target Sequence** | Identification of 20-nt protospacer sequence adjacent to 5'-NGG-3' PAM motif. | Blue boundary box over 20-nt target and amber highlight over 3-nt PAM. |
| **3** | **gRNA Approach** | Engineered single guide RNA (sgRNA) spacer + scaffold loop enters target vicinity. | 5'→3' RNA spacer with tracrRNA:crRNA secondary hairpin loop structure. |
| **4** | **gRNA Binding** | DNA unwinding; gRNA spacer forms Watson-Crick base pairs (A:U, T:A, C:G, G:C). | R-loop transcription bubble with green hydrogen bonds between gRNA and target DNA. |
| **5** | **Cas9 Arrives** | Cas9 endonuclease envelope docks around the gRNA-DNA heteroduplex. | Stylized bi-lobed envelope showing REC lobe (Recognition) and NUC lobe (Catalytic). |
| **6** | **PAM Recognition** | PAM-Interacting Domain (PID) engages 5'-NGG-3', locking catalytic conformation. | Amber PID domain engagement with arginine-guanine recognition callout. |
| **7** | **Cleavage** | HNH & RuvC catalytic centers cleave phosphodiester bonds 3 bp upstream of PAM. | Animated vertical cleavage plane, ✂ scissor markers, and dual-strand nicking. |
| **8** | **Final Result** | Double-Strand Break (DSB) with blunt ends ready for cellular repair (NHEJ/HDR). | Cleaved DNA ends + summary checklist + complete guide biophysical profile. |

---

## 4. Interactive Controls & Gene/Guide Switching
- **Playback Controls**:
  - `[ ▶ Play ]`, `[ ⏸ Pause ]`, `[ ↻ Replay ]`, `[ ◀ Prev Step ]`, `[ Next Step ▶ ]`.
  - Step timeline progress stepper with direct clickable jumping to any state (1 to 8).
  - Variable speed selector (`0.5x`, `1.0x`, `1.5x`, `2.0x`).
- **Gene Selector**:
  - Supports all canonical benchmark targets (`TP53`, `HBB`, `EGFR`, `BRCA1`) and dynamically resolved NCBI targets (e.g. `PTEN`, `KRAS`, `MYC`).
  - Switching gene dynamically repopulates the Guide Dropdown with the top 5 optimized guides for that gene and resets the mechanism to State 1.
- **Guide Selector**:
  - Allows seamless exploration across all 5 Pareto-optimal candidate guides for the active target.
  - Updating the selected guide updates the 20-nt sequence, PAM, locus, and biophysical scores instantaneously.
- **Strand Polarity Awareness**:
  - Supports both positive (`+`) sense and negative (`-`) antisense targets with orientation badges and appropriate complementary base matching.

---

## 5. Connection to Multi-Objective Optimization
Beside the mechanism animation, the **"Why This Guide?"** panel connects the molecular action directly to the optimization algorithm:
- Explains why the specific guide was selected by equal-weighted Pareto scalarization:
  $$U_i = 0.5 \cdot \text{DeepSpCas9}(i) + 0.5 \cdot \text{CFD Specificity}(i)$$
- Highlights how maximizing on-target cutting efficiency while minimizing off-target CFD cleavage risk produces the ideal candidate for gene editing.

---

## 6. Conceptual Disclaimer
The visualizer clearly features the project disclaimer:
> **Computational Mechanism Visualization:** This conceptual model visualizes biophysical and machine-learning predictions (DeepSpCas9 & CFD) for educational and exploratory purposes. Computational predictions do not constitute wet-lab experimental evidence.
