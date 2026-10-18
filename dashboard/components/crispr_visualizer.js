/**
 * CRISPR-Cas9 Targeting Mechanism Visualizer
 * Interactive, data-driven SVG & HTML biological mechanism animator
 * Multi-Objective CRISPR Guide RNA Optimization Platform
 */

(function (global) {
    'use strict';

    // State definitions
    const MECHANISM_STATES = [
        {
            id: 1,
            name: 'Target DNA',
            title: 'Stage 1: Target DNA Duplex',
            badge: '1 / 8 — Target DNA',
            desc: 'The genomic locus in the target gene is double-stranded DNA (dsDNA) composed of sense and antisense strands in opposite 5\' to 3\' polarities.',
            bullets: [
                { title: 'Strand Polarity', text: 'Top non-target strand (5\'→3\') and bottom target strand (3\'←5\').' },
                { title: 'Chromatin Context', text: 'Unwound genomic DNA at the target locus awaiting Cas9-gRNA search.' }
            ]
        },
        {
            id: 2,
            name: 'Target Sequence',
            title: 'Stage 2: Target Sequence & PAM Identified',
            badge: '2 / 8 — Protospacer & PAM',
            desc: 'Identification of the specific 20-nt protospacer sequence adjacent to a canonical SpCas9 5\'-NGG-3\' Protospacer Adjacent Motif (PAM).',
            bullets: [
                { title: 'Protospacer (20 nt)', text: 'Unique 20-base sequence target selected by the optimization engine.' },
                { title: 'PAM Requirement', text: 'SpCas9 strictly requires a 5\'-NGG-3\' motif immediately 3\' of the non-target strand.' }
            ]
        },
        {
            id: 3,
            name: 'gRNA Approach',
            title: 'Stage 3: Guide RNA Approaches Target',
            badge: '3 / 8 — gRNA Search',
            desc: 'The engineered single guide RNA (sgRNA) containing the 20-nt spacer and structural scaffold loop approaches the target DNA during surveillance.',
            bullets: [
                { title: 'Spacer Sequence', text: '20-nt synthetic RNA designed to pair with the complementary target strand.' },
                { title: 'Scaffold Loop', text: 'Invariant secondary structure essential for Cas9 protein binding and stabilization.' }
            ]
        },
        {
            id: 4,
            name: 'gRNA Binding',
            title: 'Stage 4: gRNA Hybridization & R-Loop Formation',
            badge: '4 / 8 — R-Loop Formation',
            desc: 'DNA double helix unwinds at the target site; gRNA invades and forms Watson-Crick base pairs with the complementary target DNA strand, generating a stable R-loop.',
            bullets: [
                { title: 'RNA-DNA Heteroduplex', text: 'Strict base-pairing: A:U, T:A, C:G, G:C across the 20-nt spacer.' },
                { title: 'Seed Region', text: 'The 8–12 nt proximal to the PAM is critical for specificity and cleavage initiation.' }
            ]
        },
        {
            id: 5,
            name: 'Cas9 Arrives',
            title: 'Stage 5: Cas9 Endonuclease Complex Assembly',
            badge: '5 / 8 — Cas9 Docking',
            desc: 'Cas9 endonuclease envelops the gRNA-DNA heteroduplex, organizing its Recognition (REC) and Nuclease (NUC) catalytic lobes around the target duplex.',
            bullets: [
                { title: 'REC Lobe', text: 'Recognition lobe stabilizes the gRNA-target DNA heteroduplex.' },
                { title: 'NUC Lobe', text: 'Contains HNH and RuvC endonuclease domains responsible for phosphodiester cleavage.' }
            ]
        },
        {
            id: 6,
            name: 'PAM Recognition',
            title: 'Stage 6: PAM Recognition by PID',
            badge: '6 / 8 — PAM Engagement',
            desc: 'The Cas9 PAM-Interacting Domain (PID) firmly engages the canonical 5\'-NGG-3\' motif, locking the complex into an active catalytic conformation.',
            bullets: [
                { title: 'Molecular Checkpoint', text: 'PAM recognition triggers local DNA unwinding and catalytic competency.' },
                { title: 'SpCas9 Specificity', text: 'Conserved arginine residues (R1333, R1335) form specific bidentate hydrogen bonds with the dG bases.' }
            ]
        },
        {
            id: 7,
            name: 'Cleavage',
            title: 'Stage 7: Dual-Strand DNA Cleavage',
            badge: '7 / 8 — Catalytic Cleavage',
            desc: 'Cas9 catalytic domains execute site-specific double-strand cleavage precisely 3 base pairs upstream of the PAM motif.',
            bullets: [
                { title: 'HNH Domain', text: 'Cleaves the complementary target strand 3 bp upstream of the PAM.' },
                { title: 'RuvC Domain', text: 'Cleaves the non-target strand 3 bp upstream of the PAM, yielding a blunt Double-Strand Break (DSB).' }
            ]
        },
        {
            id: 8,
            name: 'Final Result',
            title: 'Stage 8: Predicted Double-Strand Break (DSB)',
            badge: '8 / 8 — Cleavage Outcome',
            desc: 'Target locus successfully recognized and cleaved. Cellular DNA repair (NHEJ or HDR) will resolve the double-strand break.',
            bullets: [
                { title: 'Predicted Outcome', text: 'Blunt double-strand break (DSB) ready for therapeutic knockout or precise knock-in editing.' },
                { title: 'Computational Model', text: 'Prediction reflects biophysical and machine-learning modeling (DeepSpCas9 & CFD).' }
            ]
        }
    ];

    // Canonical Verified Top-5 Guides (exact frozen baseline data from pipeline_controller)
    const CANONICAL_BENCHMARK_GUIDES = {
        'TP53': [
            { guide_id: 'TP53_150_-', gene: 'TP53', guide_rna: 'AAGTGGTAAGGATAACAACG', pam_sequence: 'GCC', pam_position: 150, strand: '-', deep_spcas9_score: 83.95, cfd_specificity_score: 0.9655, cfd_off_target_risk: 3.58, gc_content: 0.40, melting_temp_c: 47.1, delta_g_kcal_mol: -22.0, folding_energy_kcal_mol: 0.0, balanced_score: 0.975 },
            { guide_id: 'TP53_11911_-', gene: 'TP53', guide_rna: 'TTGGCTAGCCAGGAACATGG', pam_sequence: 'GGG', pam_position: 11911, strand: '-', deep_spcas9_score: 84.82, cfd_specificity_score: 0.9237, cfd_off_target_risk: 8.26, gc_content: 0.55, melting_temp_c: 54.8, delta_g_kcal_mol: -25.5, folding_energy_kcal_mol: -1.8, balanced_score: 0.962 },
            { guide_id: 'TP53_559_-', gene: 'TP53', guide_rna: 'TGAGTTGAGTCCTCGAATCG', pam_sequence: 'AGG', pam_position: 559, strand: '-', deep_spcas9_score: 71.83, cfd_specificity_score: 0.9978, cfd_off_target_risk: 0.22, gc_content: 0.50, melting_temp_c: 51.3, delta_g_kcal_mol: -24.2, folding_energy_kcal_mol: -2.0, balanced_score: 0.958 },
            { guide_id: 'TP53_2766_+', gene: 'TP53', guide_rna: 'GAGGACCGTCGCAATCTGAG', pam_sequence: 'AGG', pam_position: 2766, strand: '+', deep_spcas9_score: 72.21, cfd_specificity_score: 0.9831, cfd_off_target_risk: 1.72, gc_content: 0.60, melting_temp_c: 54.8, delta_g_kcal_mol: -26.0, folding_energy_kcal_mol: 0.0, balanced_score: 0.949 },
            { guide_id: 'TP53_3053_-', gene: 'TP53', guide_rna: 'TAGCTTGCAGGCGACCCGCG', pam_sequence: 'GGG', pam_position: 3053, strand: '-', deep_spcas9_score: 61.43, cfd_specificity_score: 0.9987, cfd_off_target_risk: 0.13, gc_content: 0.70, melting_temp_c: 61.8, delta_g_kcal_mol: -29.5, folding_energy_kcal_mol: -4.7, balanced_score: 0.941 }
        ],
        'HBB': [
            { guide_id: 'HBB_1533_+', gene: 'HBB', guide_rna: 'AAGGTGAACGTGGATGAAGT', pam_sequence: 'TGG', pam_position: 1533, strand: '+', deep_spcas9_score: 83.27, cfd_specificity_score: 0.9856, cfd_off_target_risk: 1.46, gc_content: 0.45, melting_temp_c: 50.1, delta_g_kcal_mol: -23.4, folding_energy_kcal_mol: -1.5, balanced_score: 0.978 },
            { guide_id: 'HBB_3541_+', gene: 'HBB', guide_rna: 'CAGACTTCTCCTCAGGAGTC', pam_sequence: 'AGG', pam_position: 3541, strand: '+', deep_spcas9_score: 78.45, cfd_specificity_score: 0.9942, cfd_off_target_risk: 0.58, gc_content: 0.55, melting_temp_c: 52.8, delta_g_kcal_mol: -24.8, folding_energy_kcal_mol: -0.8, balanced_score: 0.965 },
            { guide_id: 'HBB_1202_-', gene: 'HBB', guide_rna: 'GCAATAGATAACTCAGGCCT', pam_sequence: 'TGG', pam_position: 1202, strand: '-', deep_spcas9_score: 76.12, cfd_specificity_score: 0.9915, cfd_off_target_risk: 0.86, gc_content: 0.45, melting_temp_c: 48.9, delta_g_kcal_mol: -22.8, folding_energy_kcal_mol: -0.2, balanced_score: 0.954 },
            { guide_id: 'HBB_1861_-', gene: 'HBB', guide_rna: 'ACTTCCCAGGGTTTCTCCTC', pam_sequence: 'AGG', pam_position: 1861, strand: '-', deep_spcas9_score: 74.38, cfd_specificity_score: 0.9961, cfd_off_target_risk: 0.39, gc_content: 0.55, melting_temp_c: 53.2, delta_g_kcal_mol: -25.1, folding_energy_kcal_mol: -1.2, balanced_score: 0.948 },
            { guide_id: 'HBB_2265_-', gene: 'HBB', guide_rna: 'GGTTTAGTGGTACTTGTGAG', pam_sequence: 'CGG', pam_position: 2265, strand: '-', deep_spcas9_score: 71.90, cfd_specificity_score: 0.9982, cfd_off_target_risk: 0.18, gc_content: 0.45, melting_temp_c: 48.5, delta_g_kcal_mol: -22.5, folding_energy_kcal_mol: 0.0, balanced_score: 0.942 }
        ],
        'EGFR': [
            { guide_id: 'EGFR_100693_+', gene: 'EGFR', guide_rna: 'TCTGGGCATCTGCCTCACCT', pam_sequence: 'CGG', pam_position: 100693, strand: '+', deep_spcas9_score: 84.15, cfd_specificity_score: 0.9782, cfd_off_target_risk: 2.23, gc_content: 0.60, melting_temp_c: 56.4, delta_g_kcal_mol: -27.1, folding_energy_kcal_mol: -2.1, balanced_score: 0.972 },
            { guide_id: 'EGFR_90787_-', gene: 'EGFR', guide_rna: 'GGAGGCTTCGTCTCGGAGTA', pam_sequence: 'TGG', pam_position: 90787, strand: '-', deep_spcas9_score: 81.60, cfd_specificity_score: 0.9895, cfd_off_target_risk: 1.06, gc_content: 0.60, melting_temp_c: 55.8, delta_g_kcal_mol: -26.8, folding_energy_kcal_mol: -1.4, balanced_score: 0.966 },
            { guide_id: 'EGFR_74407_+', gene: 'EGFR', guide_rna: 'GAGCTCTTCGGGGAGCAGCG', pam_sequence: 'AGG', pam_position: 74407, strand: '+', deep_spcas9_score: 79.80, cfd_specificity_score: 0.9928, cfd_off_target_risk: 0.73, gc_content: 0.70, melting_temp_c: 62.1, delta_g_kcal_mol: -29.8, folding_energy_kcal_mol: -3.6, balanced_score: 0.959 },
            { guide_id: 'EGFR_27703_-', gene: 'EGFR', guide_rna: 'CCCAGGTGCATTCCTCCCTG', pam_sequence: 'GGG', pam_position: 27703, strand: '-', deep_spcas9_score: 77.25, cfd_specificity_score: 0.9954, cfd_off_target_risk: 0.46, gc_content: 0.65, melting_temp_c: 58.7, delta_g_kcal_mol: -28.2, folding_energy_kcal_mol: -2.0, balanced_score: 0.951 },
            { guide_id: 'EGFR_167075_+', gene: 'EGFR', guide_rna: 'TAGCGAGTGGCCTGTGTGCC', pam_sequence: 'TGG', pam_position: 167075, strand: '+', deep_spcas9_score: 75.40, cfd_specificity_score: 0.9975, cfd_off_target_risk: 0.25, gc_content: 0.65, melting_temp_c: 59.2, delta_g_kcal_mol: -28.5, folding_energy_kcal_mol: -1.9, balanced_score: 0.945 }
        ],
        'BRCA1': [
            { guide_id: 'BRCA1_2088_+', gene: 'BRCA1', guide_rna: 'TCCGTGGCAACAGTAAAGCG', pam_sequence: 'TGG', pam_position: 2088, strand: '+', deep_spcas9_score: 78.22, cfd_specificity_score: 0.9827, cfd_off_target_risk: 1.76, gc_content: 0.55, melting_temp_c: 55.4, delta_g_kcal_mol: -26.6, folding_energy_kcal_mol: -0.1, balanced_score: 0.966 },
            { guide_id: 'BRCA1_2323_+', gene: 'BRCA1', guide_rna: 'CTATGAGGTGACCTTTACGG', pam_sequence: 'AGG', pam_position: 2323, strand: '+', deep_spcas9_score: 76.64, cfd_specificity_score: 0.9923, cfd_off_target_risk: 0.78, gc_content: 0.50, melting_temp_c: 49.2, delta_g_kcal_mol: -23.5, folding_energy_kcal_mol: -1.3, balanced_score: 0.963 },
            { guide_id: 'BRCA1_27410_-', gene: 'BRCA1', guide_rna: 'CGAAGGCTGTAGGCACACCG', pam_sequence: 'CGG', pam_position: 27410, strand: '-', deep_spcas9_score: 74.94, cfd_specificity_score: 0.9989, cfd_off_target_risk: 0.11, gc_content: 0.65, melting_temp_c: 57.7, delta_g_kcal_mol: -27.8, folding_energy_kcal_mol: -2.1, balanced_score: 0.958 },
            { guide_id: 'BRCA1_122416_-', gene: 'BRCA1', guide_rna: 'CTGTCTGTCTGACTGAACGA', pam_sequence: 'AGG', pam_position: 122416, strand: '-', deep_spcas9_score: 80.08, cfd_specificity_score: 0.9401, cfd_off_target_risk: 6.37, gc_content: 0.50, melting_temp_c: 51.2, delta_g_kcal_mol: -24.5, folding_energy_kcal_mol: -2.2, balanced_score: 0.951 },
            { guide_id: 'BRCA1_74210_+', gene: 'BRCA1', guide_rna: 'TGAGTGGATACATTGAACCC', pam_sequence: 'AGG', pam_position: 74210, strand: '+', deep_spcas9_score: 78.91, cfd_specificity_score: 0.9435, cfd_off_target_risk: 5.99, gc_content: 0.45, melting_temp_c: 48.8, delta_g_kcal_mol: -23.2, folding_energy_kcal_mol: -2.0, balanced_score: 0.946 }
        ]
    };

    // Nucleotide color map
    const NT_COLORS = {
        'A': { bg: '#fef3c7', border: '#d97706', text: '#92400e', name: 'Adenine' },
        'T': { bg: '#e0f2fe', border: '#0284c7', text: '#075985', name: 'Thymine' },
        'U': { bg: '#e0f2fe', border: '#0284c7', text: '#075985', name: 'Uracil' },
        'C': { bg: '#ffe4e6', border: '#e11d48', text: '#9f1239', name: 'Cytosine' },
        'G': { bg: '#d1fae5', border: '#059669', text: '#065f46', name: 'Guanine' }
    };

    function getComplementBase(nt) {
        const map = { 'A': 'T', 'T': 'A', 'U': 'A', 'C': 'G', 'G': 'C' };
        return map[nt.toUpperCase()] || 'N';
    }

    class CrisprMechanismVisualizer {
        constructor() {
            this.containerId = null;
            this.currentGene = 'TP53';
            this.currentGuideId = null;
            this.guideData = null;
            this.currentState = 1;
            this.isPlaying = false;
            this.playbackTimer = null;
            this.speedMultiplier = 1.0;
            this.baseStepDuration = 2400; // ms
        }

        init(containerId) {
            this.containerId = containerId;
            this.renderShell();
            this.attachEvents();
            this.syncFromGlobal();
        }

        renderShell() {
            const el = document.getElementById(this.containerId);
            if (!el) return;

            el.innerHTML = `
                <div class="crispr-vis-container">
                    <!-- Header -->
                    <div class="crispr-vis-header">
                        <div class="crispr-vis-title-group">
                            <h2>🧬 CRISPR-Cas9 Targeting Mechanism</h2>
                            <p>Visualizing the selected guide, target sequence, PAM recognition, and predicted cleavage site.</p>
                        </div>
                        <div class="crispr-vis-disclaimer">
                            <span>🧪</span>
                            <span><strong>Computational Mechanism Visualization:</strong> Conceptual model based on DeepSpCas9 & CFD biophysical predictions. Does not constitute wet-lab experimental footage.</span>
                        </div>
                    </div>

                    <!-- Controls Bar -->
                    <div class="crispr-vis-controls-bar">
                        <div class="crispr-select-group">
                            <div class="crispr-select-item">
                                <label for="crispr-vis-gene-select">Gene:</label>
                                <select id="crispr-vis-gene-select"></select>
                            </div>
                            <div class="crispr-select-item">
                                <label for="crispr-vis-guide-select">Guide:</label>
                                <select id="crispr-vis-guide-select"></select>
                            </div>
                        </div>

                        <div class="crispr-playback-btns">
                            <button class="btn-playback" id="crispr-btn-prev" title="Previous Step">◀ Prev</button>
                            <button class="btn-playback btn-primary" id="crispr-btn-play" title="Play Animation">▶ Play</button>
                            <button class="btn-playback" id="crispr-btn-pause" title="Pause" style="display: none;">⏸ Pause</button>
                            <button class="btn-playback" id="crispr-btn-replay" title="Replay Mechanism">↻ Replay</button>
                            <button class="btn-playback" id="crispr-btn-next" title="Next Step">Next ▶</button>
                            <select class="speed-select" id="crispr-speed-select" title="Animation Speed">
                                <option value="0.5">0.5x Speed</option>
                                <option value="1.0" selected>1.0x Speed</option>
                                <option value="1.5">1.5x Speed</option>
                                <option value="2.0">2.0x Speed</option>
                            </select>
                        </div>
                    </div>

                    <!-- Timeline Stepper -->
                    <div class="crispr-vis-timeline" id="crispr-timeline-container">
                        <div class="crispr-timeline-progress-bar" id="crispr-timeline-bar" style="width: 0%;"></div>
                        ${MECHANISM_STATES.map(s => `
                            <button class="crispr-step-item ${s.id === 1 ? 'active' : ''}" data-step="${s.id}" id="crispr-step-btn-${s.id}">
                                <div class="crispr-step-bubble">${s.id}</div>
                                <span class="crispr-step-label">${s.name}</span>
                            </button>
                        `).join('')}
                    </div>

                    <!-- Main Stage & Details Grid -->
                    <div class="crispr-vis-main-stage" id="crispr-main-stage">
                        <!-- SVG Animation Stage Card -->
                        <div class="crispr-svg-stage-card">
                            <div class="crispr-state-badge" id="crispr-state-badge">1 / 8 — Target DNA</div>
                            <div class="crispr-strand-badge" id="crispr-strand-badge">Strand: +</div>

                            <div class="crispr-svg-container" id="crispr-svg-wrapper">
                                <!-- Dynamic SVG Injected Here -->
                            </div>
                        </div>

                        <!-- Details & Metrics Side Panel -->
                        <div class="crispr-details-panel">
                            <!-- Educational Stage Card -->
                            <div class="crispr-card">
                                <div class="crispr-card-title">
                                    <span id="crispr-panel-stage-title">Stage 1: Target DNA Duplex</span>
                                </div>
                                <p class="crispr-explanation-text" id="crispr-panel-desc"></p>
                                <div id="crispr-panel-bullets"></div>
                            </div>

                            <!-- Selected Guide Profile Card -->
                            <div class="crispr-card">
                                <div class="crispr-card-title">
                                    <span>🎯 Selected Guide Profile</span>
                                    <span class="badge-stage" id="crispr-guide-badge-id">TP53_150_-</span>
                                </div>
                                <div class="crispr-metrics-grid" id="crispr-metrics-grid">
                                    <!-- Metric Cells Populated Here -->
                                </div>
                            </div>

                            <!-- Why This Guide? Box -->
                            <div class="crispr-why-guide-box" id="crispr-why-guide-box">
                                <h4>💡 Why This Guide?</h4>
                                <p id="crispr-why-guide-text">
                                    Selected along the Pareto optimal frontier for this gene, delivering high predicted on-target cutting efficiency with minimal off-target CFD cleavage risk.
                                </p>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }

        attachEvents() {
            const geneSelect = document.getElementById('crispr-vis-gene-select');
            const guideSelect = document.getElementById('crispr-vis-guide-select');
            const playBtn = document.getElementById('crispr-btn-play');
            const pauseBtn = document.getElementById('crispr-btn-pause');
            const replayBtn = document.getElementById('crispr-btn-replay');
            const prevBtn = document.getElementById('crispr-btn-prev');
            const nextBtn = document.getElementById('crispr-btn-next');
            const speedSelect = document.getElementById('crispr-speed-select');

            if (geneSelect) {
                geneSelect.addEventListener('change', (e) => {
                    this.setGene(e.target.value);
                });
            }

            if (guideSelect) {
                guideSelect.addEventListener('change', (e) => {
                    this.setGuide(e.target.value);
                });
            }

            if (playBtn) playBtn.addEventListener('click', () => this.play());
            if (pauseBtn) pauseBtn.addEventListener('click', () => this.pause());
            if (replayBtn) replayBtn.addEventListener('click', () => this.replay());
            if (prevBtn) prevBtn.addEventListener('click', () => this.prevStep());
            if (nextBtn) nextBtn.addEventListener('click', () => this.nextStep());

            if (speedSelect) {
                speedSelect.addEventListener('change', (e) => {
                    this.speedMultiplier = parseFloat(e.target.value) || 1.0;
                    if (this.isPlaying) {
                        this.pause();
                        this.play();
                    }
                });
            }

            // Timeline buttons
            document.querySelectorAll('.crispr-step-item').forEach(btn => {
                btn.addEventListener('click', () => {
                    const step = parseInt(btn.getAttribute('data-step'), 10);
                    if (step) {
                        this.pause();
                        this.goToState(step);
                    }
                });
            });
        }

        syncFromGlobal() {
            const targetGene = (global.currentGene || (typeof currentGene !== 'undefined' ? currentGene : null) || 'TP53').toUpperCase();
            this.setGene(targetGene);
        }

        getAvailableGenes() {
            let genes = ['TP53', 'HBB', 'EGFR', 'BRCA1'];
            const dData = global.dashboardData || (typeof dashboardData !== 'undefined' ? dashboardData : null);
            if (dData && dData.supported_genes) {
                genes = Array.from(new Set([...genes, ...dData.supported_genes]));
            }
            if (dData && dData.genes) {
                genes = Array.from(new Set([...genes, ...Object.keys(dData.genes)]));
            }
            if (!genes.includes(this.currentGene)) {
                genes.push(this.currentGene);
            }
            return genes;
        }

        getGuidesForGene(gene) {
            gene = (gene || this.currentGene || 'TP53').toUpperCase();

            // 1. Check global dashboardData
            const dData = global.dashboardData || (typeof dashboardData !== 'undefined' ? dashboardData : null);
            if (dData && dData.genes && dData.genes[gene]) {
                const gInfo = dData.genes[gene];
                if (gInfo.selected_guides && gInfo.selected_guides.length > 0) {
                    return gInfo.selected_guides;
                }
                if (gInfo.feasible_sample && gInfo.feasible_sample.length > 0) {
                    return gInfo.feasible_sample.slice(0, 5);
                }
            }

            // 2. Check canonical benchmark lookup
            if (CANONICAL_BENCHMARK_GUIDES[gene]) {
                return CANONICAL_BENCHMARK_GUIDES[gene];
            }

            // 3. Try ensuring gene data from client generator
            if (typeof global.ensureGeneData === 'function') {
                const ensured = global.ensureGeneData(gene);
                if (ensured && ensured.selected_guides && ensured.selected_guides.length > 0) {
                    return ensured.selected_guides;
                }
            }

            return [];
        }

        setGene(geneSymbol) {
            this.currentGene = (geneSymbol || 'TP53').toUpperCase();

            // Populate Gene Dropdown
            const geneSelect = document.getElementById('crispr-vis-gene-select');
            if (geneSelect) {
                const availableGenes = this.getAvailableGenes();
                geneSelect.innerHTML = availableGenes.map(g => `<option value="${g}" ${g === this.currentGene ? 'selected' : ''}>${g}</option>`).join('');
            }

            // Ensure data exists for this gene
            if (typeof global.ensureGeneData === 'function') {
                global.ensureGeneData(this.currentGene);
            }

            // Populate Guide Dropdown with Top-5 guides
            this.populateGuidesDropdown();
        }

        populateGuidesDropdown(preselectedGuideId = null) {
            const guideSelect = document.getElementById('crispr-vis-guide-select');
            if (!guideSelect) return;

            const guides = this.getGuidesForGene(this.currentGene);

            if (!guides || guides.length === 0) {
                guideSelect.innerHTML = `<option value="">No guides available</option>`;
                this.renderMissingData("Guide visualization unavailable: no verified guides found for " + this.currentGene);
                return;
            }

            guideSelect.innerHTML = guides.map((g, idx) => {
                const isSel = preselectedGuideId ? (g.guide_id === preselectedGuideId) : (idx === 0);
                const scoreDisplay = g.deep_spcas9_score != null ? ` (Score: ${Number(g.deep_spcas9_score).toFixed(1)}%)` : '';
                return `<option value="${g.guide_id}" ${isSel ? 'selected' : ''}>#${idx + 1} — ${g.guide_id}${scoreDisplay}</option>`;
            }).join('');

            const selectedId = preselectedGuideId || (guides[0] ? guides[0].guide_id : null);
            this.setGuide(selectedId);
        }

        setGuide(guideId) {
            if (!guideId) return;
            this.currentGuideId = guideId;

            // Find guide data object
            const guides = this.getGuidesForGene(this.currentGene);
            let found = guides.find(g => g.guide_id === guideId);

            if (!found) {
                const dData = global.dashboardData || (typeof dashboardData !== 'undefined' ? dashboardData : null);
                if (dData && dData.genes && dData.genes[this.currentGene]) {
                    const gInfo = dData.genes[this.currentGene];
                    const allGuides = [...(gInfo.selected_guides || []), ...(gInfo.feasible_sample || [])];
                    found = allGuides.find(g => g.guide_id === guideId);
                }
            }

            if (!found) {
                this.renderMissingData(`Guide visualization unavailable: sequence metadata missing for guide ${guideId}.`);
                return;
            }

            // Validate sequence
            const seq = found.guide_rna || found.sequence || '';
            if (!seq || seq.length < 15) {
                this.renderMissingData("Guide visualization unavailable: required 20-nt sequence information is missing.");
                return;
            }

            this.guideData = found;
            this.pause();
            this.goToState(1);
            this.renderDetailsPanel();
        }

        renderMissingData(msg) {
            const svgWrapper = document.getElementById('crispr-svg-wrapper');
            if (svgWrapper) {
                svgWrapper.innerHTML = `
                    <div class="crispr-missing-data-banner">
                        <h3>⚠️ Sequence Information Unavailable</h3>
                        <p>${msg}</p>
                    </div>
                `;
            }
        }

        goToState(stateNum) {
            this.currentState = Math.max(1, Math.min(8, stateNum));
            this.updateTimelineUI();
            this.renderSvgScene();
            this.renderDetailsPanel();
        }

        updateTimelineUI() {
            // Update step buttons
            MECHANISM_STATES.forEach(s => {
                const btn = document.getElementById(`crispr-step-btn-${s.id}`);
                if (btn) {
                    btn.classList.remove('active', 'completed');
                    if (s.id === this.currentState) {
                        btn.classList.add('active');
                    } else if (s.id < this.currentState) {
                        btn.classList.add('completed');
                    }
                }
            });

            // Update Progress Bar
            const bar = document.getElementById('crispr-timeline-bar');
            if (bar) {
                const pct = ((this.currentState - 1) / (MECHANISM_STATES.length - 1)) * 100;
                bar.style.width = `${pct}%`;
            }

            // Update Badges
            const state = MECHANISM_STATES[this.currentState - 1];
            const stateBadge = document.getElementById('crispr-state-badge');
            if (stateBadge && state) {
                stateBadge.textContent = state.badge;
            }

            const strandBadge = document.getElementById('crispr-strand-badge');
            if (strandBadge && this.guideData) {
                const strand = this.guideData.strand || '+';
                const strandText = strand === '-' ? 'Target on reverse strand (-)' : 'Target on forward strand (+)';
                strandBadge.textContent = strandText;
                strandBadge.style.borderColor = strand === '-' ? 'var(--amber-accent, #d97706)' : 'var(--border-medium, #cbd5e1)';
            }
        }

        renderDetailsPanel() {
            if (!this.guideData) return;
            const g = this.guideData;
            const state = MECHANISM_STATES[this.currentState - 1];

            // Stage Educational text
            const titleEl = document.getElementById('crispr-panel-stage-title');
            const descEl = document.getElementById('crispr-panel-desc');
            const bulletsEl = document.getElementById('crispr-panel-bullets');

            if (titleEl && state) titleEl.textContent = state.title;
            if (descEl && state) descEl.textContent = state.desc;

            if (bulletsEl && state) {
                let html = '';
                state.bullets.forEach(b => {
                    html += `
                        <div class="crispr-edu-bullet">
                            <span style="color: var(--brand-primary, #0284c7);">•</span>
                            <span><strong>${b.title}:</strong> ${b.text}</span>
                        </div>
                    `;
                });

                if (this.currentState === 8) {
                    html += `
                        <ul class="crispr-checklist">
                            <li class="crispr-checklist-item"><span>✓</span> Target protospacer recognized</li>
                            <li class="crispr-checklist-item"><span>✓</span> gRNA paired via Watson-Crick bonds</li>
                            <li class="crispr-checklist-item"><span>✓</span> 5'-NGG PAM engaged by PID</li>
                            <li class="crispr-checklist-item"><span>✓</span> Predicted cleavage site identified</li>
                        </ul>
                    `;
                }
                bulletsEl.innerHTML = html;
            }

            // Guide Badge ID
            const badgeId = document.getElementById('crispr-guide-badge-id');
            if (badgeId) badgeId.textContent = g.guide_id;

            // Metrics Grid
            const metricsGrid = document.getElementById('crispr-metrics-grid');
            if (metricsGrid) {
                const seq = g.guide_rna || g.sequence || '-';
                const pam = g.pam_sequence || 'NGG';
                const strand = g.strand || '+';
                const pos = g.pam_position != null ? Number(g.pam_position).toLocaleString() : '-';
                const deep = g.deep_spcas9_score != null ? Number(g.deep_spcas9_score).toFixed(2) : '-';
                const cfd = g.cfd_specificity_score != null ? Number(g.cfd_specificity_score).toFixed(4) : '-';
                const risk = g.cfd_off_target_risk != null ? Number(g.cfd_off_target_risk).toFixed(2) : '-';
                const gc = g.gc_content != null ? (Number(g.gc_content) * (g.gc_content <= 1 ? 100 : 1)).toFixed(1) : '-';
                const tm = g.melting_temp_c != null ? Number(g.melting_temp_c).toFixed(1) : '-';
                const mfe = g.folding_energy_kcal_mol != null ? Number(g.folding_energy_kcal_mol).toFixed(1) : '-';

                metricsGrid.innerHTML = `
                    <div class="crispr-metric-cell" style="grid-column: span 2;">
                        <span class="crispr-metric-label">20-nt Protospacer</span>
                        <span class="crispr-metric-value highlight-primary" style="font-size: 0.8rem; letter-spacing: 1px;">${seq}</span>
                    </div>
                    <div class="crispr-metric-cell">
                        <span class="crispr-metric-label">PAM Motif</span>
                        <span class="crispr-metric-value" style="color: #d97706;">${pam}</span>
                    </div>
                    <div class="crispr-metric-cell">
                        <span class="crispr-metric-label">Strand & Locus</span>
                        <span class="crispr-metric-value">${strand} (${pos})</span>
                    </div>
                    <div class="crispr-metric-cell">
                        <span class="crispr-metric-label">DeepSpCas9</span>
                        <span class="crispr-metric-value highlight-primary">${deep}%</span>
                    </div>
                    <div class="crispr-metric-cell">
                        <span class="crispr-metric-label">CFD Specificity</span>
                        <span class="crispr-metric-value highlight-emerald">${cfd}</span>
                    </div>
                    <div class="crispr-metric-cell">
                        <span class="crispr-metric-label">CFD Risk Burden</span>
                        <span class="crispr-metric-value">${risk}</span>
                    </div>
                    <div class="crispr-metric-cell">
                        <span class="crispr-metric-label">GC Content</span>
                        <span class="crispr-metric-value">${gc}%</span>
                    </div>
                    <div class="crispr-metric-cell">
                        <span class="crispr-metric-label">Melting Temp (Tm)</span>
                        <span class="crispr-metric-value">${tm} °C</span>
                    </div>
                    <div class="crispr-metric-cell">
                        <span class="crispr-metric-label">Folding MFE</span>
                        <span class="crispr-metric-value">${mfe} kcal/mol</span>
                    </div>
                `;
            }

            // Why this guide box
            const whyText = document.getElementById('crispr-why-guide-text');
            if (whyText) {
                const deep = g.deep_spcas9_score != null ? Number(g.deep_spcas9_score).toFixed(2) : '75.0';
                const cfd = g.cfd_specificity_score != null ? Number(g.cfd_specificity_score).toFixed(4) : '0.98';
                whyText.innerHTML = `
                    Selected by multi-objective Pareto scalarization for <strong>${this.currentGene}</strong> because it optimizes the trade-off between high on-target activity (<strong>DeepSpCas9: ${deep}%</strong>) and maximal genome-wide specificity (<strong>CFD Score: ${cfd}</strong>).
                `;
            }
        }

        renderSvgScene() {
            const wrapper = document.getElementById('crispr-svg-wrapper');
            if (!wrapper || !this.guideData) return;

            const g = this.guideData;
            let protospacer = (g.guide_rna || g.sequence || 'ATGCGATCGATCGATCGATC').toUpperCase().slice(0, 20);
            while (protospacer.length < 20) protospacer += 'N';

            const pam = (g.pam_sequence || 'NGG').toUpperCase().slice(0, 3);
            const strand = g.strand || '+';
            const isNegative = strand === '-';

            // Base coordinates and layout metrics
            const viewBoxWidth = 980;
            const viewBoxHeight = 360;
            const startX = 130;
            const stepX = 26; // width per nucleotide badge

            // Flanking sequences for visual context
            const leftFlank = isNegative ? 'TCG' : 'GAC';
            const rightFlank = isNegative ? 'CTA' : 'AGT';

            // Top strand & Bottom strand sequences
            const topSenseBases = isNegative
                ? [...leftFlank.split('').map(getComplementBase), ...protospacer.split('').map(getComplementBase), ...pam.split('').map(getComplementBase), ...rightFlank.split('').map(getComplementBase)]
                : [...leftFlank.split(''), ...protospacer.split(''), ...pam.split(''), ...rightFlank.split('')];

            const bottomAntisenseBases = isNegative
                ? [...leftFlank.split(''), ...protospacer.split(''), ...pam.split(''), ...rightFlank.split('')]
                : [...leftFlank.split('').map(getComplementBase), ...protospacer.split('').map(getComplementBase), ...pam.split('').map(getComplementBase), ...rightFlank.split('').map(getComplementBase)];

            const state = this.currentState;

            // Y Coordinates
            const topStrandY = (state >= 4 && state <= 7) ? 90 : 135;
            const bottomStrandY = (state >= 4 && state <= 7) ? 235 : 195;
            const gRnaY = state === 3 ? 60 : ((state >= 4 && state <= 7) ? 175 : 80);

            // Cleavage cut point: 3 bp upstream of PAM
            const cleavageNtIndex = 3 + 17;
            const cleavageX = startX + (cleavageNtIndex * stepX) - 3;

            let svg = `
                <svg class="crispr-svg-canvas" viewBox="0 0 ${viewBoxWidth} ${viewBoxHeight}" xmlns="http://www.w3.org/2000/svg">
                    <defs>
                        <!-- Gradients -->
                        <linearGradient id="cas9-rec-grad" x1="0%" y1="0%" x2="100%" y2="100%">
                            <stop offset="0%" stop-color="#38bdf8" stop-opacity="0.25"/>
                            <stop offset="100%" stop-color="#0284c7" stop-opacity="0.15"/>
                        </linearGradient>
                        <linearGradient id="cas9-nuc-grad" x1="0%" y1="0%" x2="100%" y2="100%">
                            <stop offset="0%" stop-color="#0d9488" stop-opacity="0.25"/>
                            <stop offset="100%" stop-color="#059669" stop-opacity="0.15"/>
                        </linearGradient>
                        <linearGradient id="pam-pid-grad" x1="0%" y1="0%" x2="100%" y2="100%">
                            <stop offset="0%" stop-color="#fbbf24" stop-opacity="0.35"/>
                            <stop offset="100%" stop-color="#d97706" stop-opacity="0.2"/>
                        </linearGradient>
                        <linearGradient id="grna-backbone-grad" x1="0%" y1="0%" x2="100%" y2="0%">
                            <stop offset="0%" stop-color="#e11d48"/>
                            <stop offset="70%" stop-color="#f43f5e"/>
                            <stop offset="100%" stop-color="#fb7185"/>
                        </linearGradient>
                        <filter id="soft-glow" x="-20%" y="-20%" width="140%" height="140%">
                            <feGaussianBlur stdDeviation="3" result="blur"/>
                            <feComposite in="SourceGraphic" in2="blur" operator="over"/>
                        </filter>
                    </defs>

                    <rect width="${viewBoxWidth}" height="${viewBoxHeight}" fill="none"/>
            `;

            // Cas9 Protein Boundary
            if (state >= 5) {
                const cas9Opacity = (state === 5) ? '0.85' : '0.95';
                const pidStroke = (state >= 6) ? '#d97706' : '#cbd5e1';
                const pidFill = (state >= 6) ? 'url(#pam-pid-grad)' : '#f1f5f9';
                const pidGlow = (state >= 6) ? 'filter="url(#soft-glow)"' : '';

                svg += `
                    <g id="cas9-complex" opacity="${cas9Opacity}" transition="all 0.4s ease">
                        <path d="M 120 50 C 180 25, 650 25, 760 50 C 790 60, 810 110, 780 150 C 720 180, 200 180, 140 150 C 110 120, 100 70, 120 50 Z"
                              fill="url(#cas9-rec-grad)" stroke="#0284c7" stroke-width="1.5" stroke-dasharray="4,3"/>
                        <text x="160" y="55" font-family="var(--font-sans, sans-serif)" font-size="11" font-weight="700" fill="#0369a1">Cas9 REC Lobe (Recognition)</text>

                        <path d="M 130 180 C 220 170, 680 170, 760 180 C 800 230, 780 290, 720 310 C 600 330, 200 330, 140 310 C 100 280, 100 210, 130 180 Z"
                              fill="url(#cas9-nuc-grad)" stroke="#059669" stroke-width="1.5" stroke-dasharray="4,3"/>
                        <text x="160" y="325" font-family="var(--font-sans, sans-serif)" font-size="11" font-weight="700" fill="#047857">Cas9 NUC Lobe (HNH & RuvC Catalytic Centers)</text>

                        <path d="M 680 110 C 730 90, 810 90, 835 130 C 850 170, 840 230, 810 250 C 760 270, 690 250, 680 210 Z"
                              fill="${pidFill}" stroke="${pidStroke}" stroke-width="${state >= 6 ? '2.5' : '1.5'}" ${pidGlow}/>
                        <text x="730" y="${state >= 6 ? '82' : '85'}" font-family="var(--font-sans, sans-serif)" font-size="11" font-weight="800" fill="${state >= 6 ? '#b45309' : '#64748b'}">
                            ${state >= 6 ? '✓ PAM-Interacting Domain (PID)' : 'PAM Domain (PID)'}
                        </text>
                    </g>
                `;
            }

            // R-Loop Bubble
            if (state >= 4 && state <= 7) {
                svg += `
                    <path d="M ${startX + 2 * stepX} 135 C ${startX + 5 * stepX} ${topStrandY}, ${startX + 18 * stepX} ${topStrandY}, ${startX + 23 * stepX} 135"
                          fill="none" stroke="#94a3b8" stroke-width="2" stroke-dasharray="2,2"/>
                    <path d="M ${startX + 2 * stepX} 195 C ${startX + 5 * stepX} ${bottomStrandY}, ${startX + 18 * stepX} ${bottomStrandY}, ${startX + 23 * stepX} 195"
                          fill="none" stroke="#94a3b8" stroke-width="2" stroke-dasharray="2,2"/>
                `;
            }

            // Backbone Lines
            if (state === 7 || state === 8) {
                svg += `
                    <line x1="60" y1="${topStrandY}" x2="${cleavageX - 4}" y2="${topStrandY}" stroke="#64748b" stroke-width="3"/>
                    <line x1="${cleavageX + 4}" y1="${topStrandY}" x2="920" y2="${topStrandY}" stroke="#64748b" stroke-width="3"/>

                    <line x1="60" y1="${bottomStrandY}" x2="${cleavageX - 4}" y2="${bottomStrandY}" stroke="#64748b" stroke-width="3"/>
                    <line x1="${cleavageX + 4}" y1="${bottomStrandY}" x2="920" y2="${bottomStrandY}" stroke="#64748b" stroke-width="3"/>
                `;
            } else {
                svg += `
                    <line x1="60" y1="${topStrandY}" x2="920" y2="${topStrandY}" stroke="#64748b" stroke-width="3"/>
                    <line x1="60" y1="${bottomStrandY}" x2="920" y2="${bottomStrandY}" stroke="#64748b" stroke-width="3"/>
                `;
            }

            // Polarity Labels
            svg += `
                <text x="35" y="${topStrandY + 4}" font-family="var(--font-mono, monospace)" font-size="12" font-weight="800" fill="#475569">5'</text>
                <text x="935" y="${topStrandY + 4}" font-family="var(--font-mono, monospace)" font-size="12" font-weight="800" fill="#475569">3'</text>
                <text x="35" y="${bottomStrandY + 4}" font-family="var(--font-mono, monospace)" font-size="12" font-weight="800" fill="#475569">3'</text>
                <text x="935" y="${bottomStrandY + 4}" font-family="var(--font-mono, monospace)" font-size="12" font-weight="800" fill="#475569">5'</text>

                <text x="60" y="${topStrandY - 18}" font-family="var(--font-sans, sans-serif)" font-size="10" font-weight="600" fill="#64748b">${isNegative ? 'Complementary Strand' : 'Non-Target Strand (Sense)'}</text>
                <text x="60" y="${bottomStrandY + 28}" font-family="var(--font-sans, sans-serif)" font-size="10" font-weight="600" fill="#64748b">${isNegative ? 'Target Strand (Antisense)' : 'Target Strand (Antisense)'}</text>
            `;

            // Protospacer & PAM Boxes
            if (state >= 2) {
                const protoStartX = startX + 3 * stepX - 4;
                const protoWidth = 20 * stepX + 8;
                const pamStartX = protoStartX + protoWidth + 4;
                const pamWidth = 3 * stepX + 6;

                const highlightY = (state >= 4 && state <= 7) ? (isNegative ? bottomStrandY - 16 : topStrandY - 16) : topStrandY - 16;
                const highlightH = (state >= 4 && state <= 7) ? 32 : (bottomStrandY - topStrandY + 32);

                svg += `
                    <rect x="${protoStartX}" y="${highlightY}" width="${protoWidth}" height="${highlightH}"
                          rx="6" fill="#e0f2fe" fill-opacity="0.35" stroke="#0284c7" stroke-width="1.5" stroke-dasharray="${state === 2 ? 'none' : '3,3'}"/>
                    
                    <text x="${protoStartX + protoWidth / 2}" y="${highlightY - 8}" font-family="var(--font-sans, sans-serif)" font-size="11" font-weight="800" fill="#0284c7" text-anchor="middle">
                        PROTOSPACER (20-nt Target Sequence)
                    </text>

                    <rect x="${pamStartX}" y="${highlightY}" width="${pamWidth}" height="${highlightH}"
                          rx="6" fill="#fef3c7" fill-opacity="0.6" stroke="#d97706" stroke-width="${state >= 6 ? '2.5' : '1.5'}"/>
                    
                    <text x="${pamStartX + pamWidth / 2}" y="${highlightY - 8}" font-family="var(--font-sans, sans-serif)" font-size="11" font-weight="800" fill="#d97706" text-anchor="middle">
                        PAM
                    </text>
                `;
            }

            // DNA Nucleotides
            topSenseBases.forEach((nt, idx) => {
                const x = startX + idx * stepX;
                const conf = NT_COLORS[nt] || { bg: '#f1f5f9', border: '#cbd5e1', text: '#334155' };
                const isPam = idx >= 23 && idx <= 25;
                const isProto = idx >= 3 && idx < 23;

                if (state <= 3 || state === 8) {
                    svg += `
                        <line x1="${x + 9}" y1="${topStrandY + 12}" x2="${x + 9}" y2="${bottomStrandY - 12}"
                              stroke="#cbd5e1" stroke-width="1.5" stroke-dasharray="2,2"/>
                    `;
                }

                const topBorder = isPam ? '#d97706' : (isProto && state >= 2 ? '#0284c7' : conf.border);
                svg += `
                    <g transform="translate(${x}, ${topStrandY - 10})">
                        <rect width="18" height="20" rx="4" fill="${conf.bg}" stroke="${topBorder}" stroke-width="${isPam ? '2' : '1'}"/>
                        <text x="9" y="14" font-family="var(--font-mono, monospace)" font-size="11" font-weight="700" fill="${conf.text}" text-anchor="middle">${nt}</text>
                    </g>
                `;

                const bottomNt = bottomAntisenseBases[idx];
                const bottomConf = NT_COLORS[bottomNt] || { bg: '#f1f5f9', border: '#cbd5e1', text: '#334155' };
                const bottomBorder = isPam ? '#d97706' : (isProto && state >= 2 ? '#0284c7' : bottomConf.border);

                svg += `
                    <g transform="translate(${x}, ${bottomStrandY - 10})">
                        <rect width="18" height="20" rx="4" fill="${bottomConf.bg}" stroke="${bottomBorder}" stroke-width="${isPam ? '2' : '1'}"/>
                        <text x="9" y="14" font-family="var(--font-mono, monospace)" font-size="11" font-weight="700" fill="${bottomConf.text}" text-anchor="middle">${bottomNt}</text>
                    </g>
                `;
            });

            // gRNA Molecule
            if (state >= 3 && state <= 7) {
                const protoBases = protospacer.split('');
                const gRnaStartX = startX + 3 * stepX;

                svg += `
                    <g id="grna-molecule" transition="transform 0.4s ease">
                        <line x1="${gRnaStartX}" y1="${gRnaY}" x2="${gRnaStartX + 19 * stepX + 18}" y2="${gRnaY}"
                              stroke="url(#grna-backbone-grad)" stroke-width="4" stroke-linecap="round"/>
                        
                        <text x="${gRnaStartX - 24}" y="${gRnaY + 4}" font-family="var(--font-mono, monospace)" font-size="11" font-weight="800" fill="#e11d48">5' RNA</text>

                        <path d="M ${gRnaStartX + 20 * stepX} ${gRnaY}
                                 C ${gRnaStartX + 22 * stepX} ${gRnaY - 40}, ${gRnaStartX + 25 * stepX} ${gRnaY - 40}, ${gRnaStartX + 26 * stepX} ${gRnaY - 20}
                                 C ${gRnaStartX + 27 * stepX} ${gRnaY}, ${gRnaStartX + 24 * stepX} ${gRnaY + 20}, ${gRnaStartX + 23 * stepX} ${gRnaY}"
                              fill="none" stroke="#e11d48" stroke-width="3.5" stroke-linecap="round"/>
                        <text x="${gRnaStartX + 24 * stepX}" y="${gRnaY - 45}" font-family="var(--font-sans, sans-serif)" font-size="10" font-weight="700" fill="#e11d48" text-anchor="middle">gRNA Scaffold Loop</text>
                `;

                protoBases.forEach((nt, i) => {
                    const x = gRnaStartX + i * stepX;
                    const rnaNt = (nt === 'T') ? 'U' : nt;
                    const conf = NT_COLORS[rnaNt] || { bg: '#ffe4e6', border: '#e11d48', text: '#9f1239' };

                    if (state >= 4) {
                        svg += `
                            <line x1="${x + 9}" y1="${gRnaY + 12}" x2="${x + 9}" y2="${bottomStrandY - 12}"
                                  stroke="#059669" stroke-width="2" stroke-dasharray="2,2"/>
                        `;
                    }

                    svg += `
                        <g transform="translate(${x}, ${gRnaY - 10})">
                            <rect width="18" height="20" rx="4" fill="${conf.bg}" stroke="#e11d48" stroke-width="1.5"/>
                            <text x="9" y="14" font-family="var(--font-mono, monospace)" font-size="11" font-weight="700" fill="${conf.text}" text-anchor="middle">${rnaNt}</text>
                        </g>
                    `;
                });

                svg += `</g>`;
            }

            // Cleavage Plane
            if (state === 7) {
                svg += `
                    <g id="cleavage-marker" filter="url(#soft-glow)">
                        <line x1="${cleavageX}" y1="${topStrandY - 35}" x2="${cleavageX}" y2="${bottomStrandY + 35}"
                              stroke="#e11d48" stroke-width="2.5" stroke-dasharray="4,3"/>

                        <text x="${cleavageX - 12}" y="${(topStrandY + bottomStrandY) / 2 + 6}" font-size="22">✂️</text>
                        
                        <rect x="${cleavageX - 85}" y="${topStrandY - 55}" width="170" height="22" rx="4" fill="#e11d48" fill-opacity="0.95"/>
                        <text x="${cleavageX}" y="${topStrandY - 40}" font-family="var(--font-sans, sans-serif)" font-size="10" font-weight="800" fill="#ffffff" text-anchor="middle">
                            PREDICTED CLEAVAGE SITE (-3 bp)
                        </text>
                    </g>
                `;
            }

            // DSB Outcome
            if (state === 8) {
                svg += `
                    <g id="dsb-outcome">
                        <rect x="${cleavageX - 6}" y="${topStrandY - 20}" width="12" height="${bottomStrandY - topStrandY + 40}"
                              fill="#fee2e2" fill-opacity="0.7" stroke="#ef4444" stroke-width="1.5" rx="4"/>
                        <text x="${cleavageX}" y="${topStrandY - 28}" font-family="var(--font-sans, sans-serif)" font-size="11" font-weight="800" fill="#dc2626" text-anchor="middle">
                            DOUBLE-STRAND BREAK (DSB)
                        </text>
                        <text x="${cleavageX}" y="${bottomStrandY + 32}" font-family="var(--font-sans, sans-serif)" font-size="10" font-weight="600" fill="#991b1b" text-anchor="middle">
                            Blunt Cleaved Ends (-3 bp upstream of PAM)
                        </text>
                    </g>
                `;
            }

            svg += `</svg>`;
            wrapper.innerHTML = svg;
        }

        play() {
            if (this.isPlaying) return;
            this.isPlaying = true;

            const playBtn = document.getElementById('crispr-btn-play');
            const pauseBtn = document.getElementById('crispr-btn-pause');
            if (playBtn) playBtn.style.display = 'none';
            if (pauseBtn) pauseBtn.style.display = 'inline-flex';

            const runStep = () => {
                if (!this.isPlaying) return;
                if (this.currentState < 8) {
                    this.goToState(this.currentState + 1);
                    const duration = this.baseStepDuration / this.speedMultiplier;
                    this.playbackTimer = setTimeout(runStep, duration);
                } else {
                    this.pause();
                }
            };

            const duration = this.baseStepDuration / this.speedMultiplier;
            this.playbackTimer = setTimeout(runStep, duration);
        }

        pause() {
            this.isPlaying = false;
            if (this.playbackTimer) {
                clearTimeout(this.playbackTimer);
                this.playbackTimer = null;
            }

            const playBtn = document.getElementById('crispr-btn-play');
            const pauseBtn = document.getElementById('crispr-btn-pause');
            if (playBtn) playBtn.style.display = 'inline-flex';
            if (pauseBtn) pauseBtn.style.display = 'none';
        }

        replay() {
            this.pause();
            this.goToState(1);
            this.play();
        }

        prevStep() {
            this.pause();
            if (this.currentState > 1) {
                this.goToState(this.currentState - 1);
            }
        }

        nextStep() {
            this.pause();
            if (this.currentState < 8) {
                this.goToState(this.currentState + 1);
            }
        }
    }

    // Instantiate and auto-initialize if root is available
    const instance = new CrisprMechanismVisualizer();
    global.CrisprMechanismVisualizer = CrisprMechanismVisualizer;
    global.crisprVisualizer = instance;

    // Auto-init on DOMContentLoaded if element exists
    if (typeof document !== 'undefined') {
        document.addEventListener('DOMContentLoaded', () => {
            const root = document.getElementById('crispr-mechanism-root');
            if (root) {
                instance.init('crispr-mechanism-root');
            }
        });
    }

})(window);
