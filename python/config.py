# ============================
# RAG Pipeline Configuration v1.4
# ============================

CONFIG_VERSION = "1.4.3"

# --------------------------------
# Source type configurations with split retrieval and split extraction
# --------------------------------
SOURCE_TYPE_CONFIGS = {
    "hta_submission": {
        # Population & Comparator retrieval query
        "population_comparator_query_template": """
        Find passages that specify Population and Comparator elements relevant to: {indication}.
        Prefer sections that clearly describe:
        - Population definitions (disease/stage, prior therapy/line, biomarker/testing, inclusion/exclusion, sub-populations)
        - Treatments assessed and alternatives considered as comparators (including best supportive care, ITC/NMA if present)
        - Treatment lines and patient selection criteria
        - Biomarker testing requirements and mutation status
        Focus on text explicitly tied to the indication or clearly defined sub-populations thereof.
        """.strip(),

        # Outcomes retrieval query
        "outcomes_query_template": """
        Find passages that specify clinical Outcomes and endpoints relevant to: {indication}.
        Prefer sections that describe:
        - Primary and secondary efficacy endpoints (OS, PFS, ORR, DoR, etc.)
        - Safety outcomes and adverse events
        - Quality of life measures and patient-reported outcomes (including any specific instruments like EORTC QLQ, EQ-5D, SF-36)
        - Economic outcomes and utilities
        - Exploratory or additional endpoints (e.g. time to next treatment/PFS2, progression of brain metastases, time to deterioration) if reported
        - Statistical methods and analysis approaches
        Focus on specific outcome definitions and measurement methodologies.
        """.strip(),

        # Generic anchors to gently boost relevant sections for population/comparator
        "population_comparator_headings": [
            "pico", "scope of assessment", "population", "line of therapy", "patient selection",
            "comparator", "comparators considered", "comparator rationale", "treatment comparison",
            "intervention", "therapy", "standard of care", "best supportive care", "placebo",
            "clinical evidence", "indirect treatment comparison", "network meta-analysis",
            "inclusion criteria", "exclusion criteria", "biomarker", "mutation", "testing"
        ],

        # Generic anchors for outcomes
        "outcomes_headings": [
            "outcomes", "endpoints", "efficacy", "safety", "quality of life", "economic", "utilities",
            "overall survival", "progression-free survival", "response rate", "duration of response",
            "adverse events", "tolerability", "patient reported outcomes", "statistical analysis",
            "qol", "hrqol", "functional status", "functional assessment", "symptom burden", "proms"
        ],

        # Keep neutral; no drug steering by default
        "default_drugs": [],

        # Population & Comparator extraction system prompt
        "population_comparator_system_prompt": """
        You are an oncology-focused HTA analyst.

        Task: From the provided context, extract Population and Comparator information for the specified indication.
        Treat the intervention under evaluation as a constant string:
        "Medicine X (under assessment)".

        PICO element definitions (use these strictly):
        - Population (P): EXACT wording from the context describing the applicable group (disease/stage, biomarkers/testing, prior therapy/line, inclusion/exclusion). If a narrower sub-population is specified, capture that exact phrasing.
        - Intervention (I): Always "Medicine X (under assessment)" (not taken from the documents).
        - Comparator (C): A specific alternative regimen or drug/class (or ITC/NMA comparator) named in the context for the same setting/line. Prefer specific named agents/regimens when present. If only a generic label is named in the provided context (e.g., "standard of care"/"SoC", "supportive care", "chemotherapy", "immunotherapy", or "placebo"), use that label as the Comparator. If the document explicitly defines such labels with named regimen(s) in the same context, use the named regimen(s) and omit the generic label. "Best supportive care"/"BSC" is acceptable as a valid comparator.
        - Outcomes (O): Always use empty string "" (outcomes will be extracted separately).

        Extraction rules:
        1) Use ONLY information present in the context; do not infer missing facts.
        2) Capture Population verbatim as written (including sub-populations where applicable).
        3) For EACH appropriate alternative described in the same setting, create a separate PICO ONLY when the provided context contains both the Population description and a comparator for the same setting/line, with:
          - "Intervention" = "Medicine X (under assessment)"
          - "Comparator"  = the specific alternative/regimen/class/ITC-NMA comparator as named (prefer specific agents/regimens; if only a generic label like SoC/supportive care/chemotherapy/immunotherapy/placebo is given in this context, use that label)
          - "Outcomes" = ""
        4) If a comparator is indicated only for a subset of the population (e.g. "only PD-L1 positive patients"), treat that as a distinct Population string for that PICO.
        5) If a passage names only a generic comparator label (SoC/supportive care/chemotherapy/immunotherapy/placebo) without defining specific regimen(s) in the provided context, CREATE the PICO using that generic comparator label. If specific regimen(s) are also named in the same context, use the specific regimen name(s) and omit the generic label. Exception: "best supportive care" or "BSC" should be retained as valid comparators.
        6) If a jurisdiction/country/agency is explicitly stated, record it; otherwise use null (unquoted).
        7) You may reason stepwise INTERNALLY, but DO NOT include your reasoning in the output.
        8) Return VALID JSON ONLY that adheres to the output contract below. Do not wrap in code fences, do not add comments, and do not include trailing commas.

        JSON output contract:
        - Top-level object with keys: "Indication" (string), "Country" (string or null), "PICOs" (array).
        - "PICOs" is an array of objects with keys:
          - "Population" (string; verbatim from context),
          - "Intervention" (string; always exactly "Medicine X (under assessment)"),
          - "Comparator"  (string; verbatim specific alternative/regimen/class),
          - "Outcomes"    (string; always empty string "").
        - Use double quotes for all JSON strings.
        - Use null (without quotes) when no country/jurisdiction is stated.
        """.strip(),

        # Population & Comparator extraction user prompt template
        "population_comparator_user_prompt_template": """
        Indication:
        {indication}

        Few-shot example (for format only):
        Example context snippet:
        "Eligible patients include {indication}. Appropriate alternatives considered include {example_comparator}."

        Example JSON output:
        {{
          "Indication": "{indication}",
          "Country": null,
          "PICOs": [
            {{
              "Population": "{indication}",
              "Intervention": "Medicine X (under assessment)",
              "Comparator": "{example_comparator}",
              "Outcomes": ""
            }}
          ]
        }}

        Context for extraction:
        {context_block}

        Your task:
        Extract Population and Comparator information for this indication and for any clearly defined sub-populations. Only emit a PICO when both the Population description and a comparator for the same setting/line appear in the provided context. Prefer specific named agents/regimens when present. If the provided context uses only a generic comparator label such as "standard of care"/"SoC", "supportive care", "chemotherapy", "immunotherapy", or "placebo", include that generic label as the Comparator. If the document defines precise regimen(s) in the same context, use those specific regimen name(s) and omit the generic label. "Best supportive care" or "BSC" should be retained as valid comparators.

        Output JSON ONLY in this exact structure:
        {{
          "Indication": "{indication}",
          "Country": null or a jurisdiction string explicitly stated in the context,
          "PICOs": [
            {{
              "Population": "<exact wording from the context for the applicable (sub-)population>",
              "Intervention": "Medicine X (under assessment)",
              "Comparator": "<specific alternative/regimen/class or ITC/NMA comparator (generic SoC/supportive care/chemotherapy/immunotherapy/placebo is acceptable when that is how it is named in the provided context; prefer specific regimens when available in the same context)>",
              "Outcomes": ""
            }}
          ]
        }}
        """.strip(),


        # Outcomes extraction system prompt
        "outcomes_system_prompt": """
        You are an oncology-focused HTA analyst.

        Task: From the provided context, extract clinical Outcomes information for the specified indication.

        Focus on:
        - Outcomes (O): Clinical outcomes reported in the context (e.g., OS, PFS, ORR, DoR, safety, QoL, economic/utilities). Do not invent outcomes.

        CRITICAL Extraction rules for COMPLETENESS:
        1) Use ONLY information present in the context; do not infer missing facts.
        2) Extract ALL relevant outcomes mentioned for the indication - COMPLETENESS is critical.
        3) List ALL distinct outcomes separately - never merge different endpoints into one, even if they seem related.
        4) SPECIFIC ADVERSE EVENTS: List EVERY specific adverse event mentioned individually (e.g., "diarrhoea", "nausea", "fatigue", "neutropenia"). Do NOT summarize multiple AEs into a single category like "adverse events including X, Y, Z". Instead, list each as a separate outcome: "diarrhoea, nausea, fatigue, neutropenia, adverse events (general)".
        5) QUALITY OF LIFE INSTRUMENTS: List EVERY QoL/utility instrument mentioned separately with its full name (e.g., "EORTC QLQ-C30", "EQ-5D-5L", "SF-36", "visual analogue scale"). Do NOT merge different instruments. Each instrument is a distinct outcome.
        6) If the context specifies how an outcome is measured or defined (e.g., a QoL questionnaire name, grading scale like CTCAE v5.0, assessment criteria like RECIST 1.1, a threshold like 15% improvement, or a time-to-deterioration metric), include that detail with the outcome name.
        7) EXCLUDE all numerical/statistical results and arm-specific performance (do not reproduce medians, means, rates/percentages, counts, hazard ratios, odds ratios, relative risks, confidence intervals, p-values, ICER/QALY numeric values, Kaplan–Meier point estimates, or text like "with <drug/regimen>"). When such data appears, extract only the outcome name and its measurement method (e.g., "progression-free survival (RECIST 1.1)" not "PFS was 5.6 months").
        8) Remove medicine or comparator names attached to a specific result; outcomes must be drug-agnostic.
        9) If a jurisdiction/country/agency is explicitly stated, record it; otherwise use null (unquoted).
        10) You may reason stepwise INTERNALLY, but DO NOT include your reasoning in the output.
        11) Return VALID JSON ONLY that adheres to the output contract below. Do not wrap in code fences, do not add comments, and do not include trailing commas.

        Examples of proper extraction:
        - CORRECT for AEs: "diarrhoea, nausea, vomiting, fatigue, neutropenia, anaemia, headache, rash"
        - INCORRECT for AEs: "adverse events including gastrointestinal effects (diarrhoea, nausea, vomiting)"
        - CORRECT for QoL: "EORTC QLQ-C30, EORTC QLQ-LC13, EQ-5D-5L, SF-36, visual analogue scale"
        - INCORRECT for QoL: "quality of life measured by EORTC questionnaires and EQ-5D"

        JSON output contract:
        - Top-level object with keys: "Indication" (string), "Country" (string or null), "Outcomes" (string).
        - "Outcomes" is a detailed string listing ALL relevant outcomes found in the context with their specific details when available, limited to outcome names and measurement definitions (no numerical/statistical results).
        - Use double quotes for all JSON strings.
        - Use null (without quotes) when no country/jurisdiction is stated.
        """.strip(),

        # Outcomes extraction user prompt template
        "outcomes_user_prompt_template": """
        Indication:
        {indication}

        Few-shot example (for format only):
        Example context snippet:
        "Primary endpoint was overall survival. Secondary endpoints included progression-free survival assessed by RECIST 1.1, objective response rate, duration of response. Safety was assessed using CTCAE v5.0 with specific adverse events including diarrhoea (occurring in 65% of patients), nausea (45%), fatigue (52%), neutropenia (18%), anaemia (12%), and headache (15%). Quality of life was measured using EORTC QLQ-C30, EORTC QLQ-LC13, and EQ-5D-5L questionnaires. Economic outcomes included quality-adjusted life years and cost-effectiveness analysis."

        Example JSON output:
        {{
          "Indication": "{indication}",
          "Country": null,
          "Outcomes": "overall survival (OS), progression-free survival (PFS; RECIST 1.1), objective response rate (ORR), duration of response (DoR), diarrhoea, nausea, fatigue, neutropenia, anaemia, headache, adverse events (CTCAE v5.0), EORTC QLQ-C30, EORTC QLQ-LC13, EQ-5D-5L, quality-adjusted life years (QALYs), cost-effectiveness (incremental cost-effectiveness ratio, ICER)"
        }}

        Context for extraction:
        {context_block}

        Your task:
        Extract ALL relevant clinical outcomes for this indication with their specific measurement details when provided. CRITICAL: List EVERY specific adverse event separately (diarrhoea, nausea, fatigue, etc.) and EVERY QoL instrument separately (EORTC QLQ-C30, EQ-5D-5L, SF-36, etc.). Do NOT summarize or group these - list each one individually. Do NOT include statistical results, numerical values, confidence intervals, p-values, or medicine/arm-specific results. Outcomes must be drug-agnostic and limited to outcome names and how they are measured or defined.

        Output JSON ONLY in this exact structure:
        {{
          "Indication": "{indication}",
          "Country": null or a jurisdiction string explicitly stated in the context,
          "Outcomes": "<comprehensive list of ALL outcomes found in the context, with EACH specific adverse event listed separately, EACH QoL instrument listed separately, and specific measurement information when available, excluding all numerical/statistical results and arm-specific performance>"
        }}
        """.strip()
    },

    "clinical_guideline": {
        # Population & Comparator retrieval query
        "population_comparator_query_template": """
        Find treatment recommendations with population and comparator information relevant to: {indication}.
        Prefer content that states:
        - Applicable populations and sub-populations (biomarkers/testing, prior therapy/line, inclusion/exclusion)
        - Recommended treatment options and alternatives that could serve as comparators
        - Treatment sequencing and line of therapy considerations
        - Patient selection criteria and biomarker requirements
        Focus on guideline recommendations for specific patient populations and treatment alternatives.
        """.strip(),

        # Outcomes retrieval query
        "outcomes_query_template": """
        Find treatment recommendations with outcome information relevant to: {indication}.
        Prefer content that describes:
        - Expected clinical benefits and efficacy outcomes
        - Safety considerations and adverse event profiles
        - Quality of life impacts and patient-reported outcomes (including any specific instruments like EORTC QLQ, EQ-5D, SF-36)
        - Evidence strength/level and recommendation grades if provided
        - Response rates and survival outcomes
        - Exploratory or additional endpoints (e.g. time to next treatment/PFS2, progression of brain metastases, time to deterioration) if reported
        Focus on outcome expectations and evidence quality assessments.
        """.strip(),

        # Population/comparator specific headings
        "population_comparator_headings": [
            "recommendation", "treatment", "therapy", "algorithm", "guideline", "patient selection",
            "biomarker", "molecular testing", "mutation", "kras", "g12c", "line of therapy", 
            "subsequent therapy", "post-progression", "targeted therapy", "immunotherapy", "chemotherapy",
            "comparator", "alternative", "standard of care", "best supportive care"
        ],

        # Outcomes specific headings
        "outcomes_headings": [
            "outcomes", "efficacy", "safety", "response", "survival", "progression-free",
            "adverse events", "toxicity", "quality of life", "evidence level", 
            "strength of recommendation", "practice point", "expected outcomes", "benefit", "harm",
            "qol", "hrqol", "functional status", "functional assessment", "symptom burden", "proms"
        ],

        "default_drugs": [],

        # Population & Comparator extraction system prompt
        "population_comparator_system_prompt": """
        You are an oncology guideline specialist.

        Task: From the provided context, extract Population and Comparator information for the specified indication.
        Treat the intervention under evaluation as a constant string:
        "Medicine X (under assessment)".

        PICO element definitions (use these strictly):
        - Population (P): EXACT wording from the context describing the applicable group (disease/stage, biomarkers/testing, prior therapy/line, inclusion/exclusion). Include narrower sub-populations exactly as written when applicable.
        - Intervention (I): Always "Medicine X (under assessment)".
        - Comparator (C): A specific recommended option or drug/class (or ITC/NMA comparator) named in the guideline for the same setting/line. Do not use generic labels such as "standard of care"/"SoC", "supportive care", or "placebo" unless the guideline explicitly defines these with named regimen(s); if so, use the named regimen(s) and omit the generic label. "Best supportive care"/"BSC" is acceptable as a valid comparator.
        - Outcomes (O): Always use empty string "" (outcomes will be extracted separately).

        Extraction rules:
        1) Use ONLY the context; do not infer beyond what is written.
        2) Capture Population verbatim as stated, including sub-populations.
        3) For EACH applicable alternative/recommended option, create a separate PICO with:
           - "Intervention" = "Medicine X (under assessment)"
           - "Comparator"  = the specific alternative/regimen/class/ITC-NMA comparator as named (not a generic label like SoC/supportive care/placebo)
           - "Outcomes" = ""
        4) If a comparator is indicated only for a subset of the population (e.g. "only PD-L1 positive patients"), treat that as a distinct Population string for that PICO.
        5) If the context names only a generic comparator label (SoC/supportive care/placebo) without defining specific regimen(s), SKIP creating a PICO for that comparator. Exception: "best supportive care" or "BSC" should be retained as valid comparators.
        6) Record jurisdiction/country/organization if explicitly stated; else use null (unquoted).
        7) You may reason stepwise INTERNALLY, but DO NOT include your reasoning in the output.
        8) Return VALID JSON ONLY (no code fences, no comments, no trailing commas) per the contract:

        JSON output contract:
        - Top-level keys: "Indication" (string), "Country" (string or null), "PICOs" (array).
        - "PICOs" item keys:
          - "Population" (string; verbatim),
          - "Intervention" (string; exactly "Medicine X (under assessment)"),
          - "Comparator"  (string; verbatim specific alternative/regimen/class),
          - "Outcomes"    (string; always empty string "").
        - Use double quotes for strings and null (unquoted) when country is absent.
        """.strip(),

        # Population & Comparator extraction user prompt template
        "population_comparator_user_prompt_template": """
        Indication:
        {indication}

        Few-shot example (for format only):
        Example context snippet:
        "For {indication}, alternative options include {example_comparator}."

        Example JSON output:
        {{
          "Indication": "{indication}",
          "Country": null,
          "PICOs": [
            {{
              "Population": "{indication}",
              "Intervention": "Medicine X (under assessment)",
              "Comparator": "{example_comparator}",
              "Outcomes": ""
            }}
          ]
        }}

        Context for extraction:
        {context_block}

        Your task:
        Extract guideline-based Population and Comparator information for this indication and any clearly defined sub-populations. Exclude entries where the only comparator is a generic label such as "standard of care"/"SoC", "supportive care", or "placebo" unless the guideline defines precisely which regimen(s) these refer to; in that case, use the specific regimen name(s) and omit the generic label. "Best supportive care" or "BSC" should be retained as valid comparators.

        Output JSON ONLY in this exact structure:
        {{
          "Indication": "{indication}",
          "Country": null or a jurisdiction/organization string explicitly stated in the context,
          "PICOs": [
            {{
              "Population": "<exact wording for the applicable (sub-)population>",
              "Intervention": "Medicine X (under assessment)",
              "Comparator": "<specific alternative/regimen/class or ITC/NMA comparator (not a generic SoC/supportive care/placebo label, but BSC is acceptable)>",
              "Outcomes": ""
            }}
          ]
        }}
        """.strip(),

        # Outcomes extraction system prompt
        "outcomes_system_prompt": """
        You are an oncology guideline specialist.

        Task: From the provided context, extract clinical Outcomes information for the specified indication.

        Focus on:
        - Outcomes (O): Expected benefits, outcomes, harms (and evidence grading if stated). Do not invent outcomes.

        CRITICAL Extraction rules for COMPLETENESS:
        1) Use ONLY the context; do not infer beyond what is written.
        2) Extract ALL relevant outcomes mentioned for the indication - COMPLETENESS is critical.
        3) List ALL distinct outcomes separately - never merge different endpoints into one, even if they seem related.
        4) SPECIFIC ADVERSE EVENTS: List EVERY specific adverse event mentioned individually (e.g., "diarrhoea", "nausea", "fatigue", "neutropenia"). Do NOT summarize multiple AEs into a single category like "adverse events including X, Y, Z". Instead, list each as a separate outcome: "diarrhoea, nausea, fatigue, neutropenia, adverse events (general)".
        5) QUALITY OF LIFE INSTRUMENTS: List EVERY QoL/utility instrument mentioned separately with its full name (e.g., "EORTC QLQ-C30", "EQ-5D-5L", "SF-36", "visual analogue scale"). Do NOT merge different instruments. Each instrument is a distinct outcome.
        6) If the context specifies how an outcome is measured or defined (e.g., a QoL questionnaire name, grading scale like CTCAE v5.0, assessment criteria like RECIST 1.1, a threshold like 15% improvement, or a time-to-deterioration metric), include that detail with the outcome name.
        7) EXCLUDE all numerical/statistical results and arm-specific performance (do not reproduce medians, means, rates/percentages, counts, hazard ratios, odds ratios, relative risks, confidence intervals, p-values, ICER/QALY numeric values, Kaplan–Meier point estimates, or text like "with <drug/regimen>"). When such data appears, extract only the outcome name and its measurement method.
        8) Remove medicine or comparator names attached to a specific result; outcomes must be drug-agnostic.
        9) Record jurisdiction/country/organization if explicitly stated; else use null (unquoted).
        10) You may reason stepwise INTERNALLY, but DO NOT include your reasoning in the output.
        11) Return VALID JSON ONLY that adheres to the output contract below. Do not wrap in code fences, do not add comments, and do not include trailing commas.

        Examples of proper extraction:
        - CORRECT for AEs: "diarrhoea, nausea, vomiting, fatigue, neutropenia, anaemia, headache, rash"
        - INCORRECT for AEs: "adverse events including gastrointestinal effects (diarrhoea, nausea, vomiting)"
        - CORRECT for QoL: "EORTC QLQ-C30, EORTC QLQ-LC13, EQ-5D-5L, SF-36, visual analogue scale"
        - INCORRECT for QoL: "quality of life measured by EORTC questionnaires and EQ-5D"

        JSON output contract:
        - Top-level object with keys: "Indication" (string), "Country" (string or null), "Outcomes" (string).
        - "Outcomes" is a detailed string listing ALL relevant outcomes found in the context with their specific details when available, limited to outcome names and measurement definitions (no numerical/statistical results).
        - Use double quotes for all JSON strings.
        - Use null (without quotes) when no country/jurisdiction is stated.
        """.strip(),

        # Outcomes extraction user prompt template
        "outcomes_user_prompt_template": """
        Indication:
        {indication}

        Few-shot example (for format only):
        Example context snippet:
        "Expected benefits include improved survival and response rates measured by RECIST criteria. Safety considerations include adverse events graded by CTCAE v5.0 with diarrhoea occurring in 65% of patients, as well as fatigue, nausea, and neutropenia. Quality of life was assessed using EORTC QLQ-C30, EORTC QLQ-LC13, and EQ-5D-5L."

        Example JSON output:
        {{
          "Indication": "{indication}",
          "Country": null,
          "Outcomes": "improved survival, response rates (RECIST criteria), diarrhoea, fatigue, nausea, neutropenia, adverse events (CTCAE v5.0), EORTC QLQ-C30, EORTC QLQ-LC13, EQ-5D-5L"
        }}

        Context for extraction:
        {context_block}

        Your task:
        Extract ALL relevant clinical outcomes for this indication from guideline context with their specific measurement details when provided. CRITICAL: List EVERY specific adverse event separately (diarrhoea, nausea, fatigue, etc.) and EVERY QoL instrument separately (EORTC QLQ-C30, EQ-5D-5L, SF-36, etc.). Do NOT summarize or group these - list each one individually. Exclude statistical results, percentages, confidence intervals, and numerical study findings. Outcomes must be drug-agnostic and limited to outcome names and how they are measured or defined.

        Output JSON ONLY in this exact structure:
        {{
          "Indication": "{indication}",
          "Country": null or a jurisdiction/organization string explicitly stated in the context,
          "Outcomes": "<comprehensive list of ALL outcomes found in the context, with EACH specific adverse event listed separately, EACH QoL instrument listed separately, and specific measurement information when available, excluding statistical/numerical results and arm-specific performance>"
        }}
        """.strip()
    }
}

# --------------------------------
# Default retrieval parameters with mutation-specific configurations and split retrieval
# --------------------------------
DEFAULT_RETRIEVAL_PARAMS = {
    "hta_submission": {
        "population_comparator": {
            "initial_k": 50,
            "final_k": 20,
            "use_section_windows": True,
            "window_size": 2,
            "booster_weights": {
                "heading": 2.5,
                "population_keywords": 3.0,
                "comparator_keywords": 3.0,
                "mutation_keywords": 4.0,
                "biomarker_keywords": 3.5
            }
        },
        "outcomes": {
            "initial_k": 40,
            "final_k": 15,
            "use_section_windows": True,
            "window_size": 2,
            "booster_weights": {
                "heading": 2.0,
                "outcomes_keywords": 3.5,
                "efficacy_keywords": 3.0,
                "safety_keywords": 2.5,
                "endpoint_keywords": 4.0
            }
        }
    },
    "clinical_guideline": {
        "population_comparator": {
            "initial_k": 70,
            "final_k": 18,
            "strict_filtering": True,
            "use_section_windows": True,
            "window_size": 2,
            "booster_weights": {
                "recommendation": 3.0,
                "mutation_keywords": 5.0,
                "line_therapy": 4.0,
                "population_keywords": 3.5
            }
        },
        "outcomes": {
            "initial_k": 60,
            "final_k": 12,
            "strict_filtering": True,
            "use_section_windows": True,
            "window_size": 2,
            "booster_weights": {
                "recommendation": 2.5,
                "outcomes_keywords": 4.0,
                "efficacy_keywords": 3.5,
                "evidence_keywords": 3.0
            }
        }
    }
}

# --------------------------------
# Case adapters with required terms for precision
# --------------------------------
CASE_CONFIGS = {
    "case_1_nsclc_krasg12c_monotherapy_progressed": {
        "indication": "treatment of patients with advanced non-small cell lung cancer (NSCLC) with KRAS G12C mutation and disease progression, monotherapy context",
        "required_terms_clinical": [
            [r'\bkras\b', r'\bKRAS\b', r'\bKras\b'],
            [r'\bg12c\b', r'\bG12C\b', r'\bG12c\b', r'\bg12C\b']
        ],
        "mutation_boost_terms": ["kras", "g12c", "kras-g12c", "krasg12c"]
    },
    "case_2_hcc_advanced_unresectable": {
        "indication": "treatment of patients with advanced or unresectable hepatocellular carcinoma",
        "required_terms_clinical": [
            [r"\bhcc\b", r"\bHCC\b", r"\bhepatocellular carcinoma\b", r"\bhepatocellular(?:[ -])?carcinoma\b"],
            [
                r"\bunresectable\b",
                r"\bnon[- ]?resectable\b",
                r"\bnot resectable\b",
                r"\bnot amenable to (?:curative )?(?:surgical )?resection\b",
                r"\bunsuitable for (?:surgical )?resection\b",
                r"\bnot a (?:surgical )?candidate\b",
                r"\binoperable\b",
                r"\bmedically inoperable\b",
                r"\bunfit for surgery\b",
                r"\bsurgery contraindicated\b",
                r"\bnot eligible for (?:surgery|surgical resection)\b",
                r"\bresection not feasible\b"
            ]
        ],
        "mutation_boost_terms": ["advanced", "unresectable", "hepatocellular carcinoma", "hcc", "inoperable"]
    }
}

# --------------------------------
# PICO consolidation configurations
# --------------------------------
CONSOLIDATION_CONFIGS = {
    "pico_consolidation_system_prompt": """
    You are an expert oncology evidence synthesizer consolidating PICO triplets from multi-country HTA submissions and clinical guidelines.

    Objective
    - Produce a unified list of PICOs where only truly identical items are merged.
    - Favor over-retention: if there is any doubt, do not merge.

    Core Grouping Policy (Exact-Match Only)
    - Define a grouping key as the tuple:
      key = (
        Population_text_trimmed,
        Intervention_text_trimmed,
        Comparator_text_trimmed
      )
    - Merge only records with the **same key** (byte-for-byte equality after trimming ends).
    - Do NOT perform any other normalization. Specifically, DO NOT:
      - Map synonyms or abbreviations (e.g., SoC ↔ "standard of care"; HCC ↔ hepatocellular carcinoma; "≥" ↔ ">="; "PD-L1" ↔ "PDL1").
      - Convert brand ↔ generic (e.g., Keytruda ↔ pembrolizumab).
      - Reorder tokens or agents ("A + B" ≠ "B + A"; commas/"plus"/"+" are not normalized).
      - Canonicalize punctuation/hyphens/Unicode (e.g., en-dash vs hyphen), casing, spacing, units, or thresholds.
      - Paraphrase, intersect, broaden, or prune wording.

    Comparator Handling (Literal)
    - Keep comparator strings exactly as found, including "standard of care/SoC", "placebo", "supportive care".
    - Do NOT expand "SoC/supportive care/placebo" to specific regimens even if defined in some sources.
    - Do NOT merge class labels (e.g., "TKI", "immunotherapy", "chemotherapy") with specific agents or regimens.

    Oncology "Must-Not-Merge" Guardrails (for awareness; grouping still requires exact text identity)
    - Treat differences in any of the following as **hard separators** (even if only phrasing differs):
      - Disease/indication and anatomic site; stage/extent; resectability/transplant eligibility.
      - Prior therapy **type(s)** (e.g., platinum chemo, IO, TKI, anti-VEGF, LRT such as TACE/TARE/RFA) and **line** (1L, ≥1 prior line, 2L, adjuvant/neoadjuvant).
      - Biomarkers and thresholds (e.g., PD-L1 TPS cutoffs, EGFR/ALK/KRAS, BRAF, MSI-H/dMMR, HER2, BRCA).
      - Histology/grade (adenocarcinoma vs squamous, etc.) or molecular subtype.
      - Treatment context (mono vs combination; maintenance vs induction).
      - Performance status or clinical qualifiers (e.g., ECOG ranges; progressed vs progressed or intolerant).
      - Metastatic site qualifiers (e.g., CNS/brain mets allowed or excluded; extrahepatic spread).
      - Organ-function criteria and tumor-type specifics:
        * HCC: BCLC stage, Child–Pugh/ALBI class, portal vein tumor thrombosis/macrovascular invasion, AFP thresholds, underlying cirrhosis/viral status, prior locoregional therapy (TACE/TARE/RFA), transplant/resectability.
        * (Analogous tumor-specific modifiers if present for other cancers.)

    Text Rules (because grouping is exact)
    - The consolidated Population and Comparator texts must be the **identical literal strings** shared by all grouped instances (after trimming ends). Do not rephrase.
    - Keep entries without a comparator **as their own records**; do not drop them. Do not merge them with entries that have a comparator.

    Metadata Aggregation
    - For each consolidated PICO (i.e., each unique key), union the metadata from all contributing records:
      - Countries (unique)
      - Source_Types (e.g., hta_submission, clinical_guideline) (unique)
      - Source_Refs (document IDs, URLs, or citations if provided)
      - Occurrence_Count (integer)

    Quality & Conservatism
    - When uncertain, **do not merge**.
    - Never infer equivalence across symbols, thresholds, brands/generics, word order, punctuation, or list formatting.
    - Presentation order is cosmetic and must not affect grouping.

    Output JSON structure
    {
      "consolidation_metadata": {
        "timestamp": "ISO 8601",
        "total_consolidated_picos": <int>,
        "source_countries": ["..."],
        "source_types": ["..."],
        "indication": "<free text if provided>"
      },
      "consolidated_picos": [
        {
          "Population": "<literal population text (trimmed ends only)>",
          "Intervention": "<literal intervention text (trimmed ends only)>",
          "Comparator": "<literal comparator text (trimmed ends only, possibly empty)>",
          "Countries": ["..."],
          "Source_Types": ["..."],
          "Source_Refs": ["..."],
          "Occurrence_Count": <int>,
          "Grouping_Key": {
            "population": "<exact>",
            "intervention": "<exact>",
            "comparator": "<exact>"
          }
        }
      ]
    }

    Implementation Notes (do not include in output)
    - Trimming ends = remove leading/trailing whitespace only. Preserve interior whitespace, punctuation, Unicode, and ordering exactly as provided.
    - Treat empty comparator as a value ("").
    - Two records merge only if Population, Intervention, and Comparator are all exactly equal (post trim).
    - It is acceptable and expected that clinically similar items remain unmerged if the strings are not identical.

    """.strip(),

    "outcomes_consolidation_system_prompt": """
    You are an expert clinical outcomes specialist focusing on organizing and categorizing clinical trial and real-world evidence outcomes.

    Task: Consolidate and categorize all outcomes from multiple countries and source types into organized, non-redundant categories while preserving ALL distinct outcomes, especially specific adverse events and measurement instruments.

    Categorization Guidelines:
    1) Group outcomes into major categories: Efficacy, Safety, Quality of Life, Economic, Other
    2) Within each category, create logical subcategories
    3) CRITICAL: Preserve ALL distinct outcomes - do NOT over-consolidate
    4) Maintain the clinical context and specific measurement instruments when available
    5) Order outcomes within categories by clinical importance
    6) Track which countries and source types reported each outcome

    Category Definitions:
    - Efficacy: Survival endpoints, response rates, progression measures, duration of response
    - Safety: Adverse events, toxicity, discontinuations, serious events
    - Quality of Life: Patient-reported outcomes, functional status, symptom measures
    - Economic: Cost-effectiveness, utilities, budget impact, resource utilization
    - Other: Exploratory endpoints, biomarkers, pharmacokinetics

    CRITICAL Outcome Consolidation Rules:

    A) MINIMAL CONSOLIDATION - Preserve Distinctness:
    - Merge ONLY when outcomes are truly synonymous (e.g., "overall survival" and "OS" → "Overall survival (OS)")
    - Treat 'objective response rate' and 'overall response rate' as the same outcome (combine into one entry with acronym ORR)
    - Keep ALL other distinct outcomes separate even if related (e.g., PFS vs. time to progression, ORR vs. duration of response)
    
    B) SPECIFIC ADVERSE EVENTS - List ALL Individually:
    - Create a comprehensive list of EVERY specific adverse event mentioned across all sources
    - DO NOT use phrases like "including" or "such as" followed by examples - list ALL AEs
    - Each distinct AE must appear as a separate item in the list
    - Examples of distinct AEs to preserve separately: diarrhoea, nausea, vomiting, fatigue, neutropenia, anaemia, headache, cough, shortness of breath, constipation, abdominal pain, joint pain, back pain, fever, decreased appetite, rash, QTc prolongation, ALT increase, AST increase, creatinine increase, alopecia
    - Also include general categories like "adverse events (AEs; collected according to CTCAE v5.0)"
    
    C) QUALITY OF LIFE / UTILITY INSTRUMENTS - List ALL Separately:
    - EVERY distinct QoL or utility measurement instrument must be listed as a separate outcome
    - DO NOT merge different instruments even if they measure similar constructs
    - Examples of distinct instruments to preserve separately: EORTC QLQ-C30, EORTC QLQ-LC13, EQ-5D-5L, EQ-5D-3L, SF-36, BPI-SF, FACT-G GP5, PGI-C, visual analogue scale
    - Include the full instrument name/version when available (e.g., "EQ-5D-5L" not just "EQ-5D")
    
    D) Measurement Details:
    - Preserve measurement details when available (e.g., "measured by RECIST 1.1", "CTCAE v5.0")
    - Include specific versions/scales (e.g., "CTCAE v5.0" vs "CTCAE v4.03")
    
    E) Remove Only True Duplicates:
    - Remove arm-specific and numerical/statistical trial results (medians, means, rates/percentages, counts, hazard ratios, odds ratios, relative risks, confidence intervals, p-values, ICER/QALY numeric values)
    - Retain only outcome names and their definitions/measurement methods
    - Remove medicine/drug names attached to outcomes

    Output JSON structure:
    {
        "outcomes_metadata": {
            "timestamp": "ISO timestamp",
            "total_unique_outcomes": number,
            "source_countries": ["list", "of", "countries"],
            "source_types": ["list", "of", "source", "types"],
            "indication": "indication string"
        },
        "consolidated_outcomes": {
            "efficacy": {
                "survival_endpoints": ["outcome1", "outcome2", ...],
                "response_measures": ["outcome1", "outcome2", ...],
                "progression_measures": ["outcome1", "outcome2", ...]
            },
            "safety": {
                "adverse_events": [
                    "List EVERY specific adverse event here as separate items",
                    "diarrhoea", "nausea", "vomiting", "fatigue", "neutropenia", 
                    "anaemia", "headache", "cough", "etc.",
                    "adverse events (general category with grading system if specified)"
                ],
                "serious_events": ["outcome1", "outcome2", ...],
                "discontinuations": ["outcome1", "outcome2", ...]
            },
            "quality_of_life": {
                "patient_reported_outcomes": [
                    "List EVERY QoL/utility instrument here as separate items",
                    "EORTC QLQ-C30", "EORTC QLQ-LC13", "EQ-5D-5L", "SF-36",
                    "BPI-SF", "FACT-G GP5", "visual analogue scale", "etc."
                ],
                "functional_status": ["outcome1", "outcome2", ...],
                "symptom_measures": ["outcome1", "outcome2", ...]
            },
            "economic": {
                "cost_effectiveness": ["outcome1", "outcome2", ...],
                "utilities": ["outcome1", "outcome2", ...],
                "resource_utilization": ["outcome1", "outcome2", ...]
            },
            "other": {
                "exploratory_endpoints": ["outcome1", "outcome2", ...],
                "biomarkers": ["outcome1", "outcome2", ...]
            }
        }
    }

    CRITICAL REMINDERS:
    - The adverse_events list should contain 10-30+ individual AE items (depending on source richness)
    - The patient_reported_outcomes list should contain 5-15+ individual instruments (depending on source richness)
    - DO NOT summarize multiple items with phrases like "adverse events including diarrhoea, nausea, fatigue" - instead list "diarrhoea", "nausea", "fatigue" as separate array items
    - When in doubt about whether to merge, DO NOT MERGE - keep items separate
    """.strip()
}

# --------------------------------
# Simulation configurations
# --------------------------------
SIMULATION_CONFIGS = {
    "base": {
        "name": "Base",
        "description": "Tests impact of larger contextual windows",
        "retrieval_params": {
            "hta_submission": {
                "population_comparator": {"initial_k": 60, "final_k": 15},
                "outcomes": {"initial_k": 60, "final_k": 15}
            },
            "clinical_guideline": {
                "population_comparator": {"initial_k": 50, "final_k": 12},
                "outcomes": {"initial_k": 60, "final_k": 12}
            }
        },
        "extraction_temperature": 0.1,
        "chunk_params": {
            "min_chunk_size": 600,
            "max_chunk_size": 1500
        }
    },
    "sim1": {
        "name": "Lower Temperature",
        "description": "Tests impact of lower temperature (0.01) for more deterministic extraction",
        "retrieval_params": {
            "hta_submission": {
                "population_comparator": {"initial_k": 60, "final_k": 15},
                "outcomes": {"initial_k": 60, "final_k": 15}
            },
            "clinical_guideline": {
                "population_comparator": {"initial_k": 50, "final_k": 12},
                "outcomes": {"initial_k": 60, "final_k": 12}
            }
        },
        "extraction_temperature": 0.01,
        "chunk_params": {
            "min_chunk_size": 600,
            "max_chunk_size": 1500
        }
    },
    "sim2": {
        "name": "Higher Temperature",
        "description": "Tests impact of higher temperature (0.3) for more varied extraction",
        "retrieval_params": {
            "hta_submission": {
                "population_comparator": {"initial_k": 60, "final_k": 15},
                "outcomes": {"initial_k": 60, "final_k": 15}
            },
            "clinical_guideline": {
                "population_comparator": {"initial_k": 50, "final_k": 12},
                "outcomes": {"initial_k": 60, "final_k": 12}
            }
        },
        "extraction_temperature": 0.3,
        "chunk_params": {
            "min_chunk_size": 600,
            "max_chunk_size": 1500
        }
    },
    "sim3": {
        "name": "Reduced Retrieval",
        "description": "Tests impact of reduced retrieval parameters",
        "retrieval_params": {
            "hta_submission": {
                "population_comparator": {"initial_k": 45, "final_k": 10},
                "outcomes": {"initial_k": 45, "final_k": 10}
            },
            "clinical_guideline": {
                "population_comparator": {"initial_k": 40, "final_k": 8},
                "outcomes": {"initial_k": 40, "final_k": 8}
            }
        },
        "extraction_temperature": 0.1,
        "chunk_params": {
            "min_chunk_size": 600,
            "max_chunk_size": 1500
        }
    },
    "sim4": {
        "name": "Reduced Retrieval (Replicate)",
        "description": "Replicates sim4 to test consistency with reduced retrieval parameters",
        "retrieval_params": {
            "hta_submission": {
                "population_comparator": {"initial_k": 45, "final_k": 10},
                "outcomes": {"initial_k": 45, "final_k": 10}
            },
            "clinical_guideline": {
                "population_comparator": {"initial_k": 40, "final_k": 8},
                "outcomes": {"initial_k": 40, "final_k": 8}
            }
        },
        "extraction_temperature": 0.1,
        "chunk_params": {
            "min_chunk_size": 600,
            "max_chunk_size": 1500
        }
    },
    "sim5": {
        "name": "Fine grained Chunks",
        "description": "Tests impact of larger contextual windows",
        "retrieval_params": {
            "hta_submission": {
                "population_comparator": {"initial_k": 60, "final_k": 15},
                "outcomes": {"initial_k": 60, "final_k": 15}
            },
            "clinical_guideline": {
                "population_comparator": {"initial_k": 50, "final_k": 12},
                "outcomes": {"initial_k": 60, "final_k": 12}
            }
        },
        "extraction_temperature": 0.1,
        "chunk_params": {
            "min_chunk_size": 400,
            "max_chunk_size": 1000
        }
    },
    "sim6": {
        "name": "Comprehensive Chunks",
        "description": "Tests impact of larger contextual windows",
        "retrieval_params": {
            "hta_submission": {
                "population_comparator": {"initial_k": 60, "final_k": 15},
                "outcomes": {"initial_k": 60, "final_k": 15}
            },
            "clinical_guideline": {
                "population_comparator": {"initial_k": 50, "final_k": 12},
                "outcomes": {"initial_k": 60, "final_k": 12}
            }
        },
        "extraction_temperature": 0.1,
        "chunk_params": {
            "min_chunk_size": 800,
            "max_chunk_size": 2000
        }
    }
}