# ============================
# RAG Pipeline Configuration v1.4 (split retrieval queries)
# ============================
# Goals:
# - Separate retrieval for Population & Comparator vs Outcomes
# - Generic, minimally-steering retrieval & prompting
# - Only external runtime input: {indication}
# - Intervention is NOT taken from documents; always "Medicine X (under assessment)"
# - Support extraction of populations and sub-populations (exactly as stated)
# - Clean separation of generic prompting from case adapters
# - Enhanced mutation-specific retrieval for clinical guidelines

CONFIG_VERSION = "1.4.0"

# --------------------------------
# Source type configurations with split retrieval
# --------------------------------
SOURCE_CONFIGS = {
    "hta_submission": {
        # Population & Comparator retrieval query
        "population_comparator_query_template": """
        Find passages that specify Population and Comparator elements relevant to: {indication}.
        Prefer sections that clearly describe:
        - Population definitions (disease/stage, prior therapy/line, biomarker/testing, inclusion/exclusion, sub-populations)
        - Treatments assessed and alternatives considered as comparators (including SoC/BSC/placebo; ITC/NMA if present)
        - Treatment lines and patient selection criteria
        - Biomarker testing requirements and mutation status
        Focus on text explicitly tied to the indication or clearly defined sub-populations thereof.
        """.strip(),

        # Outcomes retrieval query
        "outcomes_query_template": """
Find passages that specify **clinical outcomes/endpoints** relevant to: {indication}.
Prefer sections with explicit outcome definitions, measures, or results such as:
- Survival: overall survival (OS), progression-free survival (PFS), objective response rate (ORR),
  duration of response (DoR), time-to-response, time to second progression (PFS2),
  and progression of specific sites (e.g., CNS/brain metastases).
- Quality of life / patient-reported outcomes (PROs): disease-specific (e.g., EORTC QLQ-C30 / QLQ-LC13)
  and generic (e.g., EQ-5D, SF-36), including change-from-baseline over time, time to deterioration (TTD),
  and responder analyses.
- Safety / tolerability: adverse events (AEs) overall, serious/severe AEs (e.g., CTCAE Grade ≥3–5),
  deaths due to AEs (Grade 5), treatment discontinuations, interruptions, and dose reductions due to AEs.
Focus on outcome/result sections, endpoint definitions, and tables. Avoid background/comparator-only text
unless it explicitly defines clinical outcomes.

        """.strip(),

        # Legacy combined query (for backward compatibility)
        "query_template": """
        Find passages that specify PICO (Population, Intervention, Comparator, Outcomes)
        relevant to: {indication}.
        Prefer sections that clearly describe:
        - Population definitions (disease/stage, prior therapy/line, biomarker/testing, inclusion/exclusion, sub-populations)
        - Treatments assessed and alternatives considered as comparators (including SoC/BSC/placebo; ITC/NMA if present)
        - Outcomes reported (clinical efficacy, safety, quality of life, economic/utilities)
        Focus on text explicitly tied to the indication or clearly defined sub-populations thereof.
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
        "outcomes_headings": ["outcomes", "endpoints", "efficacy", "safety", "quality of life", "patient-reported outcomes", "PROs",
            "overall survival", "progression-free survival", "response rate", "duration of response",
            "time to response", "time to second progression", "PFS2",
            "time to deterioration", "TTD", "responder analysis", 
            "EORTC", "QLQ-C30", "QLQ-LC13", "EQ-5D", "SF-36",
            "adverse events", "serious adverse events", "grade 3", "grade 4", "grade 5", "CTCAE",
            "tolerability", "discontinuation", "interruption", "dose reduction",
            "brain metastases", "CNS", "central nervous system", "patient reported outcomes", "statistical analysis"],

        # Combined headings (legacy)
        "default_headings": [
            "pico", "scope of assessment", "population", "line of therapy",
            "comparator", "comparators considered", "comparator rationale",
            "treatment", "intervention", "therapy", "standard of care", "best supportive care", "placebo",
            "clinical evidence", "indirect treatment comparison", "network meta-analysis",
            "outcomes", "efficacy", "safety", "quality of life", "economic", "utilities"
        ],

        # Keep neutral; no drug steering by default
        "default_drugs": [],

        "system_prompt": """
        You are an oncology-focused HTA analyst.

        Task: From the provided context, extract PICO entries for the specified indication.
        Treat the intervention under evaluation as a constant string:
        "Medicine X (under assessment)".

        PICO element definitions (use these strictly):
        - Population (P): EXACT wording from the context describing the eligible patients (disease/stage, prior therapy/line, biomarker/testing, inclusion/exclusion). If a narrower sub-population is specified, capture that exact phrasing.
        - Intervention (I): Always "Medicine X (under assessment)" (not taken from the documents).
        - Comparator (C): Specific alternative regimen/class/SoC/BSC/placebo (or ITC/NMA comparator) as NAMED in the context for the same setting/line.
        - Outcomes (O): List EACH distinct clinical outcome explicitly as named in the text. Include survival (e.g., OS, PFS), response metrics (e.g., ORR, DoR, time-to-response, PFS2), quality of life/PROs (e.g., EORTC QLQ-C30/LC13; EQ-5D; SF-36; including change-from-baseline, TTD, responder analyses), and safety/tolerability (overall AEs; serious/severe AEs such as CTCAE Grade 3–5; deaths due to AEs; treatment discontinuations/interruptions/dose reductions). Do not invent outcomes.

        Extraction rules:
        1) Use ONLY information present in the context; do not infer missing facts.
        2) Capture Population verbatim as written (including sub-populations where applicable).
        3) For EACH appropriate alternative described in the same setting, create a separate PICO with:
           - "Intervention" = "Medicine X (under assessment)"
           - "Comparator"  = the specific alternative/regimen/class/SoC/BSC/placebo (or ITC/NMA comparator) as named.
        4) When outcomes are present, DO NOT merge different outcomes under broad labels (e.g., do not replace "serious adverse events" and "deaths due to AEs" with "safety"). List each outcome separately and keep any acronyms and instrument names as they appear.
5) If a jurisdiction/country/agency is explicitly stated, record it; otherwise use null (unquoted).
        6) You may reason stepwise INTERNALLY, but DO NOT include your reasoning in the output.
        7) Return VALID JSON ONLY that adheres to the output contract below. Do not wrap in code fences, do not add comments, and do not include trailing commas.

        JSON output contract:
        - Top-level object with keys: "Indication" (string), "Country" (string or null), "PICOs" (array).
        - "PICOs" is an array of objects with keys:
          - "Population" (string; verbatim from context),
          - "Intervention" (string; always exactly "Medicine X (under assessment)"),
          - "Comparator"  (string; verbatim as named in context),
          - "Outcomes"    (string; concise list/sentence from context; if none are stated, use an empty string "").
        - Use double quotes for all JSON strings.
        - Use null (without quotes) when no country/jurisdiction is stated.
        """.strip(),

        "user_prompt_template": """
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
              "Outcomes": "overall survival; progression-free survival; objective response rate; duration of response; time-to-response; time to second progression; quality of life (EORTC QLQ-C30/QLQ-LC13: change from baseline, time to deterioration, responder analyses; EQ-5D; SF-36); adverse events (all grades); serious adverse events (CTCAE Grade 3–5); deaths related to AEs; treatment discontinuations due to AEs; treatment interruptions due to AEs; dose reductions due to AEs"
            }}
          ]
        }}

        Context for extraction:
        {context_block}

        Your task:
        Extract all relevant PICO entries for this indication and for any clearly defined sub-populations.

        Output JSON ONLY in this exact structure:
        {{
          "Indication": "{indication}",
          "Country": null or a jurisdiction string explicitly stated in the context,
          "PICOs": [
            {{
              "Population": "<exact wording from the context for the applicable (sub-)population>",
              "Intervention": "Medicine X (under assessment)",
              "Comparator": "<specific alternative/regimen/class/SoC/BSC/placebo or ITC/NMA comparator>",
              "Outcomes": "<outcomes reported in the context; if none are stated, use an empty string>"
            }}
          ]
        }}
        """.strip()
    },

    "clinical_guideline": {
        # Population & Comparator retrieval query
        "population_comparator_query_template": """
        Find treatment recommendations with population and comparator information relevant to: {indication}.
        Prefer content that states:
        - Applicable populations and sub-populations (biomarkers/testing, prior therapy/line, inclusion/exclusion)
        - Recommended treatment options and alternatives/SoC that could serve as comparators
        - Treatment sequencing and line of therapy considerations
        - Patient selection criteria and biomarker requirements
        Focus on guideline recommendations for specific patient populations and treatment alternatives.
        """.strip(),

        # Outcomes retrieval query
        "outcomes_query_template": """
Find **guideline recommendations** that specify outcome expectations for: {indication}.
Prefer content that describes:
- Expected clinical benefits and efficacy outcomes (OS, PFS, ORR, DoR, time-to-response, PFS2,
  and site-specific progression such as CNS/brain metastases when discussed).
- Safety profiles and adverse events (overall AEs; serious/severe AEs such as CTCAE Grade ≥3–5;
  deaths due to AEs; discontinuations/interruptions/dose reductions due to AEs).
- Quality of life / PRO impacts (disease-specific instruments like EORTC QLQ-C30/LC13; generic instruments like EQ-5D, SF-36),
  including change-from-baseline, time to deterioration (TTD), and responder analyses when available.
- Evidence strength/level and recommendation grades where relevant.
Focus on outcome expectations, endpoint definitions, and evidence quality; avoid general background.
        """.strip(),

        # Legacy combined query
        "query_template": """
        Find guideline PICO elements relevant to: {indication}.
        Prioritize clear statements about:
        - Populations/sub-populations and patient selection criteria
        - Recommended treatment options/alternatives/SoC for the same setting/line
        - Outcomes/expected benefits/harms and evidence grading if provided
        """.strip(),

        # Guideline anchors
        "population_headings": [
            "population", "patient group", "biomarker", "testing", "line of therapy", "setting",
            "eligibility", "inclusion", "exclusion", "disease stage", "sub-population"
        ],
        "comparator_headings": [
            "recommendation", "treatment options", "alternatives", "standard of care", "practice point",
            "preferred regimen", "other options", "combination", "monotherapy", "maintenance"
        ],
        "outcomes_headings": [
            "outcomes", "expected outcomes", "benefits", "harms", "adverse events", "tolerability",
            "quality of life", "patient-reported outcomes", "PROs", "evidence level", "strength of recommendation",
            "overall survival", "progression-free survival", "response rate", "duration of response",
            "time to response", "time to deterioration", "TTD", "responder", "EORTC", "EQ-5D", "SF-36"
        ],

        "default_drugs": [],

        "system_prompt": """
        You are an oncology guideline specialist.

        Task: From the provided context, extract PICO entries for the specified indication.
        Treat the intervention under evaluation as a constant string:
        "Medicine X (under assessment)".

        PICO element definitions (use these strictly):
        - Population (P): EXACT wording from the context describing the eligible patients. Include narrower sub-populations exactly as written when applicable.
        - Intervention (I): Always "Medicine X (under assessment)".
        - Comparator (C): Recommended option(s), alternatives, SoC/BSC/placebo as named for the same setting/line. Include other options named in the guideline for the same setting/line.
        - Outcomes (O): List EACH distinct clinical outcome or expected benefit/harms explicitly as named. Include survival (OS, PFS), response metrics (ORR, DoR, time-to-response, PFS2), quality of life/PROs (EORTC QLQ-C30/LC13; EQ-5D; SF-36; including change-from-baseline, TTD, responder analyses), and safety/tolerability (overall AEs; serious/severe AEs such as CTCAE Grade 3–5; deaths due to AEs; discontinuations/interruptions/dose reductions). Do not invent outcomes.

        Extraction rules:
        1) Use ONLY the context; do not infer beyond what is written.
        2) Capture Population verbatim as stated, including sub-populations.
        3) For EACH applicable alternative/recommended option, create a separate PICO with:
           - "Intervention" = "Medicine X (under assessment)"
           - "Comparator"  = the named alternative/recommended option/SoC/BSC/placebo.
        4) When outcomes are present, do NOT merge different outcomes under broad labels; list them separately and keep acronyms/instrument names.
5) Record jurisdiction/country/organization if explicitly stated; else use null (unquoted).
        6) You may reason stepwise INTERNALLY, but DO NOT include your reasoning in the output.
        7) Return VALID JSON ONLY (no code fences, no comments, no trailing commas) per the contract:

        JSON output contract:
        - Top-level keys: "Indication" (string), "Country" (string or null), "PICOs" (array).
        - "PICOs" item keys:
          - "Population" (string; verbatim),
          - "Intervention" (string; exactly "Medicine X (under assessment)"),
          - "Comparator"  (string; verbatim name from guideline),
          - "Outcomes"    (string; concise list/sentence from context; empty string "" if none).
        - Use double quotes for strings and null (unquoted) when country is absent.
        """.strip(),

        "user_prompt_template": """
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
              "Outcomes": "overall survival; progression-free survival; objective response rate; duration of response; time-to-response; quality of life (EORTC QLQ-C30/QLQ-LC13: change from baseline, time to deterioration, responder analyses; EQ-5D; SF-36); adverse events (all grades); serious adverse events (CTCAE Grade 3–5); treatment discontinuations due to AEs"
            }}
          ]
        }}

        Context for extraction:
        {context_block}

        Your task:
        Extract guideline-based PICO entries for this indication and any clearly defined sub-populations.

        Output JSON ONLY in this exact structure:
        {{
          "Indication": "{indication}",
          "Country": null or a jurisdiction/organization string explicitly stated in the context,
          "PICOs": [
            {{
              "Population": "<exact wording for the applicable (sub-)population>",
              "Intervention": "Medicine X (under assessment)",
              "Comparator": "<alternative(s)/SoC as stated in the guideline>",
              "Outcomes": "<outcomes/expected benefits/harms; if none are stated, use an empty string>"
            }}
          ]
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
            "initial_k": 80,
            "final_k": 25,
            "strict_filtering": True,
            "use_section_windows": True,
            "window_size": 3,
            "booster_weights": {
                "heading": 2.0,
                "outcomes_keywords": 3.5,
                "efficacy_keywords": 3.0,
                "safety_keywords": 2.5,
                "endpoint_keywords": 4.0
            }
        },
        # Legacy combined parameters
        "initial_k": 40,
        "final_k": 15,
        "use_section_windows": True,
        "window_size": 2,
        "booster_weights": {
            "heading": 2.0,
            "pico_keywords": 2.0,
            "comparator_keywords": 2.0,
            "mutation_keywords": 3.0
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
                "heading": 2.5,
                "population_keywords": 3.0,
                "comparator_keywords": 3.0,
                "mutation_keywords": 4.0,
                "biomarker_keywords": 3.5
            }
        },
        "outcomes": {
            "initial_k": 90,
            "final_k": 20,
            "strict_filtering": True,
            "use_section_windows": True,
            "window_size": 3,
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
        "required_terms_clinical": None,
        "mutation_boost_terms": [],
        "drug_keywords": ["sorafenib", "lenvatinib", "atezolizumab", "bevacizumab", "regorafenib", "cabozantinib"]
    }
}

# --------------------------------
# Example adapter values used in prompts (safe defaults)
# --------------------------------
PROMPT_EXAMPLE_DEFAULTS = {
    "example_comparator": "docetaxel"
}
