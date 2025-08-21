# ============================
# RAG Pipeline Configuration v1.2
# ============================
# Goals:
# - Generic, minimally-steering retrieval & prompting
# - Only external runtime input: {indication}
# - Intervention is NOT taken from documents; always "Medicine X (under assessment)"
# - Support extraction of populations and sub-populations (exactly as stated)
# - Clean separation of generic prompting from case adapters

CONFIG_VERSION = "1.2.0"

# --------------------------------
# Source type configurations (generic)
# --------------------------------
SOURCE_TYPE_CONFIGS = {
    "hta_submission": {
        # More precise retrieval query (parameterized only by {indication})
        "query_template": """
        Find passages that specify PICO (Population, Intervention, Comparator, Outcomes)
        relevant to: {indication}.

        Prioritize sections that explicitly state:
        - Population: disease/stage, biomarker/testing, and line of therapy (e.g., post-progression, after ≥1 prior systemic therapy / second-line or later), plus any inclusion/exclusion and sub-populations.
        - Comparators: specific alternatives/regimens/classes (chemotherapy, immunotherapy, targeted agents), SoC/BSC/placebo, and any ITC/NMA comparators.
        - Outcomes: numerical clinical endpoints (OS, PFS, ORR, DoR), safety, QoL, and economic/utilities if present.

        Focus strictly on text explicitly tied to the stated indication or clearly defined sub-populations thereof.
        Prefer passages that include both the disease label and biomarker terminology when applicable.
        """.strip(),

        # Anchors to boost clearly relevant sections (expanded to catch results/benefit)
        "default_headings": [
            "pico", "scope of assessment", "population", "line of therapy",
            "comparator", "comparators considered", "comparator rationale",
            "treatment", "intervention", "therapy", "standard of care", "best supportive care", "placebo",
            "clinical evidence", "indirect treatment comparison", "network meta-analysis",
            "outcomes", "outcome measures", "results", "effectiveness", "efficacy", "safety",
            "quality of life", "clinical benefit", "benefit", "claimed benefit", "economic", "utilities"
        ],

        # Keep neutral; no drug steering by default
        "default_drugs": [],

        "system_prompt": """
        You are an oncology-focused HTA analyst.
        From the provided context, extract PICO entries for the specified indication.
        The intervention under evaluation is not defined in the documents; treat it as:
        "Medicine X (under assessment)".

        Rules:
        1) Use ONLY information present in the context; do not infer missing facts.
        2) Capture the Population EXACTLY as written (disease/stage, biomarker/testing if any,
           prior therapy/line, inclusion/exclusion). Include sub-populations when a PICO applies
           to a narrower group; do not assume the population equals the indication.
        3) For EACH appropriate alternative described in the same setting,
           create a separate PICO where:
             - "Intervention" = "Medicine X (under assessment)"
             - "Comparator" = the specific alternative/regimen/class/SoC/BSC/placebo (or ITC/NMA comparator) as named in the text
        4) List Outcomes reported (e.g., OS, PFS, ORR, DoR, safety, QoL, economic/utilities if present).
        5) If a jurisdiction/country/agency is explicitly stated, record it; otherwise use null.
        6) Do NOT include your reasoning. Return valid JSON ONLY, in the requested format.
        """.strip(),

        "user_prompt_template": """
        Indication:
        {indication}

        Context:
        {context_block}

        Task:
        Extract all relevant PICO entries for this indication and for any clearly defined sub-populations.

        Output valid JSON ONLY in this exact format:
        {{
          "Indication": "{indication}",
          "Country": "[Jurisdiction explicitly stated in the context, or null]",
          "PICOs": [
            {{
              "Population": "[Exact wording from the context for the applicable (sub-)population]",
              "Intervention": "Medicine X (under assessment)",
              "Comparator": "[Specific alternative regimen/class/SoC/BSC/placebo or ITC/NMA comparator]",
              "Outcomes": "[Outcomes reported in the context; list as stated]"
            }}
            # additional PICO objects as needed
          ]
        }}
        """.strip()
    },

    "clinical_guideline": {
        # General retrieval query (parameterized only by {indication})
        "query_template": """
        Find passages with treatment recommendations relevant to: {indication}.
        Prefer content that states:
        - Applicable populations and sub-populations (biomarkers/testing, prior therapy/line, inclusion/exclusion)
        - Recommended options and alternatives/SoC that could serve as comparators
        - Outcomes/expected benefits/harms and any evidence strength/level if provided
        """.strip(),

        "default_headings": [
            "recommendation", "treatment", "therapy", "algorithm", "guideline",
            "biomarker", "molecular testing",
            "line of therapy", "subsequent therapy", "post-progression",
            "targeted therapy", "immunotherapy", "chemotherapy",
            "evidence level", "strength of recommendation", "practice point", "expected outcomes"
        ],

        "default_drugs": [],

        "system_prompt": """
        You are an oncology guideline specialist.
        From the provided context, extract PICO entries for the specified indication.
        The intervention under evaluation is not defined in the guidelines; treat it as:
        "Medicine X (under assessment)".

        Rules:
        1) Use ONLY the context; do not infer beyond what is written.
        2) Capture the Population EXACTLY as stated, including any sub-populations.
        3) For EACH recommended option applicable to the setting:
           - "Intervention" = "Medicine X (under assessment)"
           - "Comparator"  = the recommended option's alternatives/SoC or other options named in the guideline
        4) Record Outcomes/expected benefits/harms and any evidence rating if present.
        5) Record jurisdiction/country/organization if explicitly stated; else null.
        6) Do NOT include your reasoning. Return valid JSON ONLY, in the requested format.
        """.strip(),

        "user_prompt_template": """
        Indication:
        {indication}

        Context:
        {context_block}

        Task:
        Extract guideline-based PICO entries for this indication and any clearly defined sub-populations.

        Output valid JSON ONLY in this exact format:
        {{
          "Indication": "{indication}",
          "Country": "[Jurisdiction explicitly stated in the context, or null]",
          "PICOs": [
            {{
              "Population": "[Exact wording for the applicable (sub-)population]",
              "Intervention": "Medicine X (under assessment)",
              "Comparator": "[Alternative(s)/SoC as stated in the guideline]",
              "Outcomes": "[Outcomes/expected benefits/harms; include evidence rating if present]"
            }}
            # additional PICO objects as needed
          ]
        }}
        """.strip()
    }
}

# --------------------------------
# Default retrieval parameters (updated HTA params)
# --------------------------------
DEFAULT_RETRIEVAL_PARAMS = {
    "hta_submission": {
        # Narrower K and window to reduce noise while keeping recall acceptable
        "initial_k": 24,
        "final_k": 12,
        "use_section_windows": True,
        "window_size": 1,
        "booster_weights": {
            "heading": 2.0,
            "pico_keywords": 2.0,
            "comparator_keywords": 2.0
        }
        # If your pipeline supports it, consider enabling strict filtering with required terms:
        # "strict_filtering": True,
        # "required_terms": ["NSCLC", "KRAS", "G12C"]
    },
    "clinical_guideline": {
        "initial_k": 60,
        "final_k": 12,
        "strict_filtering": True,
        "use_section_windows": True,
        "window_size": 2
    }
}

# --------------------------------
# Test queries (generic; only {indication} is injected)
# --------------------------------
TEST_QUERIES = {
    "hta_submission": (
        "Retrieve PICO-relevant passages for: {indication}. "
        "Focus on explicit populations/sub-populations (incl. line of therapy such as post-progression/second-line), "
        "alternatives as comparators (SoC/BSC/placebo/ITC/NMA), and reported outcomes (OS, PFS, ORR, DoR, safety, QoL)."
    ),
    "clinical_guideline": (
        "Guideline recommendations relevant to: {indication}, including sub-populations, "
        "alternative options/SoC as comparators, and outcomes/evidence."
    )
}

# --------------------------------
# Case adapters (the ONLY runtime adaptation is 'indication')
# --------------------------------
CASE_CONFIGS = {
    "case_1_nsclc_krasg12c_monotherapy_progressed": {
        "indication": (
            "treatment of adults with advanced non-small cell lung cancer (NSCLC) with a KRAS G12C mutation "
            "who have progressed after ≥1 prior systemic therapy (second-line), monotherapy context"
        )
    },
    "case_2_hcc_advanced_unresectable": {
        "indication": "treatment of patients with advanced or unresectable hepatocellular carcinoma"
    }
}

# --------------------------------
# General configuration
# --------------------------------
GENERAL_CONFIG = {

    "similarity_threshold": 0.7,
    "model": "gpt-4o-mini",
    "language": "auto",
    "return_json_only": True
}
