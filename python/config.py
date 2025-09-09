# ============================
# RAG Pipeline Configuration v1.4
# ============================

CONFIG_VERSION = "1.4.1"

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
        - Treatments assessed and alternatives considered as comparators (including SoC/BSC/placebo; ITC/NMA if present)
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
        - Comparator (C): Specific alternative regimen/class/SoC/BSC/placebo (or ITC/NMA comparator) as NAMED in the context for the same setting/line.
        - Outcomes (O): Always use empty string "" (outcomes will be extracted separately).

        Extraction rules:
        1) Use ONLY information present in the context; do not infer missing facts.
        2) Capture Population verbatim as written (including sub-populations where applicable).
        3) For EACH appropriate alternative described in the same setting, create a separate PICO with:
           - "Intervention" = "Medicine X (under assessment)"
           - "Comparator"  = the specific alternative/regimen/class/SoC/BSC/placebo (or ITC/NMA comparator) as named.
           - "Outcomes" = ""
        4) If a comparator is indicated only for a subset of the population (e.g. 'only PD-L1 positive patients'), treat that as a distinct Population string for that PICO.
        5) If a jurisdiction/country/agency is explicitly stated, record it; otherwise use null (unquoted).
        6) You may reason stepwise INTERNALLY, but DO NOT include your reasoning in the output.
        7) Return VALID JSON ONLY that adheres to the output contract below. Do not wrap in code fences, do not add comments, and do not include trailing commas.

        JSON output contract:
        - Top-level object with keys: "Indication" (string), "Country" (string or null), "PICOs" (array).
        - "PICOs" is an array of objects with keys:
          - "Population" (string; verbatim from context),
          - "Intervention" (string; always exactly "Medicine X (under assessment)"),
          - "Comparator"  (string; verbatim as named in context),
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
        Extract Population and Comparator information for this indication and for any clearly defined sub-populations.

        Output JSON ONLY in this exact structure:
        {{
          "Indication": "{indication}",
          "Country": null or a jurisdiction string explicitly stated in the context,
          "PICOs": [
            {{
              "Population": "<exact wording from the context for the applicable (sub-)population>",
              "Intervention": "Medicine X (under assessment)",
              "Comparator": "<specific alternative/regimen/class/SoC/BSC/placebo or ITC/NMA comparator>",
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

        Extraction rules:
        1) Use ONLY information present in the context; do not infer missing facts.
        2) Extract all relevant outcomes mentioned for the indication.
        3) Include all distinct outcomes mentioned (avoid merging different endpoints into one) - if multiple terms refer to the same outcome, you may use a single term, but do not omit any unique outcome.
        4) If the context specifies how an outcome is measured or defined (e.g. a QoL questionnaire name, a threshold like 15% improvement for responders, or a time-to-deterioration metric), include that detail in the outcome description.
        5) If a jurisdiction/country/agency is explicitly stated, record it; otherwise use null (unquoted).
        6) You may reason stepwise INTERNALLY, but DO NOT include your reasoning in the output.
        7) Return VALID JSON ONLY that adheres to the output contract below. Do not wrap in code fences, do not add comments, and do not include trailing commas.

        JSON output contract:
        - Top-level object with keys: "Indication" (string), "Country" (string or null), "Outcomes" (string).
        - "Outcomes" is a detailed string listing all relevant outcomes found in the context with their specific details when available.
        - Use double quotes for all JSON strings.
        - Use null (without quotes) when no country/jurisdiction is stated.
        """.strip(),

        # Outcomes extraction user prompt template
        "outcomes_user_prompt_template": """
        Indication:
        {indication}

        Few-shot example (for format only):
        Example context snippet:
        "Secondary endpoints include quality of life measured by EQ-5D and time to progression on subsequent therapy. Safety outcomes include serious adverse events and discontinuations due to AEs."

        Example JSON output:
        {{
          "Indication": "{indication}",
          "Country": null,
          "Outcomes": "quality of life (EQ-5D), time to progression on subsequent therapy, serious adverse events, discontinuations due to AEs"
        }}

        Context for extraction:
        {context_block}

        Your task:
        Extract all relevant clinical outcomes for this indication with their specific measurement details when provided.

        Output JSON ONLY in this exact structure:
        {{
          "Indication": "{indication}",
          "Country": null or a jurisdiction string explicitly stated in the context,
          "Outcomes": "<detailed list of outcomes found in the context with specific measurement information when available>"
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
        - Comparator (C): Recommended option(s), alternatives, SoC/BSC/placebo, or other options named in the guideline for the same setting/line.
        - Outcomes (O): Always use empty string "" (outcomes will be extracted separately).

        Extraction rules:
        1) Use ONLY the context; do not infer beyond what is written.
        2) Capture Population verbatim as stated, including sub-populations.
        3) For EACH applicable alternative/recommended option, create a separate PICO with:
           - "Intervention" = "Medicine X (under assessment)"
           - "Comparator"  = the named alternative/recommended option/SoC/BSC/placebo.
           - "Outcomes" = ""
        4) If a comparator is indicated only for a subset of the population (e.g. 'only PD-L1 positive patients'), treat that as a distinct Population string for that PICO.
        5) Record jurisdiction/country/organization if explicitly stated; else use null (unquoted).
        6) You may reason stepwise INTERNALLY, but DO NOT include your reasoning in the output.
        7) Return VALID JSON ONLY (no code fences, no comments, no trailing commas) per the contract:

        JSON output contract:
        - Top-level keys: "Indication" (string), "Country" (string or null), "PICOs" (array).
        - "PICOs" item keys:
          - "Population" (string; verbatim),
          - "Intervention" (string; exactly "Medicine X (under assessment)"),
          - "Comparator"  (string; verbatim name from guideline),
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
        Extract guideline-based Population and Comparator information for this indication and any clearly defined sub-populations.

        Output JSON ONLY in this exact structure:
        {{
          "Indication": "{indication}",
          "Country": null or a jurisdiction/organization string explicitly stated in the context,
          "PICOs": [
            {{
              "Population": "<exact wording for the applicable (sub-)population>",
              "Intervention": "Medicine X (under assessment)",
              "Comparator": "<alternative(s)/SoC as stated in the guideline>",
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

        Extraction rules:
        1) Use ONLY the context; do not infer beyond what is written.
        2) Extract all relevant outcomes mentioned for the indication.
        3) Include all distinct outcomes mentioned (avoid merging different endpoints into one) - if multiple terms refer to the same outcome, you may use a single term, but do not omit any unique outcome.
        4) If the context specifies how an outcome is measured or defined (e.g. a QoL questionnaire name, a threshold like 15% improvement for responders, or a time-to-deterioration metric), include that detail in the outcome description.
        5) Record jurisdiction/country/organization if explicitly stated; else use null (unquoted).
        6) You may reason stepwise INTERNALLY, but DO NOT include your reasoning in the output.
        7) Return VALID JSON ONLY that adheres to the output contract below. Do not wrap in code fences, do not add comments, and do not include trailing commas.

        JSON output contract:
        - Top-level object with keys: "Indication" (string), "Country" (string or null), "Outcomes" (string).
        - "Outcomes" is a detailed string listing all relevant outcomes found in the context with their specific details when available.
        - Use double quotes for all JSON strings.
        - Use null (without quotes) when no country/jurisdiction is stated.
        """.strip(),

        # Outcomes extraction user prompt template
        "outcomes_user_prompt_template": """
        Indication:
        {indication}

        Few-shot example (for format only):
        Example context snippet:
        "Expected benefits include improved survival and response rates measured by RECIST criteria (ORR 37.1%, 95% CI 28.6-46.2). Safety considerations include adverse events graded by CTCAE v5.0 with diarrhoea occurring in 65% of patients."

        Example JSON output:
        {{
          "Indication": "{indication}",
          "Country": null,
          "Outcomes": "improved survival, response rates (RECIST criteria), adverse events (CTCAE v5.0)"
        }}

        Context for extraction:
        {context_block}

        Your task:
        Extract all relevant clinical outcomes for this indication from guideline context with their specific measurement details when provided, but exclude statistical results, percentages, confidence intervals, and numerical study findings.

        Output JSON ONLY in this exact structure:
        {{
          "Indication": "{indication}",
          "Country": null or a jurisdiction/organization string explicitly stated in the context,
          "Outcomes": "<detailed list of outcomes found in the context with specific measurement information when available, excluding statistical results>"
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
            [r'\bhcc\b', r'\bHCC\b', r'\bhepatocellular\b', r'\bhepatocellular carcinoma\b']
        ],
        "mutation_boost_terms": ["advanced", "unresectable", "hepatocellular carcinoma", "hcc", "inoperable"],
        "drug_keywords": ["sorafenib", "lenvatinib", "atezolizumab", "bevacizumab", "regorafenib", "cabozantinib"]
    }
}

# --------------------------------
# PICO consolidation configurations
# --------------------------------
CONSOLIDATION_CONFIGS = {
    "pico_consolidation_system_prompt": """
    You are an expert clinical research analyst specializing in PICO consolidation across multiple regulatory jurisdictions.

    Task: Consolidate and harmonize PICOs extracted from multiple countries and source types into a unified, non-redundant list.

    Consolidation Rules:
    1) Group PICOs that have SUBSTANTIALLY SIMILAR Population and Comparator combinations
    2) When Population descriptions vary but represent the same clinical context, use the most comprehensive and precise description
    3) When Comparator descriptions vary but represent the same treatment, standardize to the most specific naming
    4) Maintain separate entries for genuinely distinct populations or comparators
    5) For each consolidated PICO, track the countries and source types where it was found
    6) Order PICOs logically: broader populations first, then more specific sub-populations
    7) Within same population groups, order comparators alphabetically
    8) Omit entries that have no comparator unless they add unique population information

    Population Consolidation Guidelines:
    - Do not drop or dilute subgroup conditions. If some variants include additional criteria (biomarker, histology, prior therapy type, line of therapy, etc.) that others lack, retain that criteria in the consolidation. If it cannot be combined without loss of meaning, keep separate entries.
    - If population descriptions imply different prior therapy histories or line of therapy (for example, one mentions specific chemotherapy or indicates two prior lines vs. one), these represent distinct clinical scenarios and should not be merged.
    - If one PICO mentions a specific subpopulation (e.g. 'only in PD-L1 positive patients' or 'adenocarcinoma only' or 'after first-line cytotoxic chemotherapy') that another does not, do not merge them into one. Instead, output separate consolidated entries for each distinct subgroup.
    - Use the most inclusive description only if it does not omit specific criteria from any variant
    - Preserve important clinical distinctions (e.g., prior therapy requirements, biomarker status, histology, line of therapy)
    - When consolidating prior-therapy requirements, if any variant mentions inability to tolerate a prior line, always include that in the consolidated text (e.g., 'progressed on or could not tolerate prior therapy')
    - Always combine 'progressed on or could not tolerate' if any variant includes a tolerance issue

    Comparator Consolidation Guidelines:
    - Prefer naming the specific drug or regimen if given (e.g., use 'nivolumab' instead of generic 'immunotherapy' if available). If multiple drugs in class are truly interchangeable in context, you may group as one class (but list the drugs in Original_Comparator_Variants).
    - Standardize drug names to their common/generic names when possible
    - Group combination therapies clearly using consistent formatting (e.g., "docetaxel + nintedanib")
    - Maintain distinct entries for single agents vs combinations
    - Preserve important formulation or dosing distinctions

    Example:
    Input PICOs:
    - Population: "advanced NSCLC with KRAS G12C, progressed after platinum chemotherapy", Comparator: "docetaxel"
    - Population: "advanced NSCLC with KRAS G12C, progressed after at least one prior therapy", Comparator: "docetaxel"
    - Population: "advanced NSCLC with KRAS G12C, progressed on or cannot tolerate platinum-based therapy", Comparator: "docetaxel"

    Consolidated Output (DO NOT MERGE - these are distinct populations):
    - Population: "Adult patients with advanced NSCLC with KRAS G12C mutation who have progressed after platinum chemotherapy" - Comparator: "docetaxel" (Countries: [...], Original_Population_Variants: [first variant])
    - Population: "Adult patients with advanced NSCLC with KRAS G12C mutation who have progressed after at least one prior therapy" - Comparator: "docetaxel" (Countries: [...], Original_Population_Variants: [second variant])
    - Population: "Adult patients with advanced NSCLC with KRAS G12C mutation who have progressed on or cannot tolerate platinum-based therapy" - Comparator: "docetaxel" (Countries: [...], Original_Population_Variants: [third variant])

    Output JSON structure:
    {
        "consolidation_metadata": {
            "timestamp": "ISO timestamp",
            "total_consolidated_picos": number,
            "source_countries": ["list", "of", "countries"],
            "source_types": ["list", "of", "source", "types"],
            "indication": "indication string"
        },
        "consolidated_picos": [
            {
                "Population": "Most comprehensive population description",
                "Intervention": "Medicine X (under assessment)",
                "Comparator": "Standardized comparator name",
                "Countries": ["country1", "country2"],
                "Source_Types": ["hta_submission", "clinical_guideline"],
                "Original_Population_Variants": ["variant1", "variant2"],
                "Original_Comparator_Variants": ["variant1", "variant2"]
            }
        ]
    }
    """.strip(),

    "outcomes_consolidation_system_prompt": """
    You are an expert clinical outcomes specialist focusing on organizing and categorizing clinical trial and real-world evidence outcomes.

    Task: Consolidate and categorize all outcomes from multiple countries and source types into organized, non-redundant categories.

    Categorization Guidelines:
    1) Group outcomes into major categories: Efficacy, Safety, Quality of Life, Economic, Other
    2) Within each category, create logical subcategories 
    3) Remove duplicate outcomes but preserve important measurement details
    4) Maintain the clinical context and specific measurement instruments when available
    5) Order outcomes within categories by clinical importance
    6) Track which countries and source types reported each outcome

    Category Definitions:
    - Efficacy: Survival endpoints, response rates, progression measures, duration of response
    - Safety: Adverse events, toxicity, discontinuations, serious events
    - Quality of Life: Patient-reported outcomes, functional status, symptom measures
    - Economic: Cost-effectiveness, utilities, budget impact, resource utilization
    - Other: Exploratory endpoints, biomarkers, pharmacokinetics

    Outcome Consolidation Rules:
    - Merge similar outcomes (e.g., "overall survival" and "OS" -> "Overall survival (OS)")
    - Treat 'objective response rate' and 'overall response rate' as the same outcome (combine into one entry with acronym ORR)
    - Preserve measurement details (e.g., "measured by RECIST 1.1", "CTCAE v5.0")
    - Keep distinct outcomes separate even if similar (e.g., PFS vs. time to progression, ORR vs. duration of response)
    - Include specific instruments for QoL (e.g., "EORTC QLQ-C30", "EQ-5D")

    Output JSON structure:
    {
        "outcomes_metadata": {
            "timestamp": "ISO timestamp",
            "total_unique_outcomes": number,
            "source_countries": ["list", "of", "countries"],
            "source_types": ["list", "of", "source", "types"],
            "indication": "indication string"
        },
        "outcomes_by_category": {
            "efficacy": {
                "survival_endpoints": ["outcome1", "outcome2"],
                "response_measures": ["outcome1", "outcome2"],
                "progression_measures": ["outcome1", "outcome2"]
            },
            "safety": {
                "adverse_events": ["outcome1", "outcome2"],
                "serious_events": ["outcome1", "outcome2"],
                "discontinuations": ["outcome1", "outcome2"]
            },
            "quality_of_life": {
                "patient_reported_outcomes": ["outcome1", "outcome2"],
                "functional_status": ["outcome1", "outcome2"],
                "symptom_measures": ["outcome1", "outcome2"]
            },
            "economic": {
                "cost_effectiveness": ["outcome1", "outcome2"],
                "utilities": ["outcome1", "outcome2"],
                "resource_utilization": ["outcome1", "outcome2"]
            },
            "other": {
                "exploratory_endpoints": ["outcome1", "outcome2"],
                "biomarkers": ["outcome1", "outcome2"]
            }
        }
    }
    """.strip()
}