import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict, Counter
from pathlib import Path
import os
from datetime import datetime
import matplotlib.patches as mpatches
import glob

try:
    import geopandas as gpd
    from shapely.geometry import Point
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False
    print("Warning: geopandas not available. Install with: pip install geopandas")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: plotly not available. Install with: pip install plotly")

try:
    from matplotlib_venn import venn2, venn2_circles
    VENN_AVAILABLE = True
except ImportError:
    VENN_AVAILABLE = False
    print("Warning: matplotlib-venn not available. Install with: pip install matplotlib-venn")

try:
    import squarify
    SQUARIFY_AVAILABLE = True
except ImportError:
    SQUARIFY_AVAILABLE = False
    print("Warning: squarify not available. Install with: pip install squarify")


class TranslationAnalyzer:
    """Analyzer for translation metadata and quality scores"""
    
    def __init__(self, translated_path="data/text_translated"):
        self.translated_path = Path(translated_path)
        self.translation_data = []
        
    def load_translation_metadata(self):
        """Load translation metadata from all translated documents"""
        print("Loading translation metadata from documents...")
        
        json_files = list(self.translated_path.rglob("*.json"))
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                if '_translation_metadata' not in data:
                    continue
                    
                metadata = data['_translation_metadata']
                
                path_parts = json_file.parts
                case = None
                source_type = None
                
                for part in path_parts:
                    if part.lower() in ['nsclc', 'hcc', 'sclc', 'breast', 'lung']:
                        case = part.upper()
                    if part.lower() in ['hta submissions', 'clinical guidelines']:
                        source_type = 'hta_submission' if 'hta' in part.lower() else 'clinical_guideline'
                
                doc_info = {
                    'file_name': json_file.name,
                    'case': case,
                    'source_type': source_type,
                    'country': data.get('country', 'Unknown'),
                    'metadata': metadata
                }
                
                self.translation_data.append(doc_info)
                
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
                continue
        
        print(f"Loaded translation metadata from {len(self.translation_data)} documents")
        
    def extract_quality_scores(self, metadata):
        """Extract all quality scores from metadata"""
        quality_scores = metadata.get('quality_scores', {})
        translation_metadata = metadata.get('translation_metadata', {})
        
        scores = {
            'overall': quality_scores.get('overall', None),
            'processing_time': translation_metadata.get('processing_time_seconds', 
                                                        metadata.get('total_processing_time_seconds', None))
        }
        
        linguistic = quality_scores.get('linguistic', {})
        scores.update({
            'linguistic_fluency': linguistic.get('fluency', None),
            'linguistic_accuracy': linguistic.get('accuracy', None),
            'linguistic_consistency': linguistic.get('consistency', None),
            'linguistic_completeness': linguistic.get('completeness', None),
            'linguistic_composite': linguistic.get('linguistic_composite', None)
        })
        
        domain_specific = quality_scores.get('domain_specific', {})
        scores.update({
            'domain_medical_terminology': domain_specific.get('medical_terminology', None),
            'domain_numerical_integrity': domain_specific.get('numerical_integrity', None),
            'domain_statistical_terms': domain_specific.get('statistical_terms', None),
            'domain_unit_preservation': domain_specific.get('unit_preservation', None),
            'domain_composite': domain_specific.get('domain_composite', None)
        })
        
        structural = quality_scores.get('structural', {})
        scores.update({
            'structural_format_preservation': structural.get('format_preservation', None),
            'structural_document_integrity': structural.get('document_integrity', None),
            'structural_information_architecture': structural.get('information_architecture', None),
            'structural_composite': structural.get('structural_composite', None)
        })
        
        return scores
    
    def calculate_statistics(self, scores_list):
        """Calculate mean and standard deviation for a list of scores"""
        valid_scores = [s for s in scores_list if s is not None]
        
        if not valid_scores:
            return None, None
            
        mean = np.mean(valid_scores)
        std = np.std(valid_scores)
        
        return mean, std
    
    def analyze_group(self, documents, group_name):
        """Analyze translation statistics for a group of documents"""
        if not documents:
            return None
            
        all_scores = defaultdict(list)
        
        for doc in documents:
            scores = self.extract_quality_scores(doc['metadata'])
            for key, value in scores.items():
                if value is not None:
                    all_scores[key].append(value)
        
        results = {
            'group_name': group_name,
            'document_count': len(documents),
            'statistics': {}
        }
        
        for score_type, values in all_scores.items():
            if values:
                mean, std = self.calculate_statistics(values)
                if score_type == 'processing_time':
                    total = sum(values)
                    results['statistics'][score_type] = {
                        'mean': mean,
                        'std': std,
                        'total': total,
                        'count': len(values)
                    }
                else:
                    results['statistics'][score_type] = {
                        'mean': mean,
                        'std': std,
                        'count': len(values)
                    }
        
        return results
    
    def print_statistics(self, results, title):
        """Print formatted statistics"""
        if not results or not results.get('statistics'):
            print(f"\n{title}")
            print("="*80)
            print("No translation data available")
            return
            
        print(f"\n{title}")
        print("="*80)
        print(f"Documents analyzed: {results['document_count']}")
        print()
        
        stats = results['statistics']
        
        if 'overall' in stats:
            print("OVERALL TRANSLATION QUALITY")
            print("-"*50)
            s = stats['overall']
            print(f"  Mean: {s['mean']:.4f}")
            print(f"  SD:   {s['std']:.4f}")
            print(f"  N:    {s['count']}")
            print()
        
        linguistic_keys = [k for k in stats.keys() if k.startswith('linguistic_')]
        if linguistic_keys:
            print("LINGUISTIC SCORES")
            print("-"*50)
            for key in sorted(linguistic_keys):
                s = stats[key]
                label = key.replace('linguistic_', '').replace('_', ' ').title()
                print(f"  {label:20s}: Mean={s['mean']:.4f}, SD={s['std']:.4f}, N={s['count']}")
            print()
        
        domain_keys = [k for k in stats.keys() if k.startswith('domain_')]
        if domain_keys:
            print("DOMAIN-SPECIFIC SCORES")
            print("-"*50)
            for key in sorted(domain_keys):
                s = stats[key]
                label = key.replace('domain_', '').replace('_', ' ').title()
                print(f"  {label:25s}: Mean={s['mean']:.4f}, SD={s['std']:.4f}, N={s['count']}")
            print()
        
        structural_keys = [k for k in stats.keys() if k.startswith('structural_')]
        if structural_keys:
            print("STRUCTURAL SCORES")
            print("-"*50)
            for key in sorted(structural_keys):
                s = stats[key]
                label = key.replace('structural_', '').replace('_', ' ').title()
                print(f"  {label:30s}: Mean={s['mean']:.4f}, SD={s['std']:.4f}, N={s['count']}")
            print()
        
        if 'processing_time' in stats:
            print("PROCESSING TIME")
            print("-"*50)
            s = stats['processing_time']
            print(f"  Mean:  {s['mean']:.2f} seconds")
            print(f"  SD:    {s['std']:.2f} seconds")
            print(f"  Total: {s['total']:.2f} seconds ({s['total']/60:.2f} minutes)")
            print(f"  N:     {s['count']}")
            print()
    
    def run_complete_analysis(self):
        """Run complete translation analysis with all groupings"""
        if not self.translation_data:
            self.load_translation_metadata()
        
        if not self.translation_data:
            print("No translation metadata found")
            return
        
        print("\n" + "="*100)
        print("TRANSLATION QUALITY ANALYSIS")
        print("="*100)
        
        all_results = self.analyze_group(self.translation_data, "All Documents")
        self.print_statistics(all_results, "ALL DOCUMENTS")
        
        hta_docs = [d for d in self.translation_data if d['source_type'] == 'hta_submission']
        if hta_docs:
            hta_results = self.analyze_group(hta_docs, "HTA Submissions")
            self.print_statistics(hta_results, "HTA SUBMISSIONS")
        
        clinical_docs = [d for d in self.translation_data if d['source_type'] == 'clinical_guideline']
        if clinical_docs:
            clinical_results = self.analyze_group(clinical_docs, "Clinical Guidelines")
            self.print_statistics(clinical_results, "CLINICAL GUIDELINES")
        
        nsclc_docs = [d for d in self.translation_data if d['case'] == 'NSCLC']
        if nsclc_docs:
            nsclc_results = self.analyze_group(nsclc_docs, "NSCLC Documents")
            self.print_statistics(nsclc_results, "NSCLC DOCUMENTS")
        
        hcc_docs = [d for d in self.translation_data if d['case'] == 'HCC']
        if hcc_docs:
            hcc_results = self.analyze_group(hcc_docs, "HCC Documents")
            self.print_statistics(hcc_results, "HCC DOCUMENTS")
        
        print("\n" + "="*100)
        print("TRANSLATION QUALITY ANALYSIS COMPLETE")
        print("="*100)


class ComprehensiveOverview:
    """Class to generate comprehensive overview summaries for RAG pipeline analysis"""
    
    def __init__(self):
        self.all_cases_data = {}
    
    def generate_case_overview(self, pico_analyzer, outcome_analyzer, case_name):
        """Generate a nice formatted overview summary for PICOs and Outcomes for a specific case"""
        data_split = getattr(pico_analyzer, 'data_split', 'unknown')
        split_info = f" ({data_split.title()} Set)" if data_split != 'unknown' else ""
        
        print("\n" + "="*100)
        print(f"{case_name.upper()} COMPREHENSIVE ANALYSIS OVERVIEW{split_info}")
        print("="*100)
        
        print("\n" + "🔬 PICO EVIDENCE OVERVIEW")
        print("-" * 50)
        
        case_data = {'picos': 0, 'outcomes': 0, 'pico_countries': set(), 'outcome_countries': set(), 'source_types': set(), 'data_split': data_split}
        
        if not pico_analyzer.picos_df.empty:
            total_picos = len(pico_analyzer.data.get('consolidated_picos', []))
            total_records = len(pico_analyzer.picos_df)
            case_data['picos'] = total_picos
            
            print(f"📊 Total Consolidated PICOs: {total_picos}")
            print(f"📈 Total PICO Records: {total_records}")
            
            if 'Country' in pico_analyzer.picos_df.columns:
                countries = pico_analyzer.picos_df['Country'].value_counts()
                case_data['pico_countries'] = set(countries.index)
                print(f"🌍 Countries Covered: {len(countries)}")
                print("   Top countries by PICOs:")
                for i, (country, count) in enumerate(countries.head(3).items()):
                    print(f"   {i+1}. {country}: {count} PICOs")
            
            if 'Source_Type' in pico_analyzer.picos_df.columns:
                sources = pico_analyzer.picos_df['Source_Type'].value_counts()
                case_data['source_types'].update(sources.index)
                print(f"📋 Source Types: {', '.join(sources.index)}")
                for source, count in sources.items():
                    print(f"   - {source.replace('_', ' ').title()}: {count} records")
            
            if 'Comparator' in pico_analyzer.picos_df.columns:
                comparators = pico_analyzer.picos_df['Comparator'].nunique()
                top_comparator = pico_analyzer.picos_df['Comparator'].mode().iloc[0] if not pico_analyzer.picos_df['Comparator'].mode().empty else 'N/A'
                print(f"⚖️  Unique Comparators: {comparators}")
                print(f"   Most Common: {top_comparator}")
        else:
            print("❌ No PICO data available for analysis")
        
        print("\n" + "🎯 OUTCOMES EVIDENCE OVERVIEW")
        print("-" * 50)
        
        if outcome_analyzer.total_outcomes > 0:
            case_data['outcomes'] = outcome_analyzer.total_outcomes
            
            print(f"📊 Total Outcome Measures: {outcome_analyzer.total_outcomes}")
            print(f"📂 Outcome Categories: {len(outcome_analyzer.data.get('consolidated_outcomes', {}))}")
            
            metadata = outcome_analyzer.data.get('outcomes_metadata', {})
            source_countries = metadata.get('source_countries', [])
            source_types = metadata.get('source_types', [])
            
            if source_countries:
                case_data['outcome_countries'] = set(source_countries)
                print(f"🌍 Countries with Outcomes: {len(source_countries)}")
                print(f"   Countries: {', '.join(source_countries)}")
            
            if source_types:
                case_data['source_types'].update(source_types)
                print(f"📋 Source Types: {', '.join(source_types)}")
                for source in source_types:
                    print(f"   - {source.replace('_', ' ').title()}")
        else:
            print("❌ No outcomes data available for analysis")
        
        print("\n" + "🗺️  COVERAGE SUMMARY")
        print("-" * 50)
        
        pico_countries = set()
        outcome_countries = set()
        
        if not pico_analyzer.picos_df.empty and 'Country' in pico_analyzer.picos_df.columns:
            pico_countries = set(pico_analyzer.picos_df['Country'].unique())
        
        metadata = outcome_analyzer.data.get('outcomes_metadata', {})
        if metadata.get('source_countries'):
            outcome_countries = set(metadata['source_countries'])
        
        all_countries = pico_countries.union(outcome_countries)
        common_countries = pico_countries.intersection(outcome_countries)
        
        print(f"🌐 Total Countries Covered: {len(all_countries)}")
        print(f"🤝 Countries with Both PICOs & Outcomes: {len(common_countries)}")
        print(f"🔬 PICO-Only Countries: {len(pico_countries - outcome_countries)}")
        print(f"🎯 Outcome-Only Countries: {len(outcome_countries - pico_countries)}")
        
        if common_countries:
            print(f"   Countries with complete coverage: {', '.join(sorted(common_countries))}")
        
        print("\n" + "="*100)
        
        case_key = f"{case_name}_{data_split}" if data_split != 'unknown' else case_name
        self.all_cases_data[case_key] = case_data
        
        return case_data
    
    def generate_cross_case_overview(self, all_pico_files, all_outcome_files, output_suffix=""):
        """Generate overview across all cases"""
        split_info = output_suffix.replace('_', ' ').title() if output_suffix else 'All Data'
        print(f"\n🌍 CROSS-CASE ANALYSIS OVERVIEW ({split_info})")
        print("-" * 60)
        
        total_consolidated_picos = 0
        total_outcome_reports = 0
        all_countries_pico = set()
        all_countries_outcome = set()
        all_source_types = set()
        
        for pico_file, case in all_pico_files:
            try:
                pico_analyzer = PICOAnalyzer(str(pico_file))
                if not pico_analyzer.picos_df.empty:
                    case_picos = len(pico_analyzer.data.get('consolidated_picos', []))
                    total_consolidated_picos += case_picos
                    print(f"   {case}: {case_picos} consolidated PICOs")
                    
                    if 'Country' in pico_analyzer.picos_df.columns:
                        all_countries_pico.update(pico_analyzer.picos_df['Country'].unique())
                    if 'Source_Type' in pico_analyzer.picos_df.columns:
                        all_source_types.update(pico_analyzer.picos_df['Source_Type'].unique())
            except Exception as e:
                print(f"   Error loading {case} PICOs: {e}")
        
        for outcome_file, case in all_outcome_files:
            try:
                outcome_analyzer = OutcomeAnalyzer(str(outcome_file))
                if outcome_analyzer.total_outcomes > 0:
                    case_outcomes = outcome_analyzer.total_outcomes
                    total_outcome_reports += case_outcomes
                    print(f"   {case}: {case_outcomes} outcome measures")
                    
                    metadata = outcome_analyzer.data.get('outcomes_metadata', {})
                    if metadata.get('source_countries'):
                        all_countries_outcome.update(metadata['source_countries'])
            except Exception as e:
                print(f"   Error loading {case} outcomes: {e}")
        
        print(f"\n📊 TOTAL ACROSS ALL CASES ({split_info}):")
        print(f"   🔬 Total Consolidated PICOs: {total_consolidated_picos}")
        print(f"   🎯 Total Outcome Measures: {total_outcome_reports}")
        print(f"   🌍 Countries with PICO Evidence: {len(all_countries_pico)}")
        print(f"   🌍 Countries with Outcome Evidence: {len(all_countries_outcome)}")
        print(f"   📋 Source Types Used: {', '.join(all_source_types)}")
        print(f"   🤝 Countries with Both Types: {len(all_countries_pico.intersection(all_countries_outcome))}")
        
        if all_countries_pico:
            print(f"   🔬 PICO Countries: {', '.join(sorted(all_countries_pico))}")
        if all_countries_outcome:
            print(f"   🎯 Outcome Countries: {', '.join(sorted(all_countries_outcome))}")
    
    def get_case_summary(self, case_name):
        """Get stored summary data for a specific case"""
        return self.all_cases_data.get(case_name, {})


class PICOAnalyzer:
    def __init__(self, pico_file_path):
        self.pico_file_path = pico_file_path
        self.data = None
        self.picos_df = None
        self.data_split = self._extract_data_split()
        self.case = self._extract_case_name()
        self.load_data()
        self.prepare_datamatrix()
    
    def _extract_data_split(self):
        """Extract data split from file path."""
        file_path = str(self.pico_file_path).lower()
        if '_train_' in file_path or file_path.endswith('_train.json'):
            return 'train'
        elif '_test_' in file_path or file_path.endswith('_test.json'):
            return 'test'
        else:
            return 'unknown'
    
    def _extract_case_name(self):
        """Extract case name from file path."""
        path_parts = Path(self.pico_file_path).parts
        for part in path_parts:
            if part.upper() in ['NSCLC', 'HCC', 'SCLC', 'BREAST', 'LUNG']:
                return part.upper()
        return 'UNKNOWN'
    
    def load_data(self):
        try:
            with open(self.pico_file_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            split_info = f" ({self.data_split} set)" if self.data_split != 'unknown' else ""
            print(f"Successfully loaded PICO data from {self.pico_file_path}{split_info}")
        except FileNotFoundError:
            raise FileNotFoundError(f"PICO file not found: {self.pico_file_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in PICO file: {e}")
        except Exception as e:
            raise Exception(f"Error loading PICO data: {e}")
    
    def prepare_datamatrix(self):
        pico_records = []
        
        try:
            if 'consolidated_picos' not in self.data:
                print("Warning: 'consolidated_picos' key not found in data")
                self.picos_df = pd.DataFrame()
                return
                
            consolidated_picos = self.data['consolidated_picos']
            
            if not isinstance(consolidated_picos, list):
                print(f"Warning: consolidated_picos is not a list, it's a {type(consolidated_picos)}")
                self.picos_df = pd.DataFrame()
                return
            
            for i, pico in enumerate(consolidated_picos):
                try:
                    if not isinstance(pico, dict):
                        print(f"Warning: PICO item {i} is not a dictionary, it's a {type(pico)}")
                        continue
                    
                    countries = pico.get('Countries', [])
                    if isinstance(countries, str):
                        countries = [countries]
                    elif not isinstance(countries, list):
                        countries = []
                    
                    source_types = pico.get('Source_Types', [])
                    if isinstance(source_types, str):
                        source_types = [source_types]
                    elif not isinstance(source_types, list):
                        source_types = []
                    
                    population = pico.get('Population', 'Unknown')
                    intervention = pico.get('Intervention', 'Unknown')
                    comparator = pico.get('Comparator', 'Unknown')
                    
                    pop_variants = pico.get('Original_Population_Variants', [])
                    comp_variants = pico.get('Original_Comparator_Variants', [])
                    
                    if not isinstance(pop_variants, list):
                        pop_variants = []
                    if not isinstance(comp_variants, list):
                        comp_variants = []
                    
                    for country in countries:
                        for source_type in source_types:
                            record = {
                                'Population': population,
                                'Intervention': intervention,
                                'Comparator': comparator,
                                'Country': country,
                                'Source_Type': source_type,
                                'Population_Variants_Count': len(pop_variants),
                                'Comparator_Variants_Count': len(comp_variants),
                                'Data_Split': self.data_split,
                                'Case': self.case
                            }
                            pico_records.append(record)
                
                except Exception as e:
                    print(f"Error processing PICO item {i}: {e}")
                    continue
            
            self.picos_df = pd.DataFrame(pico_records)
            print(f"Successfully processed {len(pico_records)} PICO records")
            
        except Exception as e:
            print(f"Error in prepare_datamatrix: {e}")
            self.picos_df = pd.DataFrame()
    
    def print_unique_picos_overview(self):
        """Print detailed overview of all unique PICOs found"""
        split_info = f" ({self.data_split.title()} Set)" if self.data_split != 'unknown' else ""
        print("\n" + "🔬 DETAILED PICO EVIDENCE LISTING" + split_info)
        print("=" * 80)
        
        if 'consolidated_picos' not in self.data or not self.data['consolidated_picos']:
            print("❌ No consolidated PICOs available for detailed listing")
            return
            
        consolidated_picos = self.data['consolidated_picos']
        
        print(f"📋 Found {len(consolidated_picos)} unique PICO combinations:\n")
        
        for i, pico in enumerate(consolidated_picos, 1):
            try:
                population = pico.get('Population', 'Not specified')
                intervention = pico.get('Intervention', 'Not specified')
                comparator = pico.get('Comparator', 'Not specified')
                countries = pico.get('Countries', [])
                source_types = pico.get('Source_Types', [])
                
                if isinstance(countries, str):
                    countries = [countries]
                if isinstance(source_types, str):
                    source_types = [source_types]
                
                print(f"PICO #{i:02d}")
                print(f"├─ 👥 Population: {population}")
                print(f"├─ 💊 Intervention: {intervention}")
                print(f"├─ ⚖️  Comparator: {comparator}")
                print(f"├─ 🌍 Countries: {', '.join(countries) if countries else 'Not specified'}")
                print(f"└─ 📋 Sources: {', '.join([s.replace('_', ' ').title() for s in source_types]) if source_types else 'Not specified'}")
                
                pop_variants = pico.get('Original_Population_Variants', [])
                comp_variants = pico.get('Original_Comparator_Variants', [])
                
                if pop_variants and len(pop_variants) > 1:
                    print(f"   📝 Population variants: {len(pop_variants)} found")
                if comp_variants and len(comp_variants) > 1:
                    print(f"   📝 Comparator variants: {len(comp_variants)} found")
                
                print()
                
            except Exception as e:
                print(f"   ❌ Error displaying PICO #{i}: {e}")
                print()
        
        print("=" * 80)
    
    def print_summary_statistics(self):
        split_info = f" ({self.data_split.title()} Set)" if self.data_split != 'unknown' else ""
        print("="*80)
        print(f"PICO ANALYSIS SUMMARY{split_info}")
        print("="*80)
        
        if self.picos_df.empty:
            print("No PICO data available for analysis")
            return
        
        try:
            metadata = self.data.get('consolidation_metadata', {})
            print(f"Analysis Date: {metadata.get('timestamp', 'Unknown')}")
            print(f"Total Consolidated PICOs: {metadata.get('total_consolidated_picos', 'Unknown')}")
            print(f"Source Countries: {', '.join(metadata.get('source_countries', []))}")
            print(f"Source Types: {', '.join(metadata.get('source_types', []))}")
            if self.data_split != 'unknown':
                print(f"Data Split: {self.data_split.title()}")
            print()
            
            print("POPULATION AND COMPARATOR STATISTICS")
            print("-" * 50)
            
            if 'Country' in self.picos_df.columns:
                country_counts = self.picos_df['Country'].value_counts()
                print("PICOs by Country:")
                for country, count in country_counts.items():
                    print(f"  {country}: {count}")
                print()
            
            if 'Source_Type' in self.picos_df.columns:
                source_counts = self.picos_df['Source_Type'].value_counts()
                print("PICOs by Source Type:")
                for source, count in source_counts.items():
                    print(f"  {source}: {count}")
                print()
            
            if 'Comparator' in self.picos_df.columns:
                comparator_counts = self.picos_df['Comparator'].value_counts()
                print("Most Common Comparators:")
                for comp, count in comparator_counts.head(10).items():
                    print(f"  {comp}: {count}")
                print()
            
            if 'Population' in self.picos_df.columns:
                population_types = self.picos_df['Population'].nunique()
                print(f"Unique Population Types: {population_types}")
            
            if 'Comparator' in self.picos_df.columns:
                print(f"Unique Comparators: {self.picos_df['Comparator'].nunique()}")
            print()
            
        except Exception as e:
            print(f"Error generating summary statistics: {e}")
    
    def get_country_comparator_matrix(self):
        if self.picos_df.empty or 'Country' not in self.picos_df.columns or 'Comparator' not in self.picos_df.columns:
            return pd.DataFrame()
        
        try:
            matrix = self.picos_df.pivot_table(
                index='Country', 
                columns='Comparator', 
                values='Intervention',
                aggfunc='count',
                fill_value=0
            )
            return matrix
        except Exception as e:
            print(f"Error creating country-comparator matrix: {e}")
            return pd.DataFrame()


class OutcomeAnalyzer:
    def __init__(self, outcome_file_path):
        self.outcome_file_path = outcome_file_path
        self.data = None
        self.outcomes_df = None
        self.total_outcomes = 0
        self.data_split = self._extract_data_split()
        self.case = self._extract_case_name()
        self.load_data()
        self.prepare_datamatrix()
        
        metadata = self.data.get('outcomes_metadata', {})
        if 'total_unique_outcomes' in metadata:
            self.total_outcomes = metadata['total_unique_outcomes']
            print(f"Using metadata total_unique_outcomes: {self.total_outcomes}")
        else:
            print(f"Using calculated total_outcomes: {self.total_outcomes}")
    
    def _extract_data_split(self):
        """Extract data split from file path."""
        file_path = str(self.outcome_file_path).lower()
        if '_train_' in file_path or file_path.endswith('_train.json'):
            return 'train'
        elif '_test_' in file_path or file_path.endswith('_test.json'):
            return 'test'
        else:
            return 'unknown'
    
    def _extract_case_name(self):
        """Extract case name from file path."""
        path_parts = Path(self.outcome_file_path).parts
        for part in path_parts:
            if part.upper() in ['NSCLC', 'HCC', 'SCLC', 'BREAST', 'LUNG']:
                return part.upper()
        return 'UNKNOWN'
    
    def load_data(self):
        try:
            with open(self.outcome_file_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            split_info = f" ({self.data_split} set)" if self.data_split != 'unknown' else ""
            print(f"Successfully loaded Outcomes data from {self.outcome_file_path}{split_info}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Outcomes file not found: {self.outcome_file_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in Outcomes file: {e}")
        except Exception as e:
            raise Exception(f"Error loading Outcomes data: {e}")
    
    def prepare_datamatrix(self):
        outcome_records = []
        total_outcomes = 0
        
        try:
            if 'consolidated_outcomes' not in self.data:
                print("Warning: 'consolidated_outcomes' key not found in data")
                self.outcomes_df = pd.DataFrame()
                return
                
            consolidated_outcomes = self.data['consolidated_outcomes']
            
            if not isinstance(consolidated_outcomes, dict):
                print(f"Warning: consolidated_outcomes is not a dictionary, it's a {type(consolidated_outcomes)}")
                self.outcomes_df = pd.DataFrame()
                return
            
            metadata = self.data.get('outcomes_metadata', {})
            source_countries = metadata.get('source_countries', [])
            source_types = metadata.get('source_types', [])
            
            for category, subcategories in consolidated_outcomes.items():
                try:
                    if not isinstance(subcategories, dict):
                        print(f"Warning: subcategories for {category} is not a dictionary, it's a {type(subcategories)}")
                        continue
                    
                    for subcategory, outcomes in subcategories.items():
                        try:
                            if not isinstance(outcomes, list):
                                print(f"Warning: outcomes for {category}/{subcategory} is not a list, it's a {type(outcomes)}")
                                continue
                            
                            total_outcomes += len(outcomes)
                            
                            for outcome in outcomes:
                                try:
                                    if isinstance(outcome, str):
                                        outcome_name = outcome
                                        
                                        for country in source_countries:
                                            for source_type in source_types:
                                                record = {
                                                    'Category': category,
                                                    'Subcategory': subcategory,
                                                    'Outcome_Name': outcome_name,
                                                    'Country': country,
                                                    'Source_Type': source_type,
                                                    'Has_Details': False,
                                                    'Data_Split': self.data_split,
                                                    'Case': self.case
                                                }
                                                outcome_records.append(record)
                                                
                                    elif isinstance(outcome, dict):
                                        outcome_name = outcome.get('name', 'Unknown')
                                        has_details = 'details' in outcome and bool(outcome.get('details', []))
                                        
                                        if 'reported_by' in outcome and isinstance(outcome['reported_by'], list):
                                            for report in outcome['reported_by']:
                                                if isinstance(report, dict):
                                                    record = {
                                                        'Category': category,
                                                        'Subcategory': subcategory,
                                                        'Outcome_Name': outcome_name,
                                                        'Country': report.get('country', 'Unknown'),
                                                        'Source_Type': report.get('source_type', 'Unknown'),
                                                        'Has_Details': has_details,
                                                        'Data_Split': self.data_split,
                                                        'Case': self.case
                                                    }
                                                    outcome_records.append(record)
                                        else:
                                            for country in source_countries:
                                                for source_type in source_types:
                                                    record = {
                                                        'Category': category,
                                                        'Subcategory': subcategory,
                                                        'Outcome_Name': outcome_name,
                                                        'Country': country,
                                                        'Source_Type': source_type,
                                                        'Has_Details': has_details,
                                                        'Data_Split': self.data_split,
                                                        'Case': self.case
                                                    }
                                                    outcome_records.append(record)
                                        
                                except Exception as e:
                                    print(f"Error processing outcome in {category}/{subcategory}: {e}")
                                    continue
                                    
                        except Exception as e:
                            print(f"Error processing subcategory {subcategory} in {category}: {e}")
                            continue
                            
                except Exception as e:
                    print(f"Error processing category {category}: {e}")
                    continue
            
            self.total_outcomes = total_outcomes
            self.outcomes_df = pd.DataFrame(outcome_records)
            print(f"Successfully processed {len(outcome_records)} outcome records from {total_outcomes} unique outcomes")
            
        except Exception as e:
            print(f"Error in prepare_datamatrix: {e}")
            self.outcomes_df = pd.DataFrame()
    
    def print_unique_outcomes_overview(self):
        """Print detailed overview of all unique outcomes found"""
        split_info = f" ({self.data_split.title()} Set)" if self.data_split != 'unknown' else ""
        print("\n" + "🎯 DETAILED OUTCOMES EVIDENCE LISTING" + split_info)
        print("=" * 80)
        
        if 'consolidated_outcomes' not in self.data or not self.data['consolidated_outcomes']:
            print("❌ No consolidated outcomes available for detailed listing")
            return
            
        consolidated_outcomes = self.data['consolidated_outcomes']
        metadata = self.data.get('outcomes_metadata', {})
        source_countries = metadata.get('source_countries', [])
        source_types = metadata.get('source_types', [])
        
        total_outcomes = 0
        for category, subcategories in consolidated_outcomes.items():
            if isinstance(subcategories, dict):
                for subcategory, outcomes in subcategories.items():
                    if isinstance(outcomes, list):
                        total_outcomes += len(outcomes)
        
        print(f"📋 Found {total_outcomes} unique outcomes across {len(consolidated_outcomes)} categories:\n")
        
        print("🌍 Coverage Information:")
        if source_countries:
            print(f"├─ Countries: {', '.join(source_countries)}")
        if source_types:
            print(f"└─ Source Types: {', '.join([s.replace('_', ' ').title() for s in source_types])}")
        print()
        
        for category, subcategories in consolidated_outcomes.items():
            try:
                if not isinstance(subcategories, dict):
                    continue
                    
                print(f"📂 {category.replace('_', ' ').upper()}")
                print("─" * 60)
                
                for subcategory, outcomes in subcategories.items():
                    try:
                        if not isinstance(outcomes, list):
                            continue
                            
                        print(f"\n📋 {subcategory.replace('_', ' ').title()} ({len(outcomes)} outcomes):")
                        
                        for i, outcome in enumerate(outcomes, 1):
                            try:
                                if isinstance(outcome, str):
                                    outcome_name = outcome
                                    print(f"  {i:2d}. {outcome_name}")
                                    
                                elif isinstance(outcome, dict):
                                    outcome_name = outcome.get('name', 'Unnamed outcome')
                                    has_details = 'details' in outcome and bool(outcome.get('details', []))
                                    reported_by = outcome.get('reported_by', [])
                                    
                                    countries = set()
                                    source_types_reported = set()
                                    if isinstance(reported_by, list):
                                        for report in reported_by:
                                            if isinstance(report, dict):
                                                if 'country' in report:
                                                    countries.add(report['country'])
                                                if 'source_type' in report:
                                                    source_types_reported.add(report['source_type'])
                                    
                                    print(f"  {i:2d}. {outcome_name}")
                                    if countries:
                                        print(f"      🌍 Countries: {', '.join(sorted(countries))}")
                                    if source_types_reported:
                                        print(f"      📋 Sources: {', '.join([s.replace('_', ' ').title() for s in sorted(source_types_reported)])}")
                                    if has_details:
                                        print(f"      📝 Additional details available")
                                        
                            except Exception as e:
                                print(f"     ❌ Error displaying outcome {i}: {e}")
                        
                    except Exception as e:
                        print(f"   ❌ Error processing subcategory {subcategory}: {e}")
                        
                print("\n")
                
            except Exception as e:
                print(f"❌ Error processing category {category}: {e}")
                print()
        
        print("=" * 80)
    
    def print_summary_statistics(self):
        split_info = f" ({self.data_split.title()} Set)" if self.data_split != 'unknown' else ""
        print("="*80)
        print(f"OUTCOMES ANALYSIS SUMMARY{split_info}")
        print("="*80)
        
        if self.total_outcomes == 0:
            print("No outcomes data available for analysis")
            return
        
        try:
            metadata = self.data.get('outcomes_metadata', {})
            print(f"Analysis Date: {metadata.get('timestamp', 'Unknown')}")
            print(f"Total Unique Outcomes: {metadata.get('total_unique_outcomes', self.total_outcomes)}")
            print(f"Source Countries: {', '.join(metadata.get('source_countries', []))}")
            print(f"Source Types: {', '.join(metadata.get('source_types', []))}")
            if self.data_split != 'unknown':
                print(f"Data Split: {self.data_split.title()}")
            print()
            
            print("OUTCOMES STATISTICS")
            print("-" * 50)
            
            if not self.outcomes_df.empty:
                if 'Category' in self.outcomes_df.columns:
                    category_counts = self.outcomes_df['Category'].value_counts()
                    print("Outcomes by Category:")
                    for category, count in category_counts.items():
                        print(f"  {category}: {count}")
                    print()
                
                if 'Country' in self.outcomes_df.columns:
                    country_outcome_counts = self.outcomes_df['Country'].value_counts()
                    print("Outcome Reports by Country:")
                    for country, count in country_outcome_counts.items():
                        print(f"  {country}: {count}")
                    print()
                
                if 'Source_Type' in self.outcomes_df.columns:
                    source_outcome_counts = self.outcomes_df['Source_Type'].value_counts()
                    print("Outcome Reports by Source Type:")
                    for source, count in source_outcome_counts.items():
                        print(f"  {source}: {count}")
                    print()
                
            else:
                consolidated_outcomes = self.data.get('consolidated_outcomes', {})
                for category, subcategories in consolidated_outcomes.items():
                    if isinstance(subcategories, dict):
                        category_total = sum(len(outcomes) for outcomes in subcategories.values() 
                                           if isinstance(outcomes, list))
                        print(f"  {category}: {category_total} outcomes")
                print()
                
        except Exception as e:
            print(f"Error generating summary statistics: {e}")
    
    def get_category_country_matrix(self):
        if self.outcomes_df.empty or 'Category' not in self.outcomes_df.columns or 'Country' not in self.outcomes_df.columns:
            return pd.DataFrame()
        
        try:
            matrix = self.outcomes_df.pivot_table(
                index='Category', 
                columns='Country', 
                values='Outcome_Name',
                aggfunc='count',
                fill_value=0
            )
            return matrix
        except Exception as e:
            print(f"Error creating category-country matrix: {e}")
            return pd.DataFrame()
    
    def get_outcome_source_matrix(self):
        if self.outcomes_df.empty or 'Outcome_Name' not in self.outcomes_df.columns or 'Source_Type' not in self.outcomes_df.columns:
            return pd.DataFrame()
        
        try:
            matrix = self.outcomes_df.pivot_table(
                index='Outcome_Name', 
                columns='Source_Type', 
                values='Country',
                aggfunc='count',
                fill_value=0
            )
            return matrix
        except Exception as e:
            print(f"Error creating outcome-source matrix: {e}")
            return pd.DataFrame()


class DataVisualizer:
    def __init__(self, pico_analyzer, outcome_analyzer, output_dir="results/visualizations"):
        self.pico_analyzer = pico_analyzer
        self.outcome_analyzer = outcome_analyzer
        
        split_suffix = f"_{pico_analyzer.data_split}" if pico_analyzer.data_split != 'unknown' else ""
        case_suffix = f"_{pico_analyzer.case.lower()}" if pico_analyzer.case != 'UNKNOWN' else ""
        self.output_dir = Path(f"{output_dir}{case_suffix}{split_suffix}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        plt.rcParams.update({
            'font.family': 'DejaVu Sans',
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'figure.titlesize': 14,
            'axes.linewidth': 0.8,
            'grid.linewidth': 0.5,
            'lines.linewidth': 1.2
        })
        
        sns.set_style("whitegrid", {
            'axes.grid': True,
            'grid.color': '.8',
            'grid.linestyle': '-',
            'grid.linewidth': 0.5
        })
        
        self.scientific_colors = {
            'primary': '#2E4057',
            'secondary': '#048A81',
            'tertiary': '#F18F01',
            'quaternary': '#C73E1D',
            'light_gray': '#E8E8E8',
            'dark_gray': '#4A4A4A'
        }

    def _get_title_suffix(self):
        """Get suffix for plot titles including case and data split info."""
        case_info = f" - {self.pico_analyzer.case}" if self.pico_analyzer.case != 'UNKNOWN' else ""
        split_info = f" ({self.pico_analyzer.data_split.title()} Set)" if self.pico_analyzer.data_split != 'unknown' else ""
        return f"{case_info}{split_info}"

    def get_comparators_by_source_type(self):
        """Extract comparators grouped by source type from consolidated PICO data."""
        if 'consolidated_picos' not in self.pico_analyzer.data:
            return set(), set()
            
        guideline_comparators = set()
        hta_comparators = set()
        
        for pico in self.pico_analyzer.data['consolidated_picos']:
            comparator = pico.get('Comparator', '')
            source_types = pico.get('Source_Types', [])
            
            if 'clinical_guideline' in source_types:
                guideline_comparators.add(comparator)
            if 'hta_submission' in source_types:
                hta_comparators.add(comparator)
                
        return guideline_comparators, hta_comparators

    def generate_summary_report(self):
        print("Generating summary report...")
        
        report_content = []
        split_info = f" ({self.pico_analyzer.data_split.title()} Set)" if self.pico_analyzer.data_split != 'unknown' else ""
        case_info = f" - {self.pico_analyzer.case}" if self.pico_analyzer.case != 'UNKNOWN' else ""
        
        report_content.append(f"RAG PIPELINE ANALYSIS SUMMARY REPORT{case_info}{split_info}")
        report_content.append("=" * 50)
        report_content.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if self.pico_analyzer.data_split != 'unknown':
            report_content.append(f"Data Split: {self.pico_analyzer.data_split.title()}")
        if self.pico_analyzer.case != 'UNKNOWN':
            report_content.append(f"Case: {self.pico_analyzer.case}")
        report_content.append("")
        
        report_content.append("PICO ANALYSIS SUMMARY:")
        report_content.append("-" * 25)
        
        if not self.pico_analyzer.picos_df.empty:
            if 'consolidated_picos' in self.pico_analyzer.data:
                report_content.append(f"Total consolidated PICOs: {len(self.pico_analyzer.data['consolidated_picos'])}")
            if 'Country' in self.pico_analyzer.picos_df.columns:
                report_content.append(f"Unique countries: {self.pico_analyzer.picos_df['Country'].nunique()}")
                report_content.append(f"Most common country: {self.pico_analyzer.picos_df['Country'].mode().iloc[0] if not self.pico_analyzer.picos_df['Country'].mode().empty else 'N/A'}")
            if 'Comparator' in self.pico_analyzer.picos_df.columns:
                report_content.append(f"Unique comparators: {self.pico_analyzer.picos_df['Comparator'].nunique()}")
                report_content.append(f"Most common comparator: {self.pico_analyzer.picos_df['Comparator'].mode().iloc[0] if not self.pico_analyzer.picos_df['Comparator'].mode().empty else 'N/A'}")
        else:
            report_content.append("No PICO data available for analysis")
        report_content.append("")
        
        report_content.append("OUTCOMES ANALYSIS SUMMARY:")
        report_content.append("-" * 27)
        
        if self.outcome_analyzer.total_outcomes > 0:
            report_content.append(f"Total outcome measures: {self.outcome_analyzer.total_outcomes}")
            metadata = self.outcome_analyzer.data.get('outcomes_metadata', {})
            if metadata.get('source_countries'):
                report_content.append(f"Countries with outcomes: {len(metadata['source_countries'])}")
                report_content.append(f"Countries: {', '.join(metadata['source_countries'])}")
            if metadata.get('source_types'):
                report_content.append(f"Source types: {', '.join(metadata['source_types'])}")
        else:
            report_content.append("No outcomes data available for analysis")
        report_content.append("")
        
        pico_countries = set()
        outcome_countries = set()
        
        if not self.pico_analyzer.picos_df.empty and 'Country' in self.pico_analyzer.picos_df.columns:
            pico_countries = set(self.pico_analyzer.picos_df['Country'].unique())
            
        metadata = self.outcome_analyzer.data.get('outcomes_metadata', {})
        if metadata.get('source_countries'):
            outcome_countries = set(metadata['source_countries'])
        
        common_countries = pico_countries.intersection(outcome_countries)
        
        report_content.append("COVERAGE ANALYSIS:")
        report_content.append("-" * 18)
        report_content.append(f"Countries with both PICOs and outcomes: {len(common_countries)}")
        report_content.append(f"PICO-only countries: {len(pico_countries - outcome_countries)}")
        report_content.append(f"Outcome-only countries: {len(outcome_countries - pico_countries)}")
        report_content.append(f"Total country coverage: {len(pico_countries.union(outcome_countries))}")
        
        if self.pico_analyzer.data_split != 'unknown':
            report_content.append("")
            report_content.append("TRAIN/TEST SPLIT INFORMATION:")
            report_content.append("-" * 30)
            report_content.append(f"Current analysis covers: {self.pico_analyzer.data_split.title()} set")
            if pico_countries:
                report_content.append(f"{self.pico_analyzer.data_split.title()} countries: {', '.join(sorted(pico_countries))}")
        
        report_filename = f'analysis_summary_report.txt'
        with open(self.output_dir / report_filename, 'w') as f:
            f.write('\n'.join(report_content))
        
        print('\n'.join(report_content))
        """Create Venn diagram showing overlap of comparators between guidelines and HTA submissions."""
        if not VENN_AVAILABLE:
            print("Matplotlib-venn not available. Skipping Venn diagram.")
            return
            
        print("Creating comparator Venn diagram...")
        
        guideline_comparators, hta_comparators = self.get_comparators_by_source_type()
        
        guideline_comparators.discard('')
        hta_comparators.discard('')
        
        if not guideline_comparators and not hta_comparators:
            print("No comparator data available for Venn diagram")
            return
            
        fig, ax = plt.subplots(figsize=(10, 8))
        
        venn = venn2([guideline_comparators, hta_comparators], 
                     set_labels=('Clinical Guidelines', 'HTA Submissions'),
                     ax=ax)
        
        if venn.get_patch_by_id('10'):
            venn.get_patch_by_id('10').set_color(self.scientific_colors['secondary'])
            venn.get_patch_by_id('10').set_alpha(0.7)
        if venn.get_patch_by_id('01'):
            venn.get_patch_by_id('01').set_color(self.scientific_colors['tertiary'])
            venn.get_patch_by_id('01').set_alpha(0.7)
        if venn.get_patch_by_id('11'):
            venn.get_patch_by_id('11').set_color(self.scientific_colors['primary'])
            venn.get_patch_by_id('11').set_alpha(0.8)
        
        venn2_circles([guideline_comparators, hta_comparators], ax=ax, linewidth=2)
        
        overlap = guideline_comparators.intersection(hta_comparators)
        guideline_only = guideline_comparators - hta_comparators
        hta_only = hta_comparators - guideline_comparators
        
        title_suffix = self._get_title_suffix()
        ax.set_title(f'Comparator Overlap: Guidelines vs HTA Submissions{title_suffix}\n'
                    f'Total Unique Comparators: {len(guideline_comparators.union(hta_comparators))}',
                    fontsize=14, fontweight='bold', pad=20)
        
        stats_text = (f"Guidelines Only: {len(guideline_only)}\n"
                     f"HTA Only: {len(hta_only)}\n"
                     f"Both Sources: {len(overlap)}")
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
               fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'comparator_venn_diagram.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        print("\nComparator Overlap Analysis:")
        print(f"Guidelines-only comparators ({len(guideline_only)}):")
        for comp in sorted(guideline_only):
            print(f"  - {comp}")
        print(f"\nHTA-only comparators ({len(hta_only)}):")
        for comp in sorted(hta_only):
            print(f"  - {comp}")
        print(f"\nShared comparators ({len(overlap)}):")
        for comp in sorted(overlap):
            print(f"  - {comp}")

    def create_comparator_breadth_by_country(self):
        """Create bar chart showing number of distinct comparators by country."""
        print("Creating comparator breadth by country chart...")
        
        if 'consolidated_picos' not in self.pico_analyzer.data:
            print("No consolidated PICO data available for comparator breadth analysis")
            return
            
        country_comparators = defaultdict(set)
        country_source_info = defaultdict(set)
        
        for pico in self.pico_analyzer.data['consolidated_picos']:
            comparator = pico.get('Comparator', '')
            countries = pico.get('Countries', [])
            source_types = pico.get('Source_Types', [])
            
            if comparator and countries:
                for country in countries:
                    country_comparators[country].add(comparator)
                    country_source_info[country].update(source_types)
        
        if not country_comparators:
            print("No country-comparator data available")
            return
            
        countries = list(country_comparators.keys())
        comparator_counts = [len(country_comparators[country]) for country in countries]
        
        colors = []
        for country in countries:
            sources = country_source_info[country]
            if len(sources) > 1:
                colors.append(self.scientific_colors['primary'])
            elif 'clinical_guideline' in sources:
                colors.append(self.scientific_colors['secondary'])
            else:
                colors.append(self.scientific_colors['tertiary'])
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        bars = ax.bar(range(len(countries)), comparator_counts, 
                     color=colors, alpha=0.8, edgecolor='white', linewidth=1)
        
        ax.set_xticks(range(len(countries)))
        ax.set_xticklabels(countries, fontweight='bold')
        ax.set_ylabel('Number of Distinct Comparators', fontweight='bold')
        ax.set_xlabel('Country', fontweight='bold')
        
        title_suffix = self._get_title_suffix()
        ax.set_title(f'Comparator Breadth by Country{title_suffix}\n(Distinct Comparators Considered in PICO Evidence)', 
                    fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar, value in zip(bars, comparator_counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   str(value), ha='center', va='bottom', fontweight='bold')
        
        legend_elements = [
            plt.Rectangle((0,0),1,1, facecolor=self.scientific_colors['primary'], alpha=0.8, label='Both Sources'),
            plt.Rectangle((0,0),1,1, facecolor=self.scientific_colors['secondary'], alpha=0.8, label='Guidelines Only'),
            plt.Rectangle((0,0),1,1, facecolor=self.scientific_colors['tertiary'], alpha=0.8, label='HTA Only')
        ]
        ax.legend(handles=legend_elements, loc='upper right', frameon=True, fancybox=True, shadow=True)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'comparator_breadth_by_country.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()

    def create_country_pico_summary_table(self):
        """Create comprehensive country-by-country PICO summary table."""
        print("Creating country-by-country PICO summary table...")
        
        if 'consolidated_picos' not in self.pico_analyzer.data:
            print("No consolidated PICO data available for summary table")
            return
            
        country_data = defaultdict(lambda: {
            'population': set(),
            'intervention': set(),
            'guideline_comparators': set(),
            'hta_comparators': set(),
            'source_types': set()
        })
        
        for pico in self.pico_analyzer.data['consolidated_picos']:
            population = pico.get('Population', '')
            intervention = pico.get('Intervention', '')
            comparator = pico.get('Comparator', '')
            countries = pico.get('Countries', [])
            source_types = pico.get('Source_Types', [])
            
            for country in countries:
                country_data[country]['population'].add(population)
                country_data[country]['intervention'].add(intervention)
                country_data[country]['source_types'].update(source_types)
                
                if 'clinical_guideline' in source_types:
                    country_data[country]['guideline_comparators'].add(comparator)
                if 'hta_submission' in source_types:
                    country_data[country]['hta_comparators'].add(comparator)
        
        table_data = []
        for country in sorted(country_data.keys()):
            data = country_data[country]
            
            population = list(data['population'])[0] if data['population'] else 'Not specified'
            if len(population) > 80:
                population = population[:77] + "..."
                
            intervention = list(data['intervention'])[0] if data['intervention'] else 'Not specified'
            
            guideline_comps = ', '.join(sorted(data['guideline_comparators'])) if data['guideline_comparators'] else '-'
            hta_comps = ', '.join(sorted(data['hta_comparators'])) if data['hta_comparators'] else '-'
            
            if len(guideline_comps) > 60:
                guideline_comps = guideline_comps[:57] + "..."
            if len(hta_comps) > 60:
                hta_comps = hta_comps[:57] + "..."
                
            sources = ', '.join([s.replace('_', ' ').title() for s in sorted(data['source_types'])])
            
            table_data.append([
                country,
                population,
                intervention,
                guideline_comps,
                hta_comps,
                sources
            ])
        
        df = pd.DataFrame(table_data, columns=[
            'Country', 'Population', 'Intervention', 
            'Guideline Comparators', 'HTA Comparators', 'Source Types'
        ])
        
        csv_path = self.output_dir / 'country_pico_summary_table.csv'
        df.to_csv(csv_path, index=False)
        print(f"PICO summary table saved to: {csv_path}")
        
        fig, ax = plt.subplots(figsize=(16, max(8, len(table_data) * 0.5)))
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=table_data,
                        colLabels=['Country', 'Population Description', 'Intervention', 
                                 'Guideline Comparators', 'HTA Comparators', 'Source Types'],
                        cellLoc='left',
                        loc='center',
                        bbox=[0, 0, 1, 1])
        
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 2)
        
        for i in range(len(df.columns)):
            table[(0, i)].set_facecolor(self.scientific_colors['primary'])
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        for i in range(1, len(table_data) + 1):
            for j in range(len(df.columns)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#F8F9FA')
                else:
                    table[(i, j)].set_facecolor('white')
        
        title_suffix = self._get_title_suffix()
        plt.title(f'Country-by-Country PICO Summary{title_suffix}\nComparative Analysis of Evidence Requirements', 
                 fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'country_pico_summary_table.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        print(f"\nSummary Statistics:")
        print(f"Total countries analyzed: {len(country_data)}")
        guideline_only = sum(1 for data in country_data.values() 
                           if 'clinical_guideline' in data['source_types'] and 'hta_submission' not in data['source_types'])
        hta_only = sum(1 for data in country_data.values() 
                     if 'hta_submission' in data['source_types'] and 'clinical_guideline' not in data['source_types'])
        both_sources = sum(1 for data in country_data.values() 
                         if len(data['source_types']) > 1)
        print(f"Countries with guidelines only: {guideline_only}")
        print(f"Countries with HTA only: {hta_only}")
        print(f"Countries with both sources: {both_sources}")

    def create_outcomes_treemap(self):
        """Create treemap visualization of outcome variables."""
        if not SQUARIFY_AVAILABLE:
            print("Squarify not available. Creating alternative outcomes visualization...")
            self._create_alternative_outcomes_hierarchy()
            return
            
        print("Creating outcomes treemap...")
        
        consolidated_outcomes = self.outcome_analyzer.data.get('consolidated_outcomes', {})
        if not consolidated_outcomes:
            print("No consolidated outcomes data available for treemap")
            return
        
        sizes = []
        labels = []
        colors = []
        color_map = {
            'efficacy': self.scientific_colors['primary'],
            'safety': self.scientific_colors['quaternary'],
            'quality_of_life': self.scientific_colors['secondary'],
            'economic': self.scientific_colors['tertiary'],
            'other': self.scientific_colors['dark_gray']
        }
        
        for category, subcategories in consolidated_outcomes.items():
            if isinstance(subcategories, dict):
                for subcategory, outcomes in subcategories.items():
                    if isinstance(outcomes, list):
                        count = len(outcomes)
                        sizes.append(count)
                        labels.append(f"{category.title()}\n{subcategory.replace('_', ' ').title()}\n({count} outcomes)")
                        colors.append(color_map.get(category, self.scientific_colors['dark_gray']))
        
        if not sizes:
            print("No outcome data available for treemap")
            return
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        squarify.plot(sizes=sizes, label=labels, color=colors, alpha=0.8,
                     text_kwargs={'fontsize': 9, 'weight': 'bold'}, ax=ax)
        
        title_suffix = self._get_title_suffix()
        ax.set_title(f'Outcomes Evidence Hierarchy{title_suffix}\nTreemap by Category and Subcategory', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.axis('off')
        
        legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, alpha=0.8, label=cat.title()) 
                          for cat, color in color_map.items() if cat in [c for c, _ in consolidated_outcomes.items()]]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1), 
                 frameon=True, fancybox=True, shadow=True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'outcomes_treemap.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()

    def _create_alternative_outcomes_hierarchy(self):
        """Create alternative hierarchical bar chart when squarify is not available."""
        print("Creating alternative outcomes hierarchy visualization...")
        
        consolidated_outcomes = self.outcome_analyzer.data.get('consolidated_outcomes', {})
        if not consolidated_outcomes:
            print("No consolidated outcomes data available")
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        category_data = []
        for category, subcategories in consolidated_outcomes.items():
            if isinstance(subcategories, dict):
                total_count = sum(len(outcomes) for outcomes in subcategories.values() 
                                if isinstance(outcomes, list))
                category_data.append((category, total_count))
        
        categories = [item[0].replace('_', ' ').title() for item in category_data]
        counts = [item[1] for item in category_data]
        
        colors = [self.scientific_colors['primary'], self.scientific_colors['quaternary'],
                 self.scientific_colors['secondary'], self.scientific_colors['tertiary'],
                 self.scientific_colors['dark_gray']][:len(categories)]
        
        bars = ax.bar(range(len(categories)), counts, color=colors, alpha=0.8,
                     edgecolor='white', linewidth=1)
        
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(categories, rotation=45, ha='right', fontweight='bold')
        ax.set_ylabel('Number of Outcomes', fontweight='bold')
        ax.set_xlabel('Outcome Category', fontweight='bold')
        
        title_suffix = self._get_title_suffix()
        ax.set_title(f'Outcomes Distribution by Category{title_suffix}\n(Alternative Hierarchy View)', 
                    fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar, value in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts) * 0.01,
                   str(value), ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'outcomes_hierarchy_alternative.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()

    def create_pico_consolidation_sankey(self):
        """Create Sankey diagram showing PICO consolidation process."""
        if not PLOTLY_AVAILABLE:
            print("Plotly not available. Creating alternative consolidation visualization...")
            self._create_alternative_consolidation_viz()
            return
            
        print("Creating PICO consolidation Sankey diagram...")
        
        pico_file_path = Path(self.pico_analyzer.pico_file_path)
        case_root_dir = pico_file_path.parent.parent
        pico_dir = case_root_dir / "PICO"
        
        individual_picos = {}
        for source_type in ['clinical_guideline', 'hta_submission']:
            pico_file = pico_dir / f"{source_type}_picos.json"
            if pico_file.exists():
                try:
                    with open(pico_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        individual_picos[source_type] = data
                except Exception as e:
                    print(f"Error loading {source_type} PICOs: {e}")
        
        individual_counts = {}
        total_individual = 0
        for source_type, data in individual_picos.items():
            count = 0
            if 'picos_by_country' in data:
                for country_data in data['picos_by_country'].values():
                    if 'PICOs' in country_data:
                        count += len(country_data['PICOs'])
            individual_counts[source_type] = count
            total_individual += count
        
        consolidated_count = len(self.pico_analyzer.data.get('consolidated_picos', []))
        
        print(f"Individual PICO counts: {individual_counts}")
        print(f"Total individual PICOs: {total_individual}")
        print(f"Consolidated PICOs: {consolidated_count}")
        
        if total_individual == 0:
            print("No individual PICO data found. Cannot create Sankey diagram.")
            return
        
        source_labels = []
        target_labels = []
        values = []
        
        for source_type, count in individual_counts.items():
            if count > 0:
                source_labels.append(f"{source_type.replace('_', ' ').title()}")
                target_labels.append("Consolidated PICOs")
                values.append(count)
        
        if not values:
            print("No valid source data found for Sankey diagram")
            return
        
        all_labels = list(set(source_labels + target_labels))
        
        source_indices = [all_labels.index(label) for label in source_labels]
        target_indices = [all_labels.index(label) for label in target_labels]
        
        colors = [
            'rgba(4, 138, 129, 0.8)',
            'rgba(241, 143, 1, 0.8)',
            'rgba(46, 64, 87, 0.8)'
        ]
        
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=all_labels,
                color=colors[:len(all_labels)]
            ),
            link=dict(
                source=source_indices,
                target=target_indices,
                value=values,
                color=['rgba(4, 138, 129, 0.4)' if 'Clinical' in source_labels[i] 
                      else 'rgba(241, 143, 1, 0.4)' for i in range(len(values))]
            )
        )])
        
        title_suffix = self._get_title_suffix()
        fig.update_layout(
            title_text=f"PICO Consolidation Process Flow{title_suffix}<br>"
                      f"<sub>From {total_individual} Individual PICOs to {consolidated_count} Consolidated PICOs</sub>",
            font_size=12,
            font_family="Arial",
            width=800,
            height=500
        )
        
        html_path = self.output_dir / 'pico_consolidation_sankey.html'
        fig.write_html(str(html_path))
        
        try:
            png_path = self.output_dir / 'pico_consolidation_sankey.png'
            fig.write_image(str(png_path), width=800, height=500, scale=2)
            print(f"Sankey diagram saved to: {png_path}")
        except Exception as e:
            print(f"Could not save PNG (install kaleido for static export): {e}")
        
        fig.show()
        print(f"Interactive Sankey diagram saved to: {html_path}")
        
        case_name = self.pico_analyzer.case
        print(f"\n{case_name} Consolidation Statistics:")
        print(f"Total individual PICOs: {total_individual}")
        for source_type, count in individual_counts.items():
            if count > 0:
                print(f"  - {source_type.replace('_', ' ').title()}: {count}")
        print(f"Consolidated PICOs: {consolidated_count}")
        reduction_pct = ((total_individual - consolidated_count) / total_individual * 100) if total_individual > 0 else 0
        print(f"Reduction: {reduction_pct:.1f}% ({total_individual - consolidated_count} PICOs consolidated)")

    def _create_alternative_consolidation_viz(self):
        """Create alternative consolidation visualization when Plotly is not available."""
        print("Creating alternative consolidation visualization...")
        
        pico_file_path = Path(self.pico_analyzer.pico_file_path)
        case_root_dir = pico_file_path.parent.parent
        pico_dir = case_root_dir / "PICO"
        
        individual_counts = {}
        total_individual = 0
        for source_type in ['clinical_guideline', 'hta_submission']:
            pico_file = pico_dir / f"{source_type}_picos.json"
            
            if pico_file.exists():
                try:
                    with open(pico_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        count = 0
                        
                        metadata = data.get('extraction_metadata', {})
                        
                        if 'total_picos' in metadata:
                            count = metadata['total_picos']
                        elif 'picos_by_country' in data:
                            for country, country_data in data['picos_by_country'].items():
                                if 'PICOs' in country_data:
                                    country_count = len(country_data['PICOs'])
                                    count += country_count
                        
                        individual_counts[source_type] = count
                        total_individual += count
                        
                except Exception as e:
                    print(f"Failed to load {source_type} PICOs: {e}")
            else:
                print(f"File not found: {pico_file}")
        
        consolidated_count = len(self.pico_analyzer.data.get('consolidated_picos', []))
        
        if total_individual == 0:
            print("No individual PICO data found for alternative visualization")
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        y_positions = [0.7, 0.3]
        x_individual = 0.2
        x_consolidated = 0.8
        
        colors = [self.scientific_colors['secondary'], self.scientific_colors['tertiary']]
        for i, (source_type, count) in enumerate(individual_counts.items()):
            if count == 0:
                continue
                
            box_height = 0.15
            box_width = 0.2
            y_pos = y_positions[i] if i < len(y_positions) else 0.5
            
            rect = plt.Rectangle((x_individual - box_width/2, y_pos - box_height/2), 
                               box_width, box_height,
                               facecolor=colors[i] if i < len(colors) else self.scientific_colors['dark_gray'],
                               alpha=0.8, edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            
            ax.text(x_individual, y_pos, f"{source_type.replace('_', ' ').title()}\n{count} PICOs",
                   ha='center', va='center', fontweight='bold', fontsize=10)
            
            ax.annotate('', xy=(x_consolidated - 0.12, 0.5), xytext=(x_individual + 0.1, y_pos),
                       arrowprops=dict(arrowstyle='->', lw=3, 
                                     color=colors[i] if i < len(colors) else self.scientific_colors['dark_gray'],
                                     alpha=0.7))
        
        box_height = 0.2
        box_width = 0.25
        rect = plt.Rectangle((x_consolidated - box_width/2, 0.5 - box_height/2), 
                           box_width, box_height,
                           facecolor=self.scientific_colors['primary'],
                           alpha=0.8, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        
        ax.text(x_consolidated, 0.5, f"Consolidated\nPICOs\n{consolidated_count}",
               ha='center', va='center', fontweight='bold', fontsize=12, color='white')
        
        reduction_pct = ((total_individual - consolidated_count) / total_individual * 100) if total_individual > 0 else 0
        stats_text = (f"Total Individual: {total_individual}\n"
                     f"Consolidated: {consolidated_count}\n"
                     f"Reduction: {reduction_pct:.1f}%")
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
               fontsize=10, fontweight='bold')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        title_suffix = self._get_title_suffix()
        ax.set_title(f'PICO Consolidation Process{title_suffix}\nFrom Individual Sources to Consolidated Evidence', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'pico_consolidation_flow.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        case_name = self.pico_analyzer.case
        print(f"\n{case_name} Consolidation Statistics:")
        print(f"Total individual PICOs: {total_individual}")
        for source_type, count in individual_counts.items():
            if count > 0:
                print(f"  - {source_type.replace('_', ' ').title()}: {count}")
        print(f"Consolidated PICOs: {consolidated_count}")
        reduction_pct = ((total_individual - consolidated_count) / total_individual * 100) if total_individual > 0 else 0
        print(f"Reduction: {reduction_pct:.1f}% ({total_individual - consolidated_count} PICOs consolidated)")

    def create_european_heatmap(self):
        if self.pico_analyzer.picos_df.empty or 'Country' not in self.pico_analyzer.picos_df.columns:
            print("No country data available for European heatmap")
            return
            
        print("Creating European PICO distribution map...")
        
        country_counts = self.pico_analyzer.picos_df['Country'].value_counts()
        print(f"Total countries with PICOs: {len(country_counts)}")
        print("Countries and their PICO counts:")
        for country, count in country_counts.items():
            print(f"  {country}: {count} PICOs")
        print()
        
        if not GEOPANDAS_AVAILABLE:
            print("Geopandas not available. Creating simplified map...")
            self._create_simplified_european_map()
            return
            
        try:
            print("Loading geographic data...")
            
            world_url = "https://raw.githubusercontent.com/holtzy/The-Python-Graph-Gallery/master/static/data/world-110m2.json"
            
            try:
                world = gpd.read_file(world_url)
                print("Successfully loaded world data from web")
            except:
                print("Could not load from web, trying alternative dataset...")
                world_url_alt = "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
                try:
                    world = gpd.read_file(world_url_alt)
                    print("Successfully loaded alternative dataset")
                except:
                    print("Could not load geographic data, falling back to simplified map")
                    self._create_simplified_european_map()
                    return
            
            country_mapping = {
                'DE': 'Germany',
                'DK': 'Denmark', 
                'EN': 'United Kingdom',
                'FR': 'France',
                'NL': 'Netherlands',
                'PO': 'Poland',
                'PT': 'Portugal',
                'SE': 'Sweden',
                'AT': 'Austria',       
                'IT': 'Italy',        
                'ES': 'Spain',         
                'BE': 'Belgium',
                'CH': 'Switzerland',
                'NO': 'Norway',
                'FI': 'Finland',
                'IE': 'Ireland',
                'CZ': 'Czech Republic',
                'HU': 'Hungary',
                'RO': 'Romania',
                'BG': 'Bulgaria',
                'GR': 'Greece',
                'HR': 'Croatia',
                'SI': 'Slovenia',
                'SK': 'Slovakia',
                'LT': 'Lithuania',
                'LV': 'Latvia',
                'EE': 'Estonia'         
            }
            
            name_column = 'NAME' if 'NAME' in world.columns else 'name'
            if name_column not in world.columns:
                for col in ['NAME_EN', 'NAME_LONG', 'ADMIN', 'Country']:
                    if col in world.columns:
                        name_column = col
                        break
            
            print(f"Countries with PICO data: {dict(country_counts)}")
            
            pico_data = []
            for country_code, count in country_counts.items():
                if country_code in country_mapping:
                    country_name = country_mapping[country_code]
                    pico_data.append({name_column: country_name, 'pico_count': count, 'code': country_code})
            
            pico_df = pd.DataFrame(pico_data)
            
            european_countries = [
                'Germany', 'Denmark', 'United Kingdom', 'France', 'Netherlands', 
                'Poland', 'Portugal', 'Sweden', 'Norway', 'Finland', 'Spain', 
                'Italy', 'Switzerland', 'Austria', 'Belgium', 'Ireland',
                'Czech Republic', 'Hungary', 'Romania', 'Bulgaria', 'Greece',
                'Croatia', 'Slovenia', 'Slovakia', 'Lithuania', 'Latvia', 'Estonia'
            ]
            
            europe = world[world[name_column].isin(european_countries)].copy()
            
            europe = europe.merge(pico_df, on=name_column, how='left')
            europe['pico_count'] = europe['pico_count'].fillna(0)
            
            fig, ax = plt.subplots(1, 1, figsize=(14, 10))
            
            ax.set_xlim(-12, 32)
            ax.set_ylim(35, 72)
            
            europe_no_data = europe[europe['pico_count'] == 0]
            if not europe_no_data.empty:
                europe_no_data.plot(ax=ax, color='#F0F0F0', edgecolor='white', linewidth=0.8)
            
            europe_with_data = europe[europe['pico_count'] > 0]
            if not europe_with_data.empty:
                vmin = 1
                vmax = europe_with_data['pico_count'].max()
                
                europe_with_data.plot(
                    column='pico_count', 
                    ax=ax, 
                    cmap='YlOrRd',
                    legend=False,
                    edgecolor='white', 
                    linewidth=0.8,
                    vmin=vmin,
                    vmax=vmax
                )
                
                for idx, row in europe_with_data.iterrows():
                    try:
                        centroid = row.geometry.centroid
                        
                        if row[name_column] == 'France':
                            x_pos, y_pos = 2.2, 46.2
                        else:
                            x_pos, y_pos = centroid.x, centroid.y
                        
                        ax.annotate(
                            f"{row['code']}\n({int(row['pico_count'])})",
                            xy=(x_pos, y_pos),
                            ha='center', va='center',
                            fontsize=10, fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                    alpha=0.8, edgecolor='none'),
                            zorder=5
                        )
                    except Exception as e:
                        print(f"Warning: Could not annotate {row.get(name_column, 'Unknown')}: {e}")
                
                sm = plt.cm.ScalarMappable(cmap='YlOrRd', 
                                         norm=plt.Normalize(vmin=vmin, vmax=vmax))
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=ax, shrink=0.6, aspect=30)
                cbar.set_label('Number of PICOs', fontweight='bold', fontsize=12)
                cbar.ax.tick_params(labelsize=10)
            
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            
            title_suffix = self._get_title_suffix()
            ax.set_title(f'European Distribution of PICO Evidence{title_suffix}\nby Country ({len(country_counts)} countries)', 
                        fontsize=16, fontweight='bold', pad=20)
            
            no_data_patch = mpatches.Patch(color='#F0F0F0', label='No PICO data')
            ax.legend(handles=[no_data_patch], loc='upper left', 
                     bbox_to_anchor=(0.02, 0.98), frameon=True, fancybox=True, shadow=True)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'european_pico_heatmap.png', 
                       dpi=300, bbox_inches='tight', facecolor='white')
            print("European choropleth map saved successfully")
            plt.show()
            
        except Exception as e:
            print(f"Error creating European map with geopandas: {str(e)}")
            print("Falling back to simplified map...")
            self._create_simplified_european_map()
    
    def _create_simplified_european_map(self):
        if self.pico_analyzer.picos_df.empty or 'Country' not in self.pico_analyzer.picos_df.columns:
            print("No country data available for simplified map")
            return
            
        print("Creating simplified European map...")
        
        country_counts = self.pico_analyzer.picos_df['Country'].value_counts()
        
        country_mapping = {
            'DE': 'Germany',
            'DK': 'Denmark', 
            'EN': 'United Kingdom',
            'FR': 'France',
            'NL': 'Netherlands',
            'PO': 'Poland',
            'PT': 'Portugal',
            'SE': 'Sweden',
            'AT': 'Austria',       
            'IT': 'Italy',        
            'ES': 'Spain',         
            'BE': 'Belgium',
            'CH': 'Switzerland',
            'NO': 'Norway',
            'FI': 'Finland',
            'IE': 'Ireland',
            'CZ': 'Czech Republic',
            'HU': 'Hungary',
            'RO': 'Romania',
            'BG': 'Bulgaria',
            'GR': 'Greece',
            'HR': 'Croatia',
            'SI': 'Slovenia',
            'SK': 'Slovakia',
            'LT': 'Lithuania',
            'LV': 'Latvia',
            'EE': 'Estonia'    
        }
        
        european_positions = {
            'Germany': (10.5, 51.5),
            'Denmark': (10.0, 56.0),
            'United Kingdom': (-2.0, 54.0),
            'France': (2.2, 46.2),
            'Netherlands': (5.5, 52.0),
            'Poland': (19.0, 52.0),
            'Portugal': (-8.0, 39.5),
            'Sweden': (15.0, 62.0),
            'Austria': (13.5, 47.5),   
            'Italy': (12.5, 42.0),     
            'Spain': (-4.0, 40.0),    
            'Belgium': (4.5, 50.5),
            'Switzerland': (8.2, 46.8),
            'Norway': (8.0, 60.0),
            'Finland': (25.0, 64.0),
            'Ireland': (-8.0, 53.0),
            'Czech Republic': (15.5, 49.8),
            'Hungary': (19.5, 47.0),
            'Romania': (24.0, 45.9),
            'Bulgaria': (25.0, 42.7),
            'Greece': (22.0, 39.0),
            'Croatia': (15.5, 45.1),
            'Slovenia': (14.5, 46.1),
            'Slovakia': (19.5, 48.7),
            'Lithuania': (23.9, 55.3),
            'Latvia': (24.0, 56.9),
            'Estonia': (25.0, 58.6)     
        }
        
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_xlim(-12, 25)
        ax.set_ylim(35, 70)
        
        ax.add_patch(plt.Rectangle((-10, 35), 35, 35, 
                                 facecolor='#F8F9FA', edgecolor='#E9ECEF', 
                                 alpha=0.5, linewidth=1))
        
        if len(country_counts) == 0:
            print("Warning: No country data found for heatmap")
            return
            
        max_count = country_counts.max()
        min_count = country_counts.min()
        
        for country_code, count in country_counts.items():
            if country_code in country_mapping:
                country_name = country_mapping[country_code]
                if country_name in european_positions:
                    lon, lat = european_positions[country_name]
                    
                    if max_count > min_count:
                        normalized_size = (count - min_count) / (max_count - min_count)
                    else:
                        normalized_size = 1.0
                        
                    circle_size = 300 + normalized_size * 700
                    color_intensity = 0.3 + 0.7 * normalized_size
                    
                    ax.scatter(lon, lat, s=circle_size, 
                             color=plt.cm.YlOrRd(color_intensity), 
                             alpha=0.8, edgecolors='white', linewidth=2,
                             zorder=5)
                    
                    ax.annotate(f'{country_code}\n({count})', 
                               (lon, lat), xytext=(0, 0), 
                               textcoords='offset points',
                               ha='center', va='center',
                               fontsize=10, fontweight='bold',
                               color='white' if normalized_size > 0.5 else 'black',
                               zorder=6)
        
        ax.set_xlabel('Longitude', fontweight='bold')
        ax.set_ylabel('Latitude', fontweight='bold')
        
        title_suffix = self._get_title_suffix()
        ax.set_title(f'European Distribution of PICO Evidence{title_suffix}\n({len(country_counts)} countries - Simplified View)', 
                    fontsize=14, fontweight='bold', pad=20)
        
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'european_pico_heatmap.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        print("Simplified European map saved successfully")
        plt.show()
    
    def create_pico_visualizations(self):
        if self.pico_analyzer.picos_df.empty:
            print("No PICO data available for visualization")
            return
            
        print("Creating PICO visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        title_suffix = self._get_title_suffix()
        fig.suptitle(f'PICO Analysis Overview{title_suffix}', fontsize=16, fontweight='bold', y=0.95)
        
        if 'Country' in self.pico_analyzer.picos_df.columns:
            country_counts = self.pico_analyzer.picos_df['Country'].value_counts()
            bars1 = axes[0, 0].bar(range(len(country_counts)), country_counts.values, 
                                  color=self.scientific_colors['primary'], alpha=0.8,
                                  edgecolor='white', linewidth=0.8)
            axes[0, 0].set_xticks(range(len(country_counts)))
            axes[0, 0].set_xticklabels(country_counts.index, fontweight='bold')
            axes[0, 0].set_ylabel('Number of PICOs', fontweight='bold')
            axes[0, 0].set_xlabel('Country', fontweight='bold')
            axes[0, 0].set_title('A. PICO Distribution by Country', fontweight='bold', pad=15)
            axes[0, 0].grid(True, alpha=0.3, axis='y')
            
            for bar, value in zip(bars1, country_counts.values):
                axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                               str(value), ha='center', va='bottom', fontweight='bold')
        else:
            axes[0, 0].text(0.5, 0.5, 'No Country Data Available', 
                           ha='center', va='center', transform=axes[0, 0].transAxes)
            axes[0, 0].set_title('A. PICO Distribution by Country', fontweight='bold', pad=15)
        
        if 'Source_Type' in self.pico_analyzer.picos_df.columns:
            source_counts = self.pico_analyzer.picos_df['Source_Type'].value_counts()
            colors = [self.scientific_colors['secondary'], self.scientific_colors['tertiary']][:len(source_counts)]
            
            bars2 = axes[0, 1].bar(range(len(source_counts)), source_counts.values,
                                  color=colors, alpha=0.8, edgecolor='white', linewidth=0.8)
            axes[0, 1].set_xticks(range(len(source_counts)))
            axes[0, 1].set_xticklabels([label.replace('_', ' ').title() for label in source_counts.index], 
                                      rotation=45, ha='right', fontweight='bold')
            axes[0, 1].set_ylabel('Number of PICOs', fontweight='bold')
            axes[0, 1].set_xlabel('Source Type', fontweight='bold')
            axes[0, 1].set_title('B. PICO Distribution by Source Type', fontweight='bold', pad=15)
            axes[0, 1].grid(True, alpha=0.3, axis='y')
            
            for bar, value in zip(bars2, source_counts.values):
                axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                               str(value), ha='center', va='bottom', fontweight='bold')
        else:
            axes[0, 1].text(0.5, 0.5, 'No Source Type Data Available', 
                           ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('B. PICO Distribution by Source Type', fontweight='bold', pad=15)
        
        if 'Comparator' in self.pico_analyzer.picos_df.columns:
            comp_counts = self.pico_analyzer.picos_df['Comparator'].value_counts().head(8)
            bars3 = axes[1, 0].barh(range(len(comp_counts)), comp_counts.values, 
                                   color=self.scientific_colors['quaternary'], alpha=0.8,
                                   edgecolor='white', linewidth=0.8)
            axes[1, 0].set_yticks(range(len(comp_counts)))
            axes[1, 0].set_yticklabels([comp[:25] + '...' if len(comp) > 25 else comp 
                                       for comp in comp_counts.index], fontweight='bold')
            axes[1, 0].set_xlabel('Number of PICOs', fontweight='bold')
            axes[1, 0].set_title('C. Most Frequent Comparators', fontweight='bold', pad=15)
            axes[1, 0].grid(True, alpha=0.3, axis='x')
            axes[1, 0].invert_yaxis()
            
            for bar, value in zip(bars3, comp_counts.values):
                axes[1, 0].text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                               str(value), ha='left', va='center', fontweight='bold')
        else:
            axes[1, 0].text(0.5, 0.5, 'No Comparator Data Available', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('C. Most Frequent Comparators', fontweight='bold', pad=15)
        
        if 'Country' in self.pico_analyzer.picos_df.columns and 'Source_Type' in self.pico_analyzer.picos_df.columns:
            country_source_matrix = self.pico_analyzer.picos_df.pivot_table(
                index='Country', columns='Source_Type', values='Intervention', 
                aggfunc='count', fill_value=0
            )
            
            if not country_source_matrix.empty:
                im = axes[1, 1].imshow(country_source_matrix.values, cmap='Blues', aspect='auto',
                                       vmin=0, vmax=country_source_matrix.values.max())
                
                axes[1, 1].set_xticks(range(len(country_source_matrix.columns)))
                axes[1, 1].set_xticklabels([col.replace('_', '\n') for col in country_source_matrix.columns], 
                                          fontweight='bold')
                axes[1, 1].set_yticks(range(len(country_source_matrix.index)))
                axes[1, 1].set_yticklabels(country_source_matrix.index, fontweight='bold')
                axes[1, 1].set_title('D. Country vs Source Type Matrix', fontweight='bold', pad=15)
                
                for i in range(len(country_source_matrix.index)):
                    for j in range(len(country_source_matrix.columns)):
                        value = country_source_matrix.iloc[i, j]
                        axes[1, 1].text(j, i, str(value), ha='center', va='center',
                                       fontweight='bold', 
                                       color='white' if value > country_source_matrix.values.max()/2 else 'black')
                
                cbar = plt.colorbar(im, ax=axes[1, 1], shrink=0.8)
                cbar.set_label('Number of PICOs', fontweight='bold')
            else:
                axes[1, 1].text(0.5, 0.5, 'No Matrix Data Available', 
                               ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('D. Country vs Source Type Matrix', fontweight='bold', pad=15)
        else:
            axes[1, 1].text(0.5, 0.5, 'No Matrix Data Available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('D. Country vs Source Type Matrix', fontweight='bold', pad=15)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'pico_analysis_dashboard.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        print("Creating additional PICO visualizations...")
        self._create_comparator_heatmap()
        self.create_european_heatmap()
        
        self.create_comparator_venn_diagram()
        self.create_comparator_breadth_by_country()
        self.create_country_pico_summary_table()
        self.create_pico_consolidation_sankey()
    
    def _create_comparator_heatmap(self):
        matrix = self.pico_analyzer.get_country_comparator_matrix()
        
        if matrix.empty:
            print("No data available for comparator heatmap")
            return
        
        plt.figure(figsize=(12, 8))
        
        ax = sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues',
                        cbar_kws={'label': 'Number of PICOs'},
                        linewidths=0.5, linecolor='white',
                        square=False)
        
        title_suffix = self._get_title_suffix()
        plt.title(f'Country vs Comparator Distribution Matrix{title_suffix}', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Comparator', fontweight='bold')
        plt.ylabel('Country', fontweight='bold')
        
        plt.xticks(rotation=45, ha='right', fontweight='bold')
        plt.yticks(rotation=0, fontweight='bold')
        
        cbar = ax.collections[0].colorbar
        cbar.set_label('Number of PICOs', fontweight='bold')
        
        for text in ax.texts:
            text.set_fontweight('bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'country_comparator_heatmap.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
    
    def create_outcome_visualizations(self):
        if self.outcome_analyzer.total_outcomes == 0:
            print("No outcomes data available for visualization")
            return
            
        print("Creating Outcome visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        title_suffix = self._get_title_suffix()
        fig.suptitle(f'Outcomes Analysis Overview{title_suffix}', fontsize=16, fontweight='bold', y=0.95)
        
        consolidated_outcomes = self.outcome_analyzer.data.get('consolidated_outcomes', {})
        metadata = self.outcome_analyzer.data.get('outcomes_metadata', {})
        
        category_counts = {}
        for category, subcategories in consolidated_outcomes.items():
            if isinstance(subcategories, dict):
                total_outcomes = sum(len(outcomes) for outcomes in subcategories.values() 
                                   if isinstance(outcomes, list))
                category_counts[category] = total_outcomes
        
        if category_counts:
            bars1 = axes[0, 0].bar(range(len(category_counts)), list(category_counts.values()), 
                                  color=self.scientific_colors['primary'], alpha=0.8,
                                  edgecolor='white', linewidth=0.8)
            axes[0, 0].set_xticks(range(len(category_counts)))
            axes[0, 0].set_xticklabels([cat.replace('_', ' ').title() for cat in category_counts.keys()], 
                                      rotation=45, ha='right', fontweight='bold')
            axes[0, 0].set_ylabel('Number of Outcome Measures', fontweight='bold')
            axes[0, 0].set_xlabel('Outcome Category', fontweight='bold')
            axes[0, 0].set_title('A. Outcomes by Category', fontweight='bold', pad=15)
            axes[0, 0].grid(True, alpha=0.3, axis='y')
            
            for bar, value in zip(bars1, category_counts.values()):
                axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                               str(value), ha='center', va='bottom', fontweight='bold')
        else:
            axes[0, 0].text(0.5, 0.5, 'No Category Data Available', 
                           ha='center', va='center', transform=axes[0, 0].transAxes)
            axes[0, 0].set_title('A. Outcomes by Category', fontweight='bold', pad=15)
        
        source_countries = metadata.get('source_countries', [])
        if source_countries:
            country_outcome_counts = {country: self.outcome_analyzer.total_outcomes for country in source_countries}
            bars2 = axes[0, 1].bar(range(len(country_outcome_counts)), list(country_outcome_counts.values()),
                                  color=self.scientific_colors['secondary'], alpha=0.8,
                                  edgecolor='white', linewidth=0.8)
            axes[0, 1].set_xticks(range(len(country_outcome_counts)))
            axes[0, 1].set_xticklabels(list(country_outcome_counts.keys()), 
                                      rotation=45, ha='right', fontweight='bold')
            axes[0, 1].set_ylabel('Coverage Indicator', fontweight='bold')
            axes[0, 1].set_xlabel('Country', fontweight='bold')
            axes[0, 1].set_title('B. Country Coverage for Outcomes', fontweight='bold', pad=15)
            axes[0, 1].grid(True, alpha=0.3, axis='y')
        else:
            axes[0, 1].text(0.5, 0.5, 'No Country Data Available', 
                           ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('B. Country Coverage for Outcomes', fontweight='bold', pad=15)

        if not self.pico_analyzer.picos_df.empty and 'Source_Type' in self.pico_analyzer.picos_df.columns:
            pico_source_counts = self.pico_analyzer.picos_df['Source_Type'].value_counts()
            source_outcome_counts = {source: self.outcome_analyzer.total_outcomes for source in pico_source_counts.index}
            
            if source_outcome_counts:
                colors = [self.scientific_colors['tertiary'], self.scientific_colors['quaternary']][:len(source_outcome_counts)]
                
                bars3 = axes[1, 0].bar(range(len(source_outcome_counts)), list(source_outcome_counts.values()),
                                      color=colors, alpha=0.8, edgecolor='white', linewidth=0.8)
                axes[1, 0].set_xticks(range(len(source_outcome_counts)))
                axes[1, 0].set_xticklabels([label.replace('_', ' ').title() for label in source_outcome_counts.keys()],
                                          rotation=45, ha='right', fontweight='bold')
                axes[1, 0].set_ylabel('Coverage Indicator', fontweight='bold')
                axes[1, 0].set_xlabel('Source Type', fontweight='bold')
                axes[1, 0].set_title('C. Source Type Coverage for Outcomes', fontweight='bold', pad=15)
                axes[1, 0].grid(True, alpha=0.3, axis='y')
                
                for bar, value in zip(bars3, source_outcome_counts.values()):
                    axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(source_outcome_counts.values()) * 0.01,
                                   str(value), ha='center', va='bottom', fontweight='bold')
            else:
                axes[1, 0].text(0.5, 0.5, 'No Source Type Data Available', 
                               ha='center', va='center', transform=axes[1, 0].transAxes)
                axes[1, 0].set_title('C. Source Type Coverage for Outcomes', fontweight='bold', pad=15)
        else:
            axes[1, 0].text(0.5, 0.5, 'No Source Type Data Available', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('C. Source Type Coverage for Outcomes', fontweight='bold', pad=15)
        
        subcategory_counts = {}
        for category, subcategories in consolidated_outcomes.items():
            if isinstance(subcategories, dict):
                for subcategory, outcomes in subcategories.items():
                    if isinstance(outcomes, list):
                        subcategory_counts[f"{category}_{subcategory}"] = len(outcomes)
        
        if subcategory_counts:
            top_subcategories = sorted(subcategory_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            bars4 = axes[1, 1].barh(range(len(top_subcategories)), [count for _, count in top_subcategories], 
                                   color=self.scientific_colors['quaternary'], alpha=0.8,
                                   edgecolor='white', linewidth=0.8)
            axes[1, 1].set_yticks(range(len(top_subcategories)))
            axes[1, 1].set_yticklabels([name.replace('_', ' ').title() for name, _ in top_subcategories], 
                                      fontweight='bold')
            axes[1, 1].set_xlabel('Number of Outcomes', fontweight='bold')
            axes[1, 1].set_title('D. Top Outcome Subcategories', fontweight='bold', pad=15)
            axes[1, 1].grid(True, alpha=0.3, axis='x')
            axes[1, 1].invert_yaxis()
            
            for bar, (_, value) in zip(bars4, top_subcategories):
                axes[1, 1].text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                               str(value), ha='left', va='center', fontweight='bold')
        else:
            axes[1, 1].text(0.5, 0.5, 'No Subcategory Data Available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('D. Top Outcome Subcategories', fontweight='bold', pad=15)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'outcomes_analysis_dashboard.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        self.create_outcomes_treemap()
    
    def create_combined_analysis(self):
        print("Creating combined analysis visualization...")
        
        pico_has_source = not self.pico_analyzer.picos_df.empty and 'Source_Type' in self.pico_analyzer.picos_df.columns
        outcome_has_source = self.outcome_analyzer.total_outcomes > 0
        
        if not pico_has_source and not outcome_has_source:
            print("No source type data available for combined analysis")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        title_suffix = self._get_title_suffix()
        fig.suptitle(f'Source Type Distribution Comparison{title_suffix}', fontsize=16, fontweight='bold', y=0.98)
        
        if pico_has_source:
            pico_sources = self.pico_analyzer.picos_df['Source_Type'].value_counts()
            colors_pico = [self.scientific_colors['primary'], self.scientific_colors['secondary']][:len(pico_sources)]
            
            bars1 = axes[0].bar(range(len(pico_sources)), pico_sources.values, 
                               color=colors_pico, alpha=0.8, edgecolor='white', linewidth=0.8)
            axes[0].set_xticks(range(len(pico_sources)))
            axes[0].set_xticklabels([label.replace('_', ' ').title() for label in pico_sources.index],
                                   rotation=45, ha='right', fontweight='bold')
            axes[0].set_ylabel('Number of PICOs', fontweight='bold')
            axes[0].set_xlabel('Source Type', fontweight='bold')
            axes[0].set_title('A. PICOs by Source Type', fontweight='bold', pad=20)
            axes[0].grid(True, alpha=0.3, axis='y')
            
            for bar, value in zip(bars1, pico_sources.values):
                axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(pico_sources.values) * 0.01,
                           str(value), ha='center', va='bottom', fontweight='bold')
        else:
            axes[0].text(0.5, 0.5, 'No PICO Source Type Data', 
                        ha='center', va='center', transform=axes[0].transAxes)
            axes[0].set_title('A. PICOs by Source Type', fontweight='bold', pad=20)
        
        if outcome_has_source:
            metadata = self.outcome_analyzer.data.get('outcomes_metadata', {})
            source_types = metadata.get('source_types', [])
            
            if source_types:
                colors_outcome = [self.scientific_colors['tertiary'], self.scientific_colors['quaternary']][:len(source_types)]
                outcome_values = [self.outcome_analyzer.total_outcomes] * len(source_types)
                
                bars2 = axes[1].bar(range(len(source_types)), outcome_values,
                                   color=colors_outcome, alpha=0.8, edgecolor='white', linewidth=0.8)
                axes[1].set_xticks(range(len(source_types)))
                axes[1].set_xticklabels([label.replace('_', ' ').title() for label in source_types],
                                       rotation=45, ha='right', fontweight='bold')
                axes[1].set_ylabel('Coverage Indicator', fontweight='bold')
                axes[1].set_xlabel('Source Type', fontweight='bold')
                axes[1].set_title('B. Outcomes by Source Type', fontweight='bold', pad=20)
                axes[1].grid(True, alpha=0.3, axis='y')
                
                for bar, value in zip(bars2, outcome_values):
                    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(outcome_values) * 0.01,
                               str(value), ha='center', va='bottom', fontweight='bold')
            else:
                axes[1].text(0.5, 0.5, 'No Outcome Source Type Data', 
                            ha='center', va='center', transform=axes[1].transAxes)
                axes[1].set_title('B. Outcomes by Source Type', fontweight='bold', pad=20)
        else:
            axes[1].text(0.5, 0.5, 'No Outcome Source Type Data', 
                        ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title('B. Outcomes by Source Type', fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'source_type_comparison.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
    def generate_summary_report(self):
        print("Generating summary report...")
        
        report_content = []
        split_info = f" ({self.pico_analyzer.data_split.title()} Set)" if self.pico_analyzer.data_split != 'unknown' else ""
        case_info = f" - {self.pico_analyzer.case}" if self.pico_analyzer.case != 'UNKNOWN' else ""
        
        report_content.append(f"RAG PIPELINE ANALYSIS SUMMARY REPORT{case_info}{split_info}")
        report_content.append("=" * 50)
        report_content.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if self.pico_analyzer.data_split != 'unknown':
            report_content.append(f"Data Split: {self.pico_analyzer.data_split.title()}")
        if self.pico_analyzer.case != 'UNKNOWN':
            report_content.append(f"Case: {self.pico_analyzer.case}")
        report_content.append("")
        
        report_content.append("PICO ANALYSIS SUMMARY:")
        report_content.append("-" * 25)
        
        if not self.pico_analyzer.picos_df.empty:
            if 'consolidated_picos' in self.pico_analyzer.data:
                report_content.append(f"Total consolidated PICOs: {len(self.pico_analyzer.data['consolidated_picos'])}")
            if 'Country' in self.pico_analyzer.picos_df.columns:
                report_content.append(f"Unique countries: {self.pico_analyzer.picos_df['Country'].nunique()}")
                report_content.append(f"Most common country: {self.pico_analyzer.picos_df['Country'].mode().iloc[0] if not self.pico_analyzer.picos_df['Country'].mode().empty else 'N/A'}")
            if 'Comparator' in self.pico_analyzer.picos_df.columns:
                report_content.append(f"Unique comparators: {self.pico_analyzer.picos_df['Comparator'].nunique()}")
                report_content.append(f"Most common comparator: {self.pico_analyzer.picos_df['Comparator'].mode().iloc[0] if not self.pico_analyzer.picos_df['Comparator'].mode().empty else 'N/A'}")
        else:
            report_content.append("No PICO data available for analysis")
        report_content.append("")
        
        report_content.append("OUTCOMES ANALYSIS SUMMARY:")
        report_content.append("-" * 27)
        
        if self.outcome_analyzer.total_outcomes > 0:
            report_content.append(f"Total outcome measures: {self.outcome_analyzer.total_outcomes}")
            metadata = self.outcome_analyzer.data.get('outcomes_metadata', {})
            if metadata.get('source_countries'):
                report_content.append(f"Countries with outcomes: {len(metadata['source_countries'])}")
                report_content.append(f"Countries: {', '.join(metadata['source_countries'])}")
            if metadata.get('source_types'):
                report_content.append(f"Source types: {', '.join(metadata['source_types'])}")
        else:
            report_content.append("No outcomes data available for analysis")
        report_content.append("")
        
        pico_countries = set()
        outcome_countries = set()
        
        if not self.pico_analyzer.picos_df.empty and 'Country' in self.pico_analyzer.picos_df.columns:
            pico_countries = set(self.pico_analyzer.picos_df['Country'].unique())
            
        metadata = self.outcome_analyzer.data.get('outcomes_metadata', {})
        if metadata.get('source_countries'):
            outcome_countries = set(metadata['source_countries'])
        
        common_countries = pico_countries.intersection(outcome_countries)
        
        report_content.append("COVERAGE ANALYSIS:")
        report_content.append("-" * 18)
        report_content.append(f"Countries with both PICOs and outcomes: {len(common_countries)}")
        report_content.append(f"PICO-only countries: {len(pico_countries - outcome_countries)}")
        report_content.append(f"Outcome-only countries: {len(outcome_countries - pico_countries)}")
        report_content.append(f"Total country coverage: {len(pico_countries.union(outcome_countries))}")
        
        if self.pico_analyzer.data_split != 'unknown':
            report_content.append("")
            report_content.append("TRAIN/TEST SPLIT INFORMATION:")
            report_content.append("-" * 30)
            report_content.append(f"Current analysis covers: {self.pico_analyzer.data_split.title()} set")
            if pico_countries:
                report_content.append(f"{self.pico_analyzer.data_split.title()} countries: {', '.join(sorted(pico_countries))}")
        
        report_filename = f'analysis_summary_report.txt'
        with open(self.output_dir / report_filename, 'w') as f:
            f.write('\n'.join(report_content))
        
        print('\n'.join(report_content))


def generate_overview_summary(pico_analyzer, outcome_analyzer, case_name):
    """Generate a nice formatted overview summary for PICOs and Outcomes"""
    data_split = getattr(pico_analyzer, 'data_split', 'unknown')
    split_info = f" ({data_split.title()} Set)" if data_split != 'unknown' else ""
    
    print("\n" + "="*100)
    print(f"{case_name.upper()} COMPREHENSIVE ANALYSIS OVERVIEW{split_info}")
    print("="*100)
    
    print("\n" + "🔬 PICO EVIDENCE OVERVIEW")
    print("-" * 50)
    
    if not pico_analyzer.picos_df.empty:
        total_picos = len(pico_analyzer.data.get('consolidated_picos', []))
        total_records = len(pico_analyzer.picos_df)
        
        print(f"📊 Total Consolidated PICOs: {total_picos}")
        print(f"📈 Total PICO Records: {total_records}")
        
        if 'Country' in pico_analyzer.picos_df.columns:
            countries = pico_analyzer.picos_df['Country'].value_counts()
            print(f"🌍 Countries Covered: {len(countries)}")
            print("   Top countries by PICOs:")
            for i, (country, count) in enumerate(countries.head(3).items()):
                print(f"   {i+1}. {country}: {count} PICOs")
        
        if 'Source_Type' in pico_analyzer.picos_df.columns:
            sources = pico_analyzer.picos_df['Source_Type'].value_counts()
            print(f"📋 Source Types: {', '.join(sources.index)}")
            for source, count in sources.items():
                print(f"   - {source.replace('_', ' ').title()}: {count} records")
        
        if 'Comparator' in pico_analyzer.picos_df.columns:
            comparators = pico_analyzer.picos_df['Comparator'].nunique()
            top_comparator = pico_analyzer.picos_df['Comparator'].mode().iloc[0] if not pico_analyzer.picos_df['Comparator'].mode().empty else 'N/A'
            print(f"⚖️  Unique Comparators: {comparators}")
            print(f"   Most Common: {top_comparator}")
    else:
        print("❌ No PICO data available for analysis")
    
    print("\n" + "🎯 OUTCOMES EVIDENCE OVERVIEW")
    print("-" * 50)
    
    if outcome_analyzer.total_outcomes > 0:
        print(f"📊 Total Outcome Measures: {outcome_analyzer.total_outcomes}")
        
        metadata = outcome_analyzer.data.get('outcomes_metadata', {})
        source_countries = metadata.get('source_countries', [])
        source_types = metadata.get('source_types', [])
        
        if source_countries:
            print(f"🌍 Countries with Outcomes: {len(source_countries)}")
            print(f"   Countries: {', '.join(source_countries)}")
        
        if source_types:
            print(f"📋 Source Types: {', '.join(source_types)}")
            for source in source_types:
                print(f"   - {source.replace('_', ' ').title()}")
    else:
        print("❌ No outcomes data available for analysis")
    
    print("\n" + "🗺️  COVERAGE SUMMARY")
    print("-" * 50)
    
    pico_countries = set()
    outcome_countries = set()
    
    if not pico_analyzer.picos_df.empty and 'Country' in pico_analyzer.picos_df.columns:
        pico_countries = set(pico_analyzer.picos_df['Country'].unique())
    
    metadata = outcome_analyzer.data.get('outcomes_metadata', {})
    if metadata.get('source_countries'):
        outcome_countries = set(metadata['source_countries'])
    
    all_countries = pico_countries.union(outcome_countries)
    common_countries = pico_countries.intersection(outcome_countries)
    
    print(f"🌐 Total Countries Covered: {len(all_countries)}")
    print(f"🤝 Countries with Both PICOs & Outcomes: {len(common_countries)}")
    print(f"🔬 PICO-Only Countries: {len(pico_countries - outcome_countries)}")
    print(f"🎯 Outcome-Only Countries: {len(outcome_countries - pico_countries)}")
    
    if common_countries:
        print(f"   Countries with complete coverage: {', '.join(sorted(common_countries))}")
    
    if data_split != 'unknown':
        print(f"\n📋 Data Split: {data_split.title()} Set")
        print(f"   Countries in this split: {', '.join(sorted(all_countries))}")
    
    print("\n" + "="*100)


def run_complete_analysis(pico_file_path, outcome_file_path, output_suffix=""):
    """
    Run complete analysis pipeline for PICO and outcomes data.
    
    Args:
        pico_file_path: Path to consolidated PICO JSON file
        outcome_file_path: Path to consolidated outcomes JSON file
        output_suffix: Suffix to add to output files
    
    Returns:
        Tuple of (PICOAnalyzer, OutcomeAnalyzer, DataVisualizer) instances
    """
    suffix_info = output_suffix.replace('_', ' ').title() if output_suffix else ""
    print(f"Starting comprehensive RAG pipeline analysis{f' for {suffix_info} set' if suffix_info else ''}...")
    print()
    
    try:
        pico_analyzer = PICOAnalyzer(pico_file_path)
        outcome_analyzer = OutcomeAnalyzer(outcome_file_path)
        
        case_name = pico_analyzer.case if pico_analyzer.case != 'UNKNOWN' else "Analysis"
        
        overview = ComprehensiveOverview()
        overview.generate_case_overview(pico_analyzer, outcome_analyzer, case_name)
        
        pico_analyzer.print_unique_picos_overview()
        outcome_analyzer.print_unique_outcomes_overview()
        
        pico_analyzer.print_summary_statistics()
        outcome_analyzer.print_summary_statistics()
        
        visualizer = DataVisualizer(pico_analyzer, outcome_analyzer)
        
        visualizer.create_pico_visualizations()
        visualizer.create_outcome_visualizations()
        visualizer.create_combined_analysis()
        
        visualizer.generate_summary_report()
        
        analysis_type = f"{suffix_info} set " if suffix_info else ""
        print(f"Analysis complete for {analysis_type}! All visualizations and reports saved to {visualizer.output_dir}/")
        
        return pico_analyzer, outcome_analyzer, visualizer
        
    except Exception as e:
        print(f"Error in complete analysis: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


class RunResults:
    """Main class to orchestrate all results analysis"""
    
    def __init__(self, translated_path="data/text_translated", results_path="results"):
        self.translated_path = translated_path
        self.results_path = results_path
        self.cases = ["NSCLC", "HCC"]
        
    def run_translation_analysis(self):
        """Run translation quality analysis"""
        print("\n" + "="*100)
        print("TRANSLATION QUALITY ANALYSIS")
        print("="*100)
        
        translation_analyzer = TranslationAnalyzer(translated_path=self.translated_path)
        translation_analyzer.run_complete_analysis()
    
    def run_comprehensive_overview(self):
        """Generate comprehensive overview for all cases"""
        print("\n" + "="*100)
        print("COMPREHENSIVE RESULTS ANALYSIS")
        print("="*100)
        
        print("\n" + "📋 GENERATING COMPREHENSIVE OVERVIEW FOR ALL CASES")
        print("="*80)
        
        comprehensive_overview = ComprehensiveOverview()
        
        all_pico_files_train = []
        all_outcome_files_train = []
        all_pico_files_test = []
        all_outcome_files_test = []
        
        for case in self.cases:
            case_dir = Path(f"{self.results_path}/{case}/consolidated")
            if case_dir.exists():
                train_pico_files = list(case_dir.glob("*consolidated_picos_train*.json"))
                train_outcome_files = list(case_dir.glob("*consolidated_outcomes_train*.json"))
                
                test_pico_files = list(case_dir.glob("*consolidated_picos_test*.json"))
                test_outcome_files = list(case_dir.glob("*consolidated_outcomes_test*.json"))
                
                if train_pico_files and train_outcome_files:
                    all_pico_files_train.extend([(max(train_pico_files, key=os.path.getmtime), case)])
                    all_outcome_files_train.extend([(max(train_outcome_files, key=os.path.getmtime), case)])
                
                if test_pico_files and test_outcome_files:
                    all_pico_files_test.extend([(max(test_pico_files, key=os.path.getmtime), case)])
                    all_outcome_files_test.extend([(max(test_outcome_files, key=os.path.getmtime), case)])
        
        if all_pico_files_train and all_outcome_files_train:
            print("\n--- Generating Training Set Overview ---")
            comprehensive_overview.generate_cross_case_overview(
                all_pico_files_train, 
                all_outcome_files_train,
                output_suffix="_train"
            )
        
        if all_pico_files_test and all_outcome_files_test:
            print("\n--- Generating Test Set Overview ---")
            comprehensive_overview.generate_cross_case_overview(
                all_pico_files_test, 
                all_outcome_files_test,
                output_suffix="_test"
            )
    
    def run_case_analysis(self, case_name, splits=["train", "test"]):
        """Run detailed analysis for a specific case"""
        print(f"\n=== {case_name.upper()} DETAILED ANALYSIS ===")
        
        consolidated_dir = Path(f"{self.results_path}/{case_name}/consolidated")
        if not consolidated_dir.exists():
            print(f"Warning: {self.results_path}/{case_name}/consolidated directory not found.")
            print(f"Make sure the {case_name} consolidation step completed successfully.")
            return
        
        for split in splits:
            print(f"\n--- {case_name} {split.title()} Set Analysis ---")
            
            pico_files = list(consolidated_dir.glob(f"*consolidated_picos_{split}*.json"))
            outcome_files = list(consolidated_dir.glob(f"*consolidated_outcomes_{split}*.json"))
            
            if pico_files and outcome_files:
                pico_file = max(pico_files, key=os.path.getmtime)
                outcome_file = max(outcome_files, key=os.path.getmtime)
                
                print(f"Analyzing {case_name} {split.title()} PICO data from: {pico_file}")
                print(f"Analyzing {case_name} {split.title()} Outcomes data from: {outcome_file}")
                print()
                
                try:
                    pico_analyzer, outcome_analyzer, visualizer = run_complete_analysis(
                        pico_file_path=str(pico_file),
                        outcome_file_path=str(outcome_file),
                        output_suffix=f"_{split}"
                    )
                    print(f"{case_name} {split} set analysis completed successfully!")
                except Exception as e:
                    print(f"Error in {case_name} {split} set analysis: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"Warning: Could not find {case_name} {split} set consolidated files.")
                if not pico_files:
                    print(f"Missing {case_name} PICO {split} files in {consolidated_dir}/")
                if not outcome_files:
                    print(f"Missing {case_name} Outcomes {split} files in {consolidated_dir}/")
    
    def print_split_summary(self):
        """Print train/test split summary for all cases"""
        print("\n" + "="*100)
        print("TRAIN/TEST SPLIT SUMMARY")
        print("="*100)
        
        for case_name in self.cases:
            consolidated_dir = Path(f"{self.results_path}/{case_name}/consolidated")
            if not consolidated_dir.exists():
                print(f"{case_name}: No consolidated directory found")
                continue
            
            train_files = len(list(consolidated_dir.glob("*_train_*.json")))
            test_files = len(list(consolidated_dir.glob("*_test_*.json")))
            
            print(f"{case_name}:")
            print(f"  Training files: {train_files}")
            print(f"  Test files: {test_files}")
            
            train_pico_files = list(consolidated_dir.glob("*consolidated_picos_train*.json"))
            test_pico_files = list(consolidated_dir.glob("*consolidated_picos_test*.json"))
            
            if train_pico_files:
                try:
                    with open(train_pico_files[0], 'r', encoding='utf-8') as f:
                        train_data = json.load(f)
                    train_countries = train_data.get("consolidation_metadata", {}).get("source_countries", [])
                    print(f"  Training countries: {', '.join(train_countries) if train_countries else 'None'}")
                except:
                    print(f"  Training countries: Unable to read")
            
            if test_pico_files:
                try:
                    with open(test_pico_files[0], 'r', encoding='utf-8') as f:
                        test_data = json.load(f)
                    test_countries = test_data.get("consolidation_metadata", {}).get("source_countries", [])
                    print(f"  Test countries: {', '.join(test_countries) if test_countries else 'None'}")
                except:
                    print(f"  Test countries: Unable to read")
            
            print()
    
    def run_all(self):
        """Run all results analysis"""
        self.run_translation_analysis()
        
        self.run_comprehensive_overview()
        
        self.run_case_analysis("NSCLC", splits=["train", "test"])
        
        self.run_case_analysis("HCC", splits=["test"])
        
        self.print_split_summary()
            