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
    from matplotlib_venn import venn2, venn2_circles
    VENN_AVAILABLE = True
except ImportError:
    VENN_AVAILABLE = False
    print("Warning: matplotlib-venn not available. Install with: pip install matplotlib-venn")


class TranslationAnalyzer:
    
    def __init__(self, translated_path="data/text_translated"):
        self.translated_path = Path(translated_path)
        self.translation_data = []
        
    def load_translation_metadata(self):
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
        valid_scores = [s for s in scores_list if s is not None]
        
        if not valid_scores:
            return None, None
            
        mean = np.mean(valid_scores)
        std = np.std(valid_scores)
        
        return mean, std
    
    def analyze_group(self, documents, group_name):
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
    
    def __init__(self):
        self.all_cases_data = {}
    
    def generate_case_overview(self, pico_analyzer, outcome_analyzer, case_name):
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
        file_path = str(self.pico_file_path).lower()
        if '_train_' in file_path or file_path.endswith('_train.json'):
            return 'train'
        elif '_test_' in file_path or file_path.endswith('_test.json'):
            return 'test'
        else:
            return 'unknown'
    
    def _extract_case_name(self):
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
        file_path = str(self.outcome_file_path).lower()
        if '_train_' in file_path or file_path.endswith('_train.json'):
            return 'train'
        elif '_test_' in file_path or file_path.endswith('_test.json'):
            return 'test'
        else:
            return 'unknown'
    
    def _extract_case_name(self):
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
    def __init__(self, output_dir="results/visualizations"):
        self.output_dir = Path(output_dir)
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
    
    def get_comparators_by_source_type(self, pico_analyzer):
        if 'consolidated_picos' not in pico_analyzer.data:
            return set(), set()
            
        guideline_comparators = set()
        hta_comparators = set()
        
        for pico in pico_analyzer.data['consolidated_picos']:
            comparator = pico.get('Comparator', '')
            source_types = pico.get('Source_Types', [])
            
            if 'clinical_guideline' in source_types:
                guideline_comparators.add(comparator)
            if 'hta_submission' in source_types:
                hta_comparators.add(comparator)
                
        return guideline_comparators, hta_comparators
    
    def create_combined_european_map(self, pico_analyzers):
        print("Creating combined European PICO distribution map...")
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 20))
        
        # Define all countries from the file tree for each case
        all_countries_by_case = {
            'HCC': ['AT', 'BE', 'CZ', 'DE', 'DK', 'EN', 'ES', 'EU', 'FR', 'IT', 'NL', 'PO', 'PT', 'SE'],
            'NSCLC': ['AT', 'DE', 'DK', 'EL', 'EN', 'ES', 'EU', 'FR', 'HR', 'IE', 'IT', 'NL', 'PO', 'PT', 'SE']
        }
        
        # Sort analyzers to put NSCLC first, HCC second
        sorted_analyzers = sorted(pico_analyzers, key=lambda x: 0 if x.case == 'NSCLC' else 1)
        
        for idx, pico_analyzer in enumerate(sorted_analyzers):
            ax = axes[idx]
            case_name = pico_analyzer.case
            
            # Get countries with PICOs
            country_counts = {}
            if not pico_analyzer.picos_df.empty and 'Country' in pico_analyzer.picos_df.columns:
                country_counts = pico_analyzer.picos_df['Country'].value_counts().to_dict()
            
            # Get all countries that should be shown for this case
            all_case_countries = all_countries_by_case.get(case_name, [])
            
            self._plot_european_map(ax, country_counts, case_name, all_case_countries)
        
        axes[0].set_position([0.1, 0.53, 0.75, 0.40])
        axes[1].set_position([0.1, 0.05, 0.75, 0.40])
        
        plt.savefig(self.output_dir / 'combined_european_pico_heatmap.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()

    def _plot_european_map(self, ax, country_counts, case_name, all_case_countries):
        if not GEOPANDAS_AVAILABLE:
            self._plot_simplified_european_map(ax, country_counts, case_name, all_case_countries)
            return
        
        try:
            world_url = "https://raw.githubusercontent.com/holtzy/The-Python-Graph-Gallery/master/static/data/world-110m2.json"
            
            try:
                world = gpd.read_file(world_url)
            except:
                world_url_alt = "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
                try:
                    world = gpd.read_file(world_url_alt)
                except:
                    self._plot_simplified_european_map(ax, country_counts, case_name, all_case_countries)
                    return
            
            # Define pastel blue for training set
            training_color = '#A8D8EA'  # Pastel blue
            no_pico_color = '#4A4A4A'  # Grey for countries with no PICOs
            
            country_mapping = {
                'DE': 'Germany', 'DK': 'Denmark', 'EN': 'United Kingdom', 'FR': 'France',
                'NL': 'Netherlands', 'PO': 'Poland', 'PT': 'Portugal', 'SE': 'Sweden',
                'AT': 'Austria', 'IT': 'Italy', 'ES': 'Spain', 'BE': 'Belgium',
                'CH': 'Switzerland', 'NO': 'Norway', 'FI': 'Finland', 'IE': 'Ireland',
                'CZ': 'Czech Republic', 'HU': 'Hungary', 'RO': 'Romania', 'BG': 'Bulgaria',
                'GR': 'Greece', 'HR': 'Croatia', 'SI': 'Slovenia', 'SK': 'Slovakia',
                'LT': 'Lithuania', 'LV': 'Latvia', 'EE': 'Estonia', 'EL': 'Greece',
                'EU': 'Europe', 'CY': 'Cyprus', 'LU': 'Luxembourg', 'MT': 'Malta'
            }
            
            # Define training countries by case
            training_countries_by_case = {
                'NSCLC': ['PO', 'NL', 'AT'],
                'HCC': []
            }
            
            training_country_codes = training_countries_by_case.get(case_name, [])
            
            name_column = 'NAME' if 'NAME' in world.columns else 'name'
            if name_column not in world.columns:
                for col in ['NAME_EN', 'NAME_LONG', 'ADMIN', 'Country']:
                    if col in world.columns:
                        name_column = col
                        break
            
            # Create data for all countries that should be shown
            pico_data = []
            for country_code in all_case_countries:
                if country_code in country_mapping:
                    country_name = country_mapping[country_code]
                    count = country_counts.get(country_code, 0)
                    is_training = country_code in training_country_codes
                    
                    pico_data.append({
                        name_column: country_name, 
                        'pico_count': count, 
                        'code': country_code,
                        'is_training': is_training
                    })
            
            pico_df = pd.DataFrame(pico_data)
            
            # All EU member states plus some others to show complete map
            european_countries = [
                'Germany', 'Denmark', 'United Kingdom', 'France', 'Netherlands', 
                'Poland', 'Portugal', 'Sweden', 'Norway', 'Finland', 'Spain', 
                'Italy', 'Switzerland', 'Austria', 'Belgium', 'Ireland',
                'Czech Republic', 'Czechia', 'Hungary', 'Romania', 'Bulgaria', 'Greece',
                'Croatia', 'Slovenia', 'Slovakia', 'Lithuania', 'Latvia', 'Estonia',
                'Cyprus', 'Luxembourg', 'Malta'
            ]
            
            europe = world[world[name_column].isin(european_countries)].copy()
            europe = europe.merge(pico_df, on=name_column, how='left')
            europe['pico_count'] = europe['pico_count'].fillna(0).astype(int)
            europe['is_training'] = europe['is_training'].fillna(False)
            europe['code'] = europe['code'].fillna('')
            
            ax.set_xlim(-12, 32)
            ax.set_ylim(35, 72)
            
            # Plot countries with no data (not in our dataset)
            europe_no_data = europe[europe['code'] == '']
            if not europe_no_data.empty:
                europe_no_data.plot(ax=ax, color='#F0F0F0', edgecolor='white', linewidth=0.8, zorder=1)
            
            # Plot countries with 0 PICOs (in dataset but no PICOs found)
            europe_zero_picos = europe[(europe['pico_count'] == 0) & (~europe['is_training']) & (europe['code'] != '')]
            if not europe_zero_picos.empty:
                europe_zero_picos.plot(
                    ax=ax,
                    color=no_pico_color,
                    edgecolor='white',
                    linewidth=0.8,
                    alpha=0.6,
                    zorder=2
                )
            
            # Plot training countries
            europe_training = europe[(europe['is_training'] == True)]
            if not europe_training.empty:
                europe_training.plot(
                    ax=ax,
                    color=training_color,
                    edgecolor='white',
                    linewidth=0.8,
                    zorder=4
                )
            
            # Plot countries with PICOs (non-training)
            europe_with_data = europe[(europe['pico_count'] > 0) & (~europe['is_training'])].copy()
            
            if not europe_with_data.empty:
                vmin = 1
                vmax = max(europe_with_data['pico_count'].max(), 1)
                
                europe_with_data.plot(
                    column='pico_count', 
                    ax=ax, 
                    cmap='YlOrRd',
                    legend=False,
                    edgecolor='white', 
                    linewidth=0.8,
                    vmin=vmin,
                    vmax=vmax,
                    zorder=3
                )
                
                sm = plt.cm.ScalarMappable(cmap='YlOrRd', 
                                        norm=plt.Normalize(vmin=vmin, vmax=vmax))
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=ax, shrink=0.6, aspect=30)
                cbar.set_label('Number of PICOs', fontweight='bold', fontsize=14)
                cbar.ax.tick_params(labelsize=12)
            
            # Annotate all countries in dataset
            for idx, row in europe[europe['code'] != ''].iterrows():
                try:
                    centroid = row.geometry.centroid
                    
                    if row[name_column] == 'France':
                        x_pos, y_pos = 2.2, 46.2
                    else:
                        x_pos, y_pos = centroid.x, centroid.y
                    
                    label_text = f"{row['code']}"
                    if row['pico_count'] > 0:
                        label_text += f"\n({int(row['pico_count'])})"
                    
                    # All text in black with white background
                    ax.annotate(
                        label_text,
                        xy=(x_pos, y_pos),
                        ha='center', va='center',
                        fontsize=12, fontweight='bold',
                        color='black',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                alpha=0.8, edgecolor='none'),
                        zorder=6
                    )
                except Exception as e:
                    pass
            
            # Create legend
            legend_elements = []
            
            # Add training set legend if applicable
            if training_country_codes:
                legend_elements.append(
                    plt.Rectangle((0,0),1,1, facecolor=training_color, 
                                edgecolor='white', linewidth=0.8, label='Training Set')
                )
            
            # Add "No PICOs found" legend if there are countries with 0 PICOs
            if not europe_zero_picos.empty:
                legend_elements.append(
                    plt.Rectangle((0,0),1,1, facecolor=no_pico_color, 
                                edgecolor='white', linewidth=0.8, alpha=0.6, label='No PICOs Found')
                )
            
            if legend_elements:
                legend = ax.legend(handles=legend_elements, loc='upper left', 
                        bbox_to_anchor=(0.02, 0.98), frameon=True, fancybox=True, 
                        shadow=True, fontsize=16, bbox_transform=ax.transAxes)
                legend.set_zorder(10)
                ax.add_artist(legend)
            
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.set_aspect('equal', adjustable='box')
            
            total_countries = len([c for c in all_case_countries if c in country_counts and country_counts[c] > 0])
            ax.set_title(f'{case_name} - European Distribution ({total_countries} countries with PICOs)', 
                        fontsize=20, fontweight='bold', pad=15)
            
        except Exception as e:
            print(f"Error creating European map for {case_name}: {str(e)}")
            self._plot_simplified_european_map(ax, country_counts, case_name, all_case_countries)

    def _plot_simplified_european_map(self, ax, country_counts, case_name, all_case_countries):
        # Define pastel blue for training set
        training_color = '#A8D8EA'  # Pastel blue
        no_pico_color = '#4A4A4A'  # Grey for countries with no PICOs
        
        country_mapping = {
            'DE': 'Germany', 'DK': 'Denmark', 'EN': 'United Kingdom', 'FR': 'France',
            'NL': 'Netherlands', 'PO': 'Poland', 'PT': 'Portugal', 'SE': 'Sweden',
            'AT': 'Austria', 'IT': 'Italy', 'ES': 'Spain', 'BE': 'Belgium',
            'CH': 'Switzerland', 'NO': 'Norway', 'FI': 'Finland', 'IE': 'Ireland',
            'CZ': 'Czech Republic', 'HU': 'Hungary', 'RO': 'Romania', 'BG': 'Bulgaria',
            'GR': 'Greece', 'HR': 'Croatia', 'SI': 'Slovenia', 'SK': 'Slovakia',
            'LT': 'Lithuania', 'LV': 'Latvia', 'EE': 'Estonia', 'EL': 'Greece',
            'EU': 'Europe', 'CY': 'Cyprus', 'LU': 'Luxembourg', 'MT': 'Malta'
        }
        
        european_positions = {
            'Germany': (10.5, 51.5), 'Denmark': (10.0, 56.0), 'United Kingdom': (-2.0, 54.0),
            'France': (2.2, 46.2), 'Netherlands': (5.5, 52.0), 'Poland': (19.0, 52.0),
            'Portugal': (-8.0, 39.5), 'Sweden': (15.0, 62.0), 'Austria': (13.5, 47.5),
            'Italy': (12.5, 42.0), 'Spain': (-4.0, 40.0), 'Belgium': (4.5, 50.5),
            'Switzerland': (8.2, 46.8), 'Norway': (8.0, 60.0), 'Finland': (25.0, 64.0),
            'Ireland': (-8.0, 53.0), 'Czech Republic': (15.5, 49.8), 'Hungary': (19.5, 47.0),
            'Romania': (24.0, 45.9), 'Bulgaria': (25.0, 42.7), 'Greece': (22.0, 39.0),
            'Croatia': (15.5, 45.1), 'Slovenia': (14.5, 46.1), 'Slovakia': (19.5, 48.7),
            'Lithuania': (23.9, 55.3), 'Latvia': (24.0, 56.9), 'Estonia': (25.0, 58.6),
            'Europe': (10.0, 50.0), 'Cyprus': (33.0, 35.0), 'Luxembourg': (6.1, 49.8),
            'Malta': (14.5, 35.9)
        }
        
        # Define training countries by case
        training_countries_by_case = {
            'NSCLC': ['PO', 'NL', 'AT'],
            'HCC': []
        }
        
        training_country_codes = training_countries_by_case.get(case_name, [])
        
        ax.set_xlim(-12, 35)
        ax.set_ylim(34, 70)
        
        ax.add_patch(plt.Rectangle((-10, 35), 45, 35, 
                                facecolor='#F8F9FA', edgecolor='#E9ECEF', 
                                alpha=0.5, linewidth=1))
        
        if len(country_counts) == 0 and len(all_case_countries) == 0:
            return
        
        # Calculate sizes for countries with PICOs
        if country_counts:
            max_count = max(country_counts.values())
            min_count = min(country_counts.values())
        else:
            max_count = 1
            min_count = 1
        
        # Plot all countries in the dataset
        for country_code in all_case_countries:
            if country_code in country_mapping:
                country_name = country_mapping[country_code]
                if country_name in european_positions:
                    lon, lat = european_positions[country_name]
                    count = country_counts.get(country_code, 0)
                    is_training = country_code in training_country_codes
                    
                    # Determine circle size and color
                    if is_training:
                        circle_size = 800
                        color = training_color
                        alpha = 0.8
                        label = f'{country_code}\n(Train)'
                    elif count == 0:
                        circle_size = 600
                        color = no_pico_color
                        alpha = 0.6
                        label = f'{country_code}'
                    else:
                        if max_count > min_count:
                            normalized_size = (count - min_count) / (max_count - min_count)
                        else:
                            normalized_size = 1.0
                        
                        circle_size = 300 + normalized_size * 700
                        color_intensity = 0.3 + 0.7 * normalized_size
                        color = plt.cm.YlOrRd(color_intensity)
                        alpha = 0.8
                        label = f'{country_code}\n({count})'
                    
                    ax.scatter(lon, lat, s=circle_size, 
                            color=color, 
                            alpha=alpha, edgecolors='white', linewidth=2,
                            zorder=5)
                    
                    # All text in black with white background
                    ax.annotate(label, 
                            (lon, lat), xytext=(0, 0), 
                            textcoords='offset points',
                            ha='center', va='center',
                            fontsize=12, fontweight='bold',
                            color='black',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                    alpha=0.8, edgecolor='none'),
                            zorder=6)
        
        # Create legend
        legend_elements = []
        
        # Add training set legend if applicable
        if training_country_codes:
            legend_elements.append(
                plt.Rectangle((0,0),1,1, facecolor=training_color, 
                            edgecolor='white', linewidth=2, alpha=0.8, label='Training Set')
            )
        
        # Add "No PICOs found" legend if there are countries with 0 PICOs
        countries_with_zero_picos = [c for c in all_case_countries if country_counts.get(c, 0) == 0 and c not in training_country_codes]
        if countries_with_zero_picos:
            legend_elements.append(
                plt.Rectangle((0,0),1,1, facecolor=no_pico_color, 
                            edgecolor='white', linewidth=2, alpha=0.6, label='No PICOs Found')
            )
        
        if legend_elements:
            legend = ax.legend(handles=legend_elements, loc='upper left', 
                    bbox_to_anchor=(0.02, 0.98), frameon=True, fancybox=True, 
                    shadow=True, fontsize=16, bbox_transform=ax.transAxes)
            legend.set_zorder(10)
            ax.add_artist(legend)
        
        ax.set_xlabel('Longitude', fontweight='bold', fontsize=12)
        ax.set_ylabel('Latitude', fontweight='bold', fontsize=12)
        ax.set_aspect('equal', adjustable='box')
        
        total_countries = len([c for c in all_case_countries if country_counts.get(c, 0) > 0])
        ax.set_title(f'{case_name} - European Distribution ({total_countries} countries with PICOs - Simplified View)', 
                    fontsize=20, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3)

    def create_combined_venn_diagram(self, pico_analyzers):
        if not VENN_AVAILABLE:
            print("Matplotlib-venn not available. Skipping Venn diagram.")
            return
        
        print("Creating combined comparator Venn diagrams...")
        
        fig, axes = plt.subplots(2, 1, figsize=(10, 16))
        fig.suptitle('Comparator Overlap: Guidelines vs HTA Submissions', fontsize=16, fontweight='bold', y=0.995)
        
        for idx, pico_analyzer in enumerate(pico_analyzers):
            ax = axes[idx]
            case_name = pico_analyzer.case
            
            guideline_comparators, hta_comparators = self.get_comparators_by_source_type(pico_analyzer)
            
            guideline_comparators.discard('')
            hta_comparators.discard('')
            
            if not guideline_comparators and not hta_comparators:
                ax.text(0.5, 0.5, f'No {case_name} comparator data available', 
                       ha='center', va='center', transform=ax.transAxes)
                continue
            
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
            
            ax.set_title(f'{case_name} - Total Unique Comparators: {len(guideline_comparators.union(hta_comparators))}',
                        fontsize=14, fontweight='bold', pad=20)
            
            stats_text = (f"Guidelines Only: {len(guideline_only)}\n"
                         f"HTA Only: {len(hta_only)}\n"
                         f"Both Sources: {len(overlap)}")
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                   fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'combined_comparator_venn_diagram.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
    
    def create_combined_source_type_comparison(self, pico_analyzers, outcome_analyzers):
        print("Creating combined source type distribution comparison...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle('Source Type Distribution Comparison', fontsize=16, fontweight='bold', y=0.995)
        
        for idx, (pico_analyzer, outcome_analyzer) in enumerate(zip(pico_analyzers, outcome_analyzers)):
            case_name = pico_analyzer.case
            
            ax_pico = axes[idx, 0]
            ax_outcome = axes[idx, 1]
            
            pico_has_source = not pico_analyzer.picos_df.empty and 'Source_Type' in pico_analyzer.picos_df.columns
            outcome_has_source = outcome_analyzer.total_outcomes > 0
            
            if pico_has_source:
                pico_sources = pico_analyzer.picos_df['Source_Type'].value_counts()
                colors_pico = [self.scientific_colors['primary'], self.scientific_colors['secondary']][:len(pico_sources)]
                
                bars1 = ax_pico.bar(range(len(pico_sources)), pico_sources.values, 
                                   color=colors_pico, alpha=0.8, edgecolor='white', linewidth=0.8)
                ax_pico.set_xticks(range(len(pico_sources)))
                ax_pico.set_xticklabels([label.replace('_', ' ').title() for label in pico_sources.index],
                                       rotation=45, ha='right', fontweight='bold')
                ax_pico.set_ylabel('Number of PICOs', fontweight='bold')
                ax_pico.set_xlabel('Source Type', fontweight='bold')
                ax_pico.set_title(f'{case_name} - PICOs by Source Type', fontweight='bold', pad=20)
                ax_pico.grid(True, alpha=0.3, axis='y')
                
                for bar, value in zip(bars1, pico_sources.values):
                    ax_pico.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(pico_sources.values) * 0.01,
                               str(value), ha='center', va='bottom', fontweight='bold')
            else:
                ax_pico.text(0.5, 0.5, f'No {case_name} PICO Source Type Data', 
                            ha='center', va='center', transform=ax_pico.transAxes)
                ax_pico.set_title(f'{case_name} - PICOs by Source Type', fontweight='bold', pad=20)
            
            if outcome_has_source:
                metadata = outcome_analyzer.data.get('outcomes_metadata', {})
                source_types = metadata.get('source_types', [])
                
                if source_types:
                    colors_outcome = [self.scientific_colors['tertiary'], self.scientific_colors['quaternary']][:len(source_types)]
                    outcome_values = [outcome_analyzer.total_outcomes] * len(source_types)
                    
                    bars2 = ax_outcome.bar(range(len(source_types)), outcome_values,
                                           color=colors_outcome, alpha=0.8, edgecolor='white', linewidth=0.8)
                    ax_outcome.set_xticks(range(len(source_types)))
                    ax_outcome.set_xticklabels([label.replace('_', ' ').title() for label in source_types],
                                               rotation=45, ha='right', fontweight='bold')
                    ax_outcome.set_ylabel('Coverage Indicator', fontweight='bold')
                    ax_outcome.set_xlabel('Source Type', fontweight='bold')
                    ax_outcome.set_title(f'{case_name} - Outcomes by Source Type', fontweight='bold', pad=20)
                    ax_outcome.grid(True, alpha=0.3, axis='y')
                    
                    for bar, value in zip(bars2, outcome_values):
                        ax_outcome.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(outcome_values) * 0.01,
                                       str(value), ha='center', va='bottom', fontweight='bold')
                else:
                    ax_outcome.text(0.5, 0.5, f'No {case_name} Outcome Source Type Data', 
                                    ha='center', va='center', transform=ax_outcome.transAxes)
                    ax_outcome.set_title(f'{case_name} - Outcomes by Source Type', fontweight='bold', pad=20)
            else:
                ax_outcome.text(0.5, 0.5, f'No {case_name} Outcome Source Type Data', 
                                ha='center', va='center', transform=ax_outcome.transAxes)
                ax_outcome.set_title(f'{case_name} - Outcomes by Source Type', fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'combined_source_type_comparison.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
    
    def generate_summary_report(self, pico_analyzers, outcome_analyzers):
        print("Generating summary report...")
        
        report_content = []
        
        report_content.append(f"RAG PIPELINE ANALYSIS SUMMARY REPORT")
        report_content.append("=" * 50)
        report_content.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_content.append("")
        
        for pico_analyzer, outcome_analyzer in zip(pico_analyzers, outcome_analyzers):
            case_name = pico_analyzer.case
            split_info = f" ({pico_analyzer.data_split.title()} Set)" if pico_analyzer.data_split != 'unknown' else ""
            
            report_content.append(f"{case_name}{split_info} ANALYSIS:")
            report_content.append("-" * 30)
            
            if not pico_analyzer.picos_df.empty:
                if 'consolidated_picos' in pico_analyzer.data:
                    report_content.append(f"Total consolidated PICOs: {len(pico_analyzer.data['consolidated_picos'])}")
                if 'Country' in pico_analyzer.picos_df.columns:
                    report_content.append(f"Unique countries: {pico_analyzer.picos_df['Country'].nunique()}")
                    report_content.append(f"Most common country: {pico_analyzer.picos_df['Country'].mode().iloc[0] if not pico_analyzer.picos_df['Country'].mode().empty else 'N/A'}")
                if 'Comparator' in pico_analyzer.picos_df.columns:
                    report_content.append(f"Unique comparators: {pico_analyzer.picos_df['Comparator'].nunique()}")
            else:
                report_content.append("No PICO data available")
            
            if outcome_analyzer.total_outcomes > 0:
                report_content.append(f"Total outcome measures: {outcome_analyzer.total_outcomes}")
                metadata = outcome_analyzer.data.get('outcomes_metadata', {})
                if metadata.get('source_countries'):
                    report_content.append(f"Countries with outcomes: {len(metadata['source_countries'])}")
            else:
                report_content.append("No outcomes data available")
            
            report_content.append("")
        
        report_filename = f'combined_analysis_summary_report.txt'
        with open(self.output_dir / report_filename, 'w') as f:
            f.write('\n'.join(report_content))
        
        print('\n'.join(report_content))


def run_complete_analysis(pico_file_path, outcome_file_path, output_suffix=""):
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
        
        return pico_analyzer, outcome_analyzer
        
    except Exception as e:
        print(f"Error in complete analysis: {e}")
        import traceback
        traceback.print_exc()
        return None, None


class RunResults:
    
    def __init__(self, translated_path="data/text_translated", results_path="results", mode="full", simulation_ids=None):
        """
        Initialize the RunResults analyzer.
        
        Args:
            translated_path: Path to translated documents
            results_path: Path to results directory
            mode: Analysis mode - "full" for complete analysis with visualizations,
                  "consolidated_only" for consolidated results only
            simulation_ids: List of simulation IDs to analyze. If None, analyzes all simulations.
                           Example: ["base", "base_b", "base_c", "sim1", "sim2"]
        """
        self.translated_path = translated_path
        self.results_path = results_path
        self.mode = mode
        self.simulation_ids = simulation_ids  # NEW: Store simulation IDs to analyze
        
    def _get_case_folders(self):
        """Detect available case folders regardless of case sensitivity."""
        if not os.path.exists(self.results_path):
            return []
        
        case_folders = []
        for item in os.listdir(self.results_path):
            item_path = os.path.join(self.results_path, item)
            if os.path.isdir(item_path):
                if item.upper() in ["NSCLC", "HCC", "SCLC", "BREAST", "LUNG"]:
                    case_folders.append(item)
        
        return case_folders
    
    def _get_simulation_folders(self):
        """
        Get simulation folders to analyze, filtered by self.simulation_ids if provided.
        
        Returns:
            List of simulation folder names (e.g., ["base", "base_b", "sim1"])
        """
        results_path_obj = Path(self.results_path)
        
        # Determine the parent results path
        if results_path_obj.name in ["base"] or results_path_obj.name.startswith("sim") or results_path_obj.name.startswith("base_"):
            parent_results_path = results_path_obj.parent
        else:
            parent_results_path = results_path_obj
        
        # Get all simulation folders
        all_simulation_folders = []
        if parent_results_path.exists():
            for item in sorted(os.listdir(parent_results_path)):
                item_path = os.path.join(parent_results_path, item)
                # Include folders that are "base" or start with "sim" or "base_"
                if os.path.isdir(item_path) and (item == "base" or item.startswith("sim") or item.startswith("base_")):
                    all_simulation_folders.append(item)
        
        # Filter by simulation_ids if provided
        if self.simulation_ids is not None:
            filtered_folders = [sim for sim in all_simulation_folders if sim in self.simulation_ids]
            return filtered_folders, parent_results_path
        
        return all_simulation_folders, parent_results_path
    
    def run_translation_analysis(self):
        print("\n" + "="*100)
        print("TRANSLATION QUALITY ANALYSIS")
        print("="*100)
        
        translation_analyzer = TranslationAnalyzer(translated_path=self.translated_path)
        translation_analyzer.run_complete_analysis()
    
    def run_comprehensive_overview(self):
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
        
        case_folders = self._get_case_folders()
        
        for case_folder in case_folders:
            case_dir = Path(f"{self.results_path}/{case_folder}/consolidated")
            if case_dir.exists():
                train_pico_files = list(case_dir.glob("*consolidated_picos_train*.json"))
                train_outcome_files = list(case_dir.glob("*consolidated_outcomes_train*.json"))
                
                test_pico_files = list(case_dir.glob("*consolidated_picos_test*.json"))
                test_outcome_files = list(case_dir.glob("*consolidated_outcomes_test*.json"))
                
                if train_pico_files and train_outcome_files:
                    all_pico_files_train.extend([(max(train_pico_files, key=os.path.getmtime), case_folder.upper())])
                    all_outcome_files_train.extend([(max(train_outcome_files, key=os.path.getmtime), case_folder.upper())])
                
                if test_pico_files and test_outcome_files:
                    all_pico_files_test.extend([(max(test_pico_files, key=os.path.getmtime), case_folder.upper())])
                    all_outcome_files_test.extend([(max(test_outcome_files, key=os.path.getmtime), case_folder.upper())])
        
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
    
    def run_case_analysis(self, case_folder, splits=["train", "test"]):
        print(f"\n=== {case_folder.upper()} DETAILED ANALYSIS ===")
        
        consolidated_dir = Path(f"{self.results_path}/{case_folder}/consolidated")
        if not consolidated_dir.exists():
            print(f"Warning: {self.results_path}/{case_folder}/consolidated directory not found.")
            return
        
        analyzers = []
        for split in splits:
            print(f"\n--- {case_folder.upper()} {split.title()} Set Analysis ---")
            
            pico_files = list(consolidated_dir.glob(f"*consolidated_picos_{split}*.json"))
            outcome_files = list(consolidated_dir.glob(f"*consolidated_outcomes_{split}*.json"))
            
            if pico_files and outcome_files:
                pico_file = max(pico_files, key=os.path.getmtime)
                outcome_file = max(outcome_files, key=os.path.getmtime)
                
                print(f"Analyzing {case_folder.upper()} {split.title()} PICO data from: {pico_file}")
                print(f"Analyzing {case_folder.upper()} {split.title()} Outcomes data from: {outcome_file}")
                print()
                
                try:
                    pico_analyzer, outcome_analyzer = run_complete_analysis(
                        pico_file_path=str(pico_file),
                        outcome_file_path=str(outcome_file),
                        output_suffix=f"_{split}"
                    )
                    if pico_analyzer and outcome_analyzer:
                        analyzers.append((pico_analyzer, outcome_analyzer))
                    print(f"{case_folder.upper()} {split} set analysis completed successfully!")
                except Exception as e:
                    print(f"Error in {case_folder.upper()} {split} set analysis: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"Warning: Could not find {case_folder.upper()} {split} set consolidated files.")
        
        return analyzers
    
    def print_split_summary(self):
        print("\n" + "="*100)
        print("TRAIN/TEST SPLIT SUMMARY")
        print("="*100)
        
        case_folders = self._get_case_folders()
        
        for case_folder in case_folders:
            consolidated_dir = Path(f"{self.results_path}/{case_folder}/consolidated")
            if not consolidated_dir.exists():
                print(f"{case_folder.upper()}: No consolidated directory found")
                continue
            
            train_files = len(list(consolidated_dir.glob("*_train_*.json")))
            test_files = len(list(consolidated_dir.glob("*_test_*.json")))
            
            print(f"{case_folder.upper()}:")
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
    
    def run_consolidated_analysis(self):
        """Print consolidated PICOs and outcomes for specified simulations across all cases."""
        print("\n" + "="*100)
        print("CONSOLIDATED RESULTS ANALYSIS - SPECIFIED SIMULATIONS")
        print("="*100)
        
        simulation_folders, parent_results_path = self._get_simulation_folders()
        
        if not simulation_folders:
            print("No simulation folders found matching the specified simulation IDs.")
            if self.simulation_ids:
                print(f"Requested simulations: {', '.join(self.simulation_ids)}")
            return
        
        print(f"\nAnalyzing {len(simulation_folders)} simulation(s): {', '.join(simulation_folders)}")
        if self.simulation_ids:
            print(f"(Filtered from requested: {', '.join(self.simulation_ids)})")
        
        case_types = ["nsclc", "hcc"]
        
        for case_type in case_types:
            print(f"\n{'#'*100}")
            print(f"# {case_type.upper()} RESULTS ACROSS SPECIFIED SIMULATIONS")
            print(f"{'#'*100}")
            
            if case_type.upper() == "NSCLC":
                splits = ["test"]
            else:
                splits = ["test"]
            
            for split in splits:
                print(f"\n{'='*100}")
                print(f"{case_type.upper()} - {split.upper()} SET - SPECIFIED SIMULATIONS")
                print(f"{'='*100}")
                
                for sim_folder in simulation_folders:
                    sim_results_path = os.path.join(parent_results_path, sim_folder)
                    
                    case_folders = []
                    if os.path.exists(sim_results_path):
                        for item in os.listdir(sim_results_path):
                            item_path = os.path.join(sim_results_path, item)
                            if os.path.isdir(item_path) and item.lower() == case_type.lower():
                                case_folders.append(item)
                    
                    if not case_folders:
                        print(f"\n[{sim_folder.upper()}] - No {case_type.upper()} folder found")
                        continue
                    
                    case_folder = case_folders[0]
                    consolidated_dir = Path(f"{sim_results_path}/{case_folder}/consolidated")
                    
                    if not consolidated_dir.exists():
                        print(f"\n[{sim_folder.upper()}] - No consolidated directory found")
                        continue
                    
                    pico_files = list(consolidated_dir.glob(f"*consolidated_picos_{split}*.json"))
                    outcome_files = list(consolidated_dir.glob(f"*consolidated_outcomes_{split}*.json"))
                    
                    if pico_files and outcome_files:
                        pico_file = max(pico_files, key=os.path.getmtime)
                        outcome_file = max(outcome_files, key=os.path.getmtime)
                        
                        try:
                            pico_analyzer = PICOAnalyzer(str(pico_file))
                            outcome_analyzer = OutcomeAnalyzer(str(outcome_file))
                            
                            print(f"\n{'*'*100}")
                            print(f"SIMULATION: {sim_folder.upper()}")
                            print(f"{'*'*100}")
                            
                            pico_analyzer.print_unique_picos_overview()
                            outcome_analyzer.print_unique_outcomes_overview()
                            
                        except Exception as e:
                            print(f"\n[{sim_folder.upper()}] - Error analyzing: {e}")
                            import traceback
                            traceback.print_exc()
                    else:
                        print(f"\n[{sim_folder.upper()}] - No consolidated files found for {split} set")
    
    def run_all(self):
        """Run analysis based on mode and simulation IDs."""
        if self.mode == "consolidated_only":
            self.run_consolidated_analysis()
        else:
            self.run_translation_analysis()
            
            self.run_comprehensive_overview()
            
            print("\n" + "="*100)
            print("RUNNING INDIVIDUAL CASE ANALYSES")
            print("="*100)
            
            all_pico_analyzers = []
            all_outcome_analyzers = []
            
            case_folders = self._get_case_folders()
            
            for case_folder in case_folders:
                if case_folder.upper() == "NSCLC":
                    analyzers = self.run_case_analysis(case_folder, splits=["train", "test"])
                else:
                    analyzers = self.run_case_analysis(case_folder, splits=["test"])
                
                if analyzers:
                    for pico_analyzer, outcome_analyzer in analyzers:
                        all_pico_analyzers.append(pico_analyzer)
                        all_outcome_analyzers.append(outcome_analyzer)
            
            if all_pico_analyzers and all_outcome_analyzers:
                print("\n" + "="*100)
                print("GENERATING COMBINED VISUALIZATIONS")
                print("="*100)
                
                visualizer = DataVisualizer()
                
                visualizer.create_combined_european_map(all_pico_analyzers)
                visualizer.create_combined_venn_diagram(all_pico_analyzers)
                visualizer.create_combined_source_type_comparison(all_pico_analyzers, all_outcome_analyzers)
                visualizer.generate_summary_report(all_pico_analyzers, all_outcome_analyzers)
                
                print(f"\nAll combined visualizations saved to {visualizer.output_dir}/")
            
            self.print_split_summary()