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


class ComprehensiveOverview:
    """Class to generate comprehensive overview summaries for RAG pipeline analysis"""
    
    def __init__(self):
        self.all_cases_data = {}
    
    def generate_case_overview(self, pico_analyzer, outcome_analyzer, case_name):
        """Generate a nice formatted overview summary for PICOs and Outcomes for a specific case"""
        print("\n" + "="*100)
        print(f"{case_name.upper()} COMPREHENSIVE ANALYSIS OVERVIEW")
        print("="*100)
        
        # PICO Overview
        print("\n" + "🔬 PICO EVIDENCE OVERVIEW")
        print("-" * 50)
        
        case_data = {'picos': 0, 'outcomes': 0, 'pico_countries': set(), 'outcome_countries': set(), 'source_types': set()}
        
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
        
        # Outcomes Overview
        print("\n" + "🎯 OUTCOMES EVIDENCE OVERVIEW")
        print("-" * 50)
        
        if outcome_analyzer.total_outcomes > 0:
            case_data['outcomes'] = outcome_analyzer.total_outcomes
            
            print(f"📊 Total Outcome Measures: {outcome_analyzer.total_outcomes}")
            print(f"📂 Outcome Categories: {len(outcome_analyzer.data.get('consolidated_outcomes', {}))}")
            
            # Get countries from metadata instead of processed DataFrame
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
        
        # Coverage Summary
        print("\n" + "🗺️  COVERAGE SUMMARY")
        print("-" * 50)
        
        pico_countries = set()
        outcome_countries = set()
        
        if not pico_analyzer.picos_df.empty and 'Country' in pico_analyzer.picos_df.columns:
            pico_countries = set(pico_analyzer.picos_df['Country'].unique())
        
        # Get outcome countries from metadata
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
        
        # Store case data for cross-case analysis
        self.all_cases_data[case_name] = case_data
        
        return case_data
    
    def generate_cross_case_overview(self, all_pico_files, all_outcome_files):
        """Generate overview across all cases"""
        print("\n" + "🌍 CROSS-CASE ANALYSIS OVERVIEW")
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
        
        print(f"\n📊 TOTAL ACROSS ALL CASES:")
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
        self.load_data()
        self.prepare_datamatrix()
    
    def load_data(self):
        try:
            with open(self.pico_file_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            print(f"Successfully loaded PICO data from {self.pico_file_path}")
        except FileNotFoundError:
            raise FileNotFoundError(f"PICO file not found: {self.pico_file_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in PICO file: {e}")
        except Exception as e:
            raise Exception(f"Error loading PICO data: {e}")
    
    def prepare_datamatrix(self):
        pico_records = []
        
        try:
            # Check if data structure is as expected
            if 'consolidated_picos' not in self.data:
                print("Warning: 'consolidated_picos' key not found in data")
                self.picos_df = pd.DataFrame()
                return
                
            consolidated_picos = self.data['consolidated_picos']
            
            # Handle case where consolidated_picos might be a string or other type
            if not isinstance(consolidated_picos, list):
                print(f"Warning: consolidated_picos is not a list, it's a {type(consolidated_picos)}")
                self.picos_df = pd.DataFrame()
                return
            
            for i, pico in enumerate(consolidated_picos):
                try:
                    # Ensure pico is a dictionary
                    if not isinstance(pico, dict):
                        print(f"Warning: PICO item {i} is not a dictionary, it's a {type(pico)}")
                        continue
                    
                    # Extract countries with fallback handling
                    countries = pico.get('Countries', [])
                    if isinstance(countries, str):
                        countries = [countries]
                    elif not isinstance(countries, list):
                        countries = []
                    
                    # Extract source types with fallback handling
                    source_types = pico.get('Source_Types', [])
                    if isinstance(source_types, str):
                        source_types = [source_types]
                    elif not isinstance(source_types, list):
                        source_types = []
                    
                    # Extract other fields with safe defaults
                    population = pico.get('Population', 'Unknown')
                    intervention = pico.get('Intervention', 'Unknown')
                    comparator = pico.get('Comparator', 'Unknown')
                    
                    # Handle population and comparator variants
                    pop_variants = pico.get('Original_Population_Variants', [])
                    comp_variants = pico.get('Original_Comparator_Variants', [])
                    
                    if not isinstance(pop_variants, list):
                        pop_variants = []
                    if not isinstance(comp_variants, list):
                        comp_variants = []
                    
                    # Create records for each country-source combination
                    for country in countries:
                        for source_type in source_types:
                            record = {
                                'Population': population,
                                'Intervention': intervention,
                                'Comparator': comparator,
                                'Country': country,
                                'Source_Type': source_type,
                                'Population_Variants_Count': len(pop_variants),
                                'Comparator_Variants_Count': len(comp_variants)
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
        print("\n" + "🔬 DETAILED PICO EVIDENCE LISTING")
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
                
                # Ensure countries and source_types are lists
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
                
                # Show population and comparator variants if available
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
        print("="*80)
        print("PICO ANALYSIS SUMMARY")
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
        self.load_data()
        self.prepare_datamatrix()
        
        # Use metadata value if available as it might be deduplicated
        metadata = self.data.get('outcomes_metadata', {})
        if 'total_unique_outcomes' in metadata:
            self.total_outcomes = metadata['total_unique_outcomes']
            print(f"Using metadata total_unique_outcomes: {self.total_outcomes}")
        else:
            print(f"Using calculated total_outcomes: {self.total_outcomes}")
    
    def load_data(self):
        try:
            with open(self.outcome_file_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            print(f"Successfully loaded Outcomes data from {self.outcome_file_path}")
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
            # Check if data structure is as expected
            if 'consolidated_outcomes' not in self.data:
                print("Warning: 'consolidated_outcomes' key not found in data")
                self.outcomes_df = pd.DataFrame()
                return
                
            consolidated_outcomes = self.data['consolidated_outcomes']
            
            # Handle case where consolidated_outcomes might not be a dictionary
            if not isinstance(consolidated_outcomes, dict):
                print(f"Warning: consolidated_outcomes is not a dictionary, it's a {type(consolidated_outcomes)}")
                self.outcomes_df = pd.DataFrame()
                return
            
            # Get metadata for country and source information
            metadata = self.data.get('outcomes_metadata', {})
            source_countries = metadata.get('source_countries', [])
            source_types = metadata.get('source_types', [])
            
            for category, subcategories in consolidated_outcomes.items():
                try:
                    # Ensure subcategories is a dictionary
                    if not isinstance(subcategories, dict):
                        print(f"Warning: subcategories for {category} is not a dictionary, it's a {type(subcategories)}")
                        continue
                    
                    for subcategory, outcomes in subcategories.items():
                        try:
                            # Ensure outcomes is a list
                            if not isinstance(outcomes, list):
                                print(f"Warning: outcomes for {category}/{subcategory} is not a list, it's a {type(outcomes)}")
                                continue
                            
                            total_outcomes += len(outcomes)
                            
                            for outcome in outcomes:
                                try:
                                    # Handle both string and dictionary outcomes
                                    if isinstance(outcome, str):
                                        outcome_name = outcome
                                        
                                        # Create records for each country-source combination from metadata
                                        for country in source_countries:
                                            for source_type in source_types:
                                                record = {
                                                    'Category': category,
                                                    'Subcategory': subcategory,
                                                    'Outcome_Name': outcome_name,
                                                    'Country': country,
                                                    'Source_Type': source_type,
                                                    'Has_Details': False
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
                                                        'Has_Details': has_details
                                                    }
                                                    outcome_records.append(record)
                                        else:
                                            # Use metadata for countries and sources
                                            for country in source_countries:
                                                for source_type in source_types:
                                                    record = {
                                                        'Category': category,
                                                        'Subcategory': subcategory,
                                                        'Outcome_Name': outcome_name,
                                                        'Country': country,
                                                        'Source_Type': source_type,
                                                        'Has_Details': has_details
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
        print("\n" + "🎯 DETAILED OUTCOMES EVIDENCE LISTING")
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
        
        # Show metadata information
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
                                    # Simple string outcome
                                    outcome_name = outcome
                                    print(f"  {i:2d}. {outcome_name}")
                                    
                                elif isinstance(outcome, dict):
                                    # Complex outcome object
                                    outcome_name = outcome.get('name', 'Unnamed outcome')
                                    has_details = 'details' in outcome and bool(outcome.get('details', []))
                                    reported_by = outcome.get('reported_by', [])
                                    
                                    # Count unique countries and source types
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
        print("="*80)
        print("OUTCOMES ANALYSIS SUMMARY")
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
                # Show category breakdown from the raw data
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

    def create_comparator_venn_diagram(self):
        """Create Venn diagram showing overlap of comparators between guidelines and HTA submissions."""
        if not VENN_AVAILABLE:
            print("Matplotlib-venn not available. Skipping Venn diagram.")
            return
            
        print("Creating comparator Venn diagram...")
        
        guideline_comparators, hta_comparators = self.get_comparators_by_source_type()
        
        # Remove empty comparators
        guideline_comparators.discard('')
        hta_comparators.discard('')
        
        if not guideline_comparators and not hta_comparators:
            print("No comparator data available for Venn diagram")
            return
            
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create the Venn diagram
        venn = venn2([guideline_comparators, hta_comparators], 
                     set_labels=('Clinical Guidelines', 'HTA Submissions'),
                     ax=ax)
        
        # Customize colors
        if venn.get_patch_by_id('10'):
            venn.get_patch_by_id('10').set_color(self.scientific_colors['secondary'])
            venn.get_patch_by_id('10').set_alpha(0.7)
        if venn.get_patch_by_id('01'):
            venn.get_patch_by_id('01').set_color(self.scientific_colors['tertiary'])
            venn.get_patch_by_id('01').set_alpha(0.7)
        if venn.get_patch_by_id('11'):
            venn.get_patch_by_id('11').set_color(self.scientific_colors['primary'])
            venn.get_patch_by_id('11').set_alpha(0.8)
        
        # Add circles for better visibility
        venn2_circles([guideline_comparators, hta_comparators], ax=ax, linewidth=2)
        
        # Calculate overlap statistics
        overlap = guideline_comparators.intersection(hta_comparators)
        guideline_only = guideline_comparators - hta_comparators
        hta_only = hta_comparators - guideline_comparators
        
        ax.set_title(f'Comparator Overlap: Guidelines vs HTA Submissions\n'
                    f'Total Unique Comparators: {len(guideline_comparators.union(hta_comparators))}',
                    fontsize=14, fontweight='bold', pad=20)
        
        # Add statistics text box
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
        
        # Print detailed breakdown
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
            
        # Count comparators per country
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
            
        # Prepare data for plotting
        countries = list(country_comparators.keys())
        comparator_counts = [len(country_comparators[country]) for country in countries]
        
        # Determine colors based on source types
        colors = []
        for country in countries:
            sources = country_source_info[country]
            if len(sources) > 1:
                colors.append(self.scientific_colors['primary'])  # Both sources
            elif 'clinical_guideline' in sources:
                colors.append(self.scientific_colors['secondary'])  # Guidelines only
            else:
                colors.append(self.scientific_colors['tertiary'])  # HTA only
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        bars = ax.bar(range(len(countries)), comparator_counts, 
                     color=colors, alpha=0.8, edgecolor='white', linewidth=1)
        
        ax.set_xticks(range(len(countries)))
        ax.set_xticklabels(countries, fontweight='bold')
        ax.set_ylabel('Number of Distinct Comparators', fontweight='bold')
        ax.set_xlabel('Country', fontweight='bold')
        ax.set_title('Comparator Breadth by Country\n(Distinct Comparators Considered in PICO Evidence)', 
                    fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, comparator_counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   str(value), ha='center', va='bottom', fontweight='bold')
        
        # Create legend
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
            
        # Collect data by country
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
        
        # Create table
        table_data = []
        for country in sorted(country_data.keys()):
            data = country_data[country]
            
            # Get most common population description (first one if multiple)
            population = list(data['population'])[0] if data['population'] else 'Not specified'
            if len(population) > 80:
                population = population[:77] + "..."
                
            intervention = list(data['intervention'])[0] if data['intervention'] else 'Not specified'
            
            guideline_comps = ', '.join(sorted(data['guideline_comparators'])) if data['guideline_comparators'] else '-'
            hta_comps = ', '.join(sorted(data['hta_comparators'])) if data['hta_comparators'] else '-'
            
            # Truncate long comparator lists
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
        
        # Create DataFrame for better formatting
        df = pd.DataFrame(table_data, columns=[
            'Country', 'Population', 'Intervention', 
            'Guideline Comparators', 'HTA Comparators', 'Source Types'
        ])
        
        # Save as CSV
        csv_path = self.output_dir / 'country_pico_summary_table.csv'
        df.to_csv(csv_path, index=False)
        print(f"PICO summary table saved to: {csv_path}")
        
        # Create figure for display
        fig, ax = plt.subplots(figsize=(16, max(8, len(table_data) * 0.5)))
        ax.axis('tight')
        ax.axis('off')
        
        # Create table
        table = ax.table(cellText=table_data,
                        colLabels=['Country', 'Population Description', 'Intervention', 
                                 'Guideline Comparators', 'HTA Comparators', 'Source Types'],
                        cellLoc='left',
                        loc='center',
                        bbox=[0, 0, 1, 1])
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 2)
        
        # Header styling
        for i in range(len(df.columns)):
            table[(0, i)].set_facecolor(self.scientific_colors['primary'])
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Alternate row colors
        for i in range(1, len(table_data) + 1):
            for j in range(len(df.columns)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#F8F9FA')
                else:
                    table[(i, j)].set_facecolor('white')
        
        plt.title('Country-by-Country PICO Summary\nComparative Analysis of Evidence Requirements', 
                 fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'country_pico_summary_table.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        # Print summary statistics
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
        
        # Prepare data for treemap
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
        
        # Create treemap
        squarify.plot(sizes=sizes, label=labels, color=colors, alpha=0.8,
                     text_kwargs={'fontsize': 9, 'weight': 'bold'}, ax=ax)
        
        ax.set_title('Outcomes Evidence Hierarchy\nTreemap by Category and Subcategory', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.axis('off')
        
        # Create legend
        legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, alpha=0.8, label=cat.title()) 
                          for cat, color in color_map.items() if cat in [c for c, _ in consolidated_outcomes.items()]]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1), 
                 frameon=True, fancybox=True, shadow=True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'outcomes_treemap.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()

    def _create_alternative_outcomes_hierarchy(self):
        """Create alternative hierarchical visualization when squarify is not available."""
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
        
        categories = [item[0] for item in category_data]
        counts = [item[1] for item in category_data]
        
        colors = [self.scientific_colors['primary'], self.scientific_colors['quaternary'],
                 self.scientific_colors['secondary'], self.scientific_colors['tertiary'],
                 self.scientific_colors['dark_gray']][:len(categories)]
        
        wedges, texts, autotexts = ax.pie(counts, labels=[cat.replace('_', ' ').title() for cat in categories],
                                         autopct='%1.1f%%', colors=colors, startangle=90,
                                         wedgeprops=dict(edgecolor='white', linewidth=2))
        
        ax.set_title('Outcomes Distribution by Category\n(Alternative Hierarchy View)', 
                    fontweight='bold', pad=20)
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
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
        
        # Find original PICO files
        case_name = "NSCLC"  # Extract from path if needed
        if "/HCC/" in self.pico_analyzer.pico_file_path:
            case_name = "HCC"
        
        pico_dir = Path(self.pico_analyzer.pico_file_path).parent
        
        # Load individual PICO files
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
        
        # Calculate individual PICO counts
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
        
        # Get consolidated count
        consolidated_count = len(self.pico_analyzer.data.get('consolidated_picos', []))
        
        print(f"Individual PICO counts: {individual_counts}")
        print(f"Total individual PICOs: {total_individual}")
        print(f"Consolidated PICOs: {consolidated_count}")
        
        if total_individual == 0:
            print("No individual PICO data found. Cannot create Sankey diagram.")
            return
        
        # Create Sankey data
        source_labels = []
        target_labels = []
        values = []
        
        # Individual sources to consolidated
        for source_type, count in individual_counts.items():
            if count > 0:  # Only include sources that have data
                source_labels.append(f"{source_type.replace('_', ' ').title()}")
                target_labels.append("Consolidated PICOs")
                values.append(count)
        
        if not values:
            print("No valid source data found for Sankey diagram")
            return
        
        # All labels
        all_labels = list(set(source_labels + target_labels))
        
        # Create source and target indices
        source_indices = [all_labels.index(label) for label in source_labels]
        target_indices = [all_labels.index(label) for label in target_labels]
        
        # Color mapping
        colors = [
            'rgba(4, 138, 129, 0.8)',  # Clinical guidelines
            'rgba(241, 143, 1, 0.8)',  # HTA submissions
            'rgba(46, 64, 87, 0.8)'    # Consolidated
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
        
        fig.update_layout(
            title_text=f"{case_name} PICO Consolidation Process Flow<br>"
                      f"<sub>From {total_individual} Individual PICOs to {consolidated_count} Consolidated PICOs</sub>",
            font_size=12,
            font_family="Arial",
            width=800,
            height=500
        )
        
        # Save as HTML
        html_path = self.output_dir / f'{case_name.lower()}_pico_consolidation_sankey.html'
        fig.write_html(str(html_path))
        
        # Save as static image
        try:
            png_path = self.output_dir / f'{case_name.lower()}_pico_consolidation_sankey.png'
            fig.write_image(str(png_path), width=800, height=500, scale=2)
            print(f"Sankey diagram saved to: {png_path}")
        except Exception as e:
            print(f"Could not save PNG (install kaleido for static export): {e}")
        
        fig.show()
        print(f"Interactive Sankey diagram saved to: {html_path}")
        
        # Print consolidation statistics
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
        
        # Extract case name from path more robustly (same logic as main method)
        file_path = Path(self.pico_analyzer.pico_file_path)
        print(f"DEBUG ALT: Input file path: {file_path}")
        print(f"DEBUG ALT: Path parts: {file_path.parts}")
        
        case_name = None
        
        # Look for case directory in path
        for part in file_path.parts:
            if part.upper() in ['NSCLC', 'HCC', 'SCLC', 'BREAST', 'LUNG']:
                case_name = part.upper()
                break
        
        if not case_name:
            case_name = "UNKNOWN"
            print(f"DEBUG ALT: Could not detect case name from path: {file_path}")
        
        print(f"DEBUG ALT: Detected case name: {case_name}")
        
        # Navigate to the PICO directory
        case_root_dir = file_path.parent.parent
        pico_dir = case_root_dir / "PICO"
        
        print(f"DEBUG ALT: Case root dir: {case_root_dir}")
        print(f"DEBUG ALT: PICO dir: {pico_dir}")
        print(f"DEBUG ALT: PICO dir exists: {pico_dir.exists()}")
        
        # Load individual PICO files and count
        individual_counts = {}
        total_individual = 0
        for source_type in ['clinical_guideline', 'hta_submission']:
            pico_file = pico_dir / f"{source_type}_picos.json"
            print(f"DEBUG ALT: Checking for: {pico_file}")
            print(f"DEBUG ALT: File exists: {pico_file.exists()}")
            
            if pico_file.exists():
                try:
                    with open(pico_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        count = 0
                        
                        # Try metadata first, then fallback to country counting
                        metadata = data.get('extraction_metadata', {})
                        print(f"DEBUG ALT: {source_type} metadata: {metadata}")
                        
                        if 'total_picos' in metadata:
                            count = metadata['total_picos']
                            print(f"DEBUG ALT: {source_type} count from metadata: {count}")
                        elif 'picos_by_country' in data:
                            print(f"DEBUG ALT: Using fallback counting for {source_type}")
                            for country, country_data in data['picos_by_country'].items():
                                if 'PICOs' in country_data:
                                    country_count = len(country_data['PICOs'])
                                    count += country_count
                                    print(f"DEBUG ALT: {country}: {country_count} PICOs")
                            print(f"DEBUG ALT: {source_type} total count from fallback: {count}")
                        
                        individual_counts[source_type] = count
                        total_individual += count
                        
                except Exception as e:
                    print(f"ERROR ALT: Failed to load {source_type} PICOs: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"WARNING ALT: File not found: {pico_file}")
        
        print(f"DEBUG ALT: Final individual_counts: {individual_counts}")
        print(f"DEBUG ALT: Total individual: {total_individual}")
        
        consolidated_count = len(self.pico_analyzer.data.get('consolidated_picos', []))
        print(f"DEBUG ALT: Consolidated count: {consolidated_count}")
        
        if total_individual == 0:
            print("ERROR ALT: No individual PICO data found for alternative visualization")
            return
        
        # Create flow diagram
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Define positions
        y_positions = [0.7, 0.3]  # For two source types
        x_individual = 0.2
        x_consolidated = 0.8
        
        # Draw individual sources
        colors = [self.scientific_colors['secondary'], self.scientific_colors['tertiary']]
        for i, (source_type, count) in enumerate(individual_counts.items()):
            if count == 0:
                continue
                
            # Source box
            box_height = 0.15
            box_width = 0.2
            y_pos = y_positions[i] if i < len(y_positions) else 0.5
            
            rect = plt.Rectangle((x_individual - box_width/2, y_pos - box_height/2), 
                               box_width, box_height,
                               facecolor=colors[i] if i < len(colors) else self.scientific_colors['dark_gray'],
                               alpha=0.8, edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            
            # Label
            ax.text(x_individual, y_pos, f"{source_type.replace('_', ' ').title()}\n{count} PICOs",
                   ha='center', va='center', fontweight='bold', fontsize=10)
            
            # Arrow to consolidated
            ax.annotate('', xy=(x_consolidated - 0.12, 0.5), xytext=(x_individual + 0.1, y_pos),
                       arrowprops=dict(arrowstyle='->', lw=3, 
                                     color=colors[i] if i < len(colors) else self.scientific_colors['dark_gray'],
                                     alpha=0.7))
        
        # Consolidated box
        box_height = 0.2
        box_width = 0.25
        rect = plt.Rectangle((x_consolidated - box_width/2, 0.5 - box_height/2), 
                           box_width, box_height,
                           facecolor=self.scientific_colors['primary'],
                           alpha=0.8, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        
        ax.text(x_consolidated, 0.5, f"Consolidated\nPICOs\n{consolidated_count}",
               ha='center', va='center', fontweight='bold', fontsize=12, color='white')
        
        # Add statistics
        reduction_pct = ((total_individual - consolidated_count) / total_individual * 100) if total_individual > 0 else 0
        stats_text = (f"Total Individual: {total_individual}\n"
                     f"Consolidated: {consolidated_count}\n"
                     f"Reduction: {reduction_pct:.1f}%")
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
               fontsize=10, fontweight='bold')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f'{case_name} PICO Consolidation Process\nFrom Individual Sources to Consolidated Evidence', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{case_name.lower()}_pico_consolidation_flow.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        # Print consolidation statistics
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
        
        # Print country information before plotting
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
                
                # Use a better colormap that doesn't include white for data countries
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
                
                # Annotate all countries with PICOs
                for idx, row in europe_with_data.iterrows():
                    try:
                        centroid = row.geometry.centroid
                        
                        # Manual positioning for France to fix location issue
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
            
            ax.set_title(f'European Distribution of PICO Evidence\nby Country ({len(country_counts)} countries)', 
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
        
        # Use YlOrRd colormap to avoid white for countries with data
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
        ax.set_title(f'European Distribution of PICO Evidence\n({len(country_counts)} countries - Simplified View)', 
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
        fig.suptitle('PICO Analysis Overview', fontsize=16, fontweight='bold', y=0.95)
        
        # Chart 1: Country distribution
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
        
        # Chart 2: Source type distribution
        if 'Source_Type' in self.pico_analyzer.picos_df.columns:
            source_counts = self.pico_analyzer.picos_df['Source_Type'].value_counts()
            colors = [self.scientific_colors['secondary'], self.scientific_colors['tertiary']][:len(source_counts)]
            wedges, texts, autotexts = axes[0, 1].pie(source_counts.values, 
                                                     labels=[label.replace('_', ' ').title() for label in source_counts.index],
                                                     autopct='%1.1f%%', 
                                                     colors=colors,
                                                     startangle=90,
                                                     wedgeprops=dict(edgecolor='white', linewidth=2))
            axes[0, 1].set_title('B. PICO Distribution by Source Type', fontweight='bold', pad=15)
            
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
        else:
            axes[0, 1].text(0.5, 0.5, 'No Source Type Data Available', 
                           ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('B. PICO Distribution by Source Type', fontweight='bold', pad=15)
        
        # Chart 3: Comparator frequency
        if 'Comparator' in self.pico_analyzer.picos_df.columns:
            comp_counts = self.pico_analyzer.picos_df['Comparator'].value_counts().head(8)
            bars2 = axes[1, 0].barh(range(len(comp_counts)), comp_counts.values, 
                                   color=self.scientific_colors['quaternary'], alpha=0.8,
                                   edgecolor='white', linewidth=0.8)
            axes[1, 0].set_yticks(range(len(comp_counts)))
            axes[1, 0].set_yticklabels([comp[:25] + '...' if len(comp) > 25 else comp 
                                       for comp in comp_counts.index], fontweight='bold')
            axes[1, 0].set_xlabel('Number of PICOs', fontweight='bold')
            axes[1, 0].set_title('C. Most Frequent Comparators', fontweight='bold', pad=15)
            axes[1, 0].grid(True, alpha=0.3, axis='x')
            axes[1, 0].invert_yaxis()
            
            for bar, value in zip(bars2, comp_counts.values):
                axes[1, 0].text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                               str(value), ha='left', va='center', fontweight='bold')
        else:
            axes[1, 0].text(0.5, 0.5, 'No Comparator Data Available', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('C. Most Frequent Comparators', fontweight='bold', pad=15)
        
        # Chart 4: Country vs Source Type matrix
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
        
        # Create new enhanced visualizations
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
        
        plt.title('Country vs Comparator Distribution Matrix', 
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
        fig.suptitle('Outcomes Analysis Overview', fontsize=16, fontweight='bold', y=0.95)
        
        # For visualizations, we'll use the consolidated outcomes structure directly
        consolidated_outcomes = self.outcome_analyzer.data.get('consolidated_outcomes', {})
        metadata = self.outcome_analyzer.data.get('outcomes_metadata', {})
        
        # Chart 1: Category distribution
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
        
        # Chart 2: Country distribution (using metadata)
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

        # Chart 3: Source type distribution - show actual representation
        # Since outcomes aren't source-specific in this structure, we can show the coverage
        # by using the PICO data which shows which source types contributed to the evidence
        if not self.pico_analyzer.picos_df.empty and 'Source_Type' in self.pico_analyzer.picos_df.columns:
            pico_source_counts = self.pico_analyzer.picos_df['Source_Type'].value_counts()
            # For outcomes visualization, show how many outcome measures are available from each source type
            source_outcome_counts = {source: self.outcome_analyzer.total_outcomes for source in pico_source_counts.index}
            
            if source_outcome_counts:
                colors = [self.scientific_colors['tertiary'], self.scientific_colors['quaternary']][:len(source_outcome_counts)]
                wedges, texts, autotexts = axes[1, 0].pie(list(source_outcome_counts.values()),
                                                         labels=[label.replace('_', ' ').title() for label in source_outcome_counts.keys()],
                                                         autopct='%1.1f%%', 
                                                         colors=colors,
                                                         startangle=90,
                                                         wedgeprops=dict(edgecolor='white', linewidth=2))
                axes[1, 0].set_title('C. Source Type Coverage for Outcomes', fontweight='bold', pad=15)
                
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
            else:
                axes[1, 0].text(0.5, 0.5, 'No Source Type Data Available', 
                               ha='center', va='center', transform=axes[1, 0].transAxes)
                axes[1, 0].set_title('C. Source Type Coverage for Outcomes', fontweight='bold', pad=15)
        else:
            axes[1, 0].text(0.5, 0.5, 'No Source Type Data Available', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('C. Source Type Coverage for Outcomes', fontweight='bold', pad=15)
        
        # Chart 4: Subcategory breakdown
        subcategory_counts = {}
        for category, subcategories in consolidated_outcomes.items():
            if isinstance(subcategories, dict):
                for subcategory, outcomes in subcategories.items():
                    if isinstance(outcomes, list):
                        subcategory_counts[f"{category}_{subcategory}"] = len(outcomes)
        
        if subcategory_counts:
            # Show top 10 subcategories
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
        
        # Create enhanced outcomes visualizations
        self.create_outcomes_treemap()
    
    def create_combined_analysis(self):
        print("Creating combined analysis visualization...")
        
        # Check if we have data for both analyzers
        pico_has_source = not self.pico_analyzer.picos_df.empty and 'Source_Type' in self.pico_analyzer.picos_df.columns
        outcome_has_source = self.outcome_analyzer.total_outcomes > 0
        
        if not pico_has_source and not outcome_has_source:
            print("No source type data available for combined analysis")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Source Type Distribution Comparison', fontsize=16, fontweight='bold', y=0.98)
        
        # PICO sources pie chart
        if pico_has_source:
            pico_sources = self.pico_analyzer.picos_df['Source_Type'].value_counts()
            colors_pico = [self.scientific_colors['primary'], self.scientific_colors['secondary']][:len(pico_sources)]
            wedges1, texts1, autotexts1 = axes[0].pie(pico_sources.values, 
                                                      labels=[label.replace('_', ' ').title() 
                                                             for label in pico_sources.index],
                                                      autopct='%1.1f%%', 
                                                      colors=colors_pico,
                                                      startangle=90,
                                                      wedgeprops=dict(edgecolor='white', linewidth=2))
            axes[0].set_title('A. PICOs by Source Type', fontweight='bold', pad=20)
            
            for autotext in autotexts1:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
        else:
            axes[0].text(0.5, 0.5, 'No PICO Source Type Data', 
                        ha='center', va='center', transform=axes[0].transAxes)
            axes[0].set_title('A. PICOs by Source Type', fontweight='bold', pad=20)
        
        # Outcome sources pie chart
        if outcome_has_source:
            metadata = self.outcome_analyzer.data.get('outcomes_metadata', {})
            source_types = metadata.get('source_types', [])
            
            if source_types:
                colors_outcome = [self.scientific_colors['tertiary'], self.scientific_colors['quaternary']][:len(source_types)]
                wedges2, texts2, autotexts2 = axes[1].pie([1]*len(source_types),  # Equal weight for visualization
                                                          labels=[label.replace('_', ' ').title() for label in source_types],
                                                          autopct='%1.1f%%', 
                                                          colors=colors_outcome,
                                                          startangle=90,
                                                          wedgeprops=dict(edgecolor='white', linewidth=2))
                axes[1].set_title('B. Outcomes by Source Type', fontweight='bold', pad=20)
                
                for autotext in autotexts2:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
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
        report_content.append("RAG PIPELINE ANALYSIS SUMMARY REPORT")
        report_content.append("=" * 50)
        report_content.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
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
        
        # Coverage analysis
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
        
        with open(self.output_dir / 'analysis_summary_report.txt', 'w') as f:
            f.write('\n'.join(report_content))
        
        print('\n'.join(report_content))


def generate_overview_summary(pico_analyzer, outcome_analyzer, case_name):
    """Generate a nice formatted overview summary for PICOs and Outcomes"""
    print("\n" + "="*100)
    print(f"{case_name.upper()} COMPREHENSIVE ANALYSIS OVERVIEW")
    print("="*100)
    
    # PICO Overview
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
    
    # Outcomes Overview
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
    
    # Coverage Summary
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
    print(f"PICO-Only Countries: {len(pico_countries - outcome_countries)}")
    print(f"🎯 Outcome-Only Countries: {len(outcome_countries - pico_countries)}")
    
    if common_countries:
        print(f"   Countries with complete coverage: {', '.join(sorted(common_countries))}")
    
    print("\n" + "="*100)


def run_complete_analysis(pico_file_path, outcome_file_path):
    print("Starting comprehensive RAG pipeline analysis...")
    print()
    
    try:
        pico_analyzer = PICOAnalyzer(pico_file_path)
        outcome_analyzer = OutcomeAnalyzer(outcome_file_path)
        
        # Extract case name from file path for overview
        case_name = "Analysis"
        if "/NSCLC/" in pico_file_path or "/nsclc/" in pico_file_path:
            case_name = "NSCLC"
        elif "/HCC/" in pico_file_path or "/hcc/" in pico_file_path:
            case_name = "HCC"
        
        # Generate overview summary using the new class
        overview = ComprehensiveOverview()
        overview.generate_case_overview(pico_analyzer, outcome_analyzer, case_name)
        
        # Print detailed unique PICOs and outcomes before visualizations
        pico_analyzer.print_unique_picos_overview()
        outcome_analyzer.print_unique_outcomes_overview()
        
        pico_analyzer.print_summary_statistics()
        outcome_analyzer.print_summary_statistics()
        
        visualizer = DataVisualizer(pico_analyzer, outcome_analyzer)
        
        visualizer.create_pico_visualizations()
        visualizer.create_outcome_visualizations()
        visualizer.create_combined_analysis()
        
        visualizer.generate_summary_report()
        
        print("Analysis complete! All visualizations and reports saved to results/visualizations/")
        
        return pico_analyzer, outcome_analyzer, visualizer
        
    except Exception as e:
        print(f"Error in complete analysis: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None