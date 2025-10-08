import torch
import pandas as pd
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from collections import defaultdict
import os

# Import from your existing code
from classify_embeddings import (
    Config, load_trained_model, classify_embeddings, 
    add_time_component, validate_lorentz_embedding
)


class DatasetTester:
    """Comprehensive testing class for hyperbolic MLR classification with robust CSV handling"""
    
    def __init__(self, state_dict_path, output_dir="classification_results"):
        self.state_dict_path = state_dict_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load models
        print("Loading models...")
        self.hysac_model, self.tokenizer, self.lorentz_mlr = load_trained_model(state_dict_path)
        
        # Results storage
        self.results = {}
        self.dataset_stats = {}
        
        print(f"Models loaded successfully. Output directory: {self.output_dir}")
    
    def robust_csv_read(self, csv_path):
        """Robustly read CSV files with various formatting issues"""
        print(f"Attempting to read CSV: {csv_path}")
        
        # Try different approaches to read the CSV
        approaches = [
            # Standard approach
            lambda: pd.read_csv(csv_path),
            
            # With error_bad_lines=False (older pandas)
            lambda: pd.read_csv(csv_path, on_bad_lines='skip') if hasattr(pd.read_csv, '__code__') and 'on_bad_lines' in pd.read_csv.__code__.co_varnames else pd.read_csv(csv_path, error_bad_lines=False),
            
            # With different separator
            lambda: pd.read_csv(csv_path, sep=',', quotechar='"', escapechar='\\'),
            
            # With manual error handling
            lambda: pd.read_csv(csv_path, on_bad_lines='skip', quoting=csv.QUOTE_MINIMAL),
            
            # Read with python engine
            lambda: pd.read_csv(csv_path, engine='python', on_bad_lines='skip'),
            
            # Last resort: read line by line
            lambda: self.read_csv_line_by_line(csv_path)
        ]
        
        for i, approach in enumerate(approaches):
            try:
                print(f"  Trying approach {i+1}...")
                df = approach()
                print(f"  ✓ Success with approach {i+1}! Shape: {df.shape}")
                return df
            except Exception as e:
                print(f"  ✗ Approach {i+1} failed: {str(e)[:100]}...")
                continue
        
        raise Exception("All CSV reading approaches failed")
    
    def read_csv_line_by_line(self, csv_path):
        """Manual line-by-line CSV reading as last resort"""
        print("  Using manual line-by-line reading...")
        data = []
        headers = None
        skipped_lines = 0
        
        with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.reader(f, quoting=csv.QUOTE_MINIMAL)
            
            for line_num, row in enumerate(reader):
                try:
                    if line_num == 0:
                        headers = row
                        continue
                    
                    # If row has different number of fields, try to fix it
                    if len(row) != len(headers):
                        if len(row) > len(headers):
                            # Too many fields - join extra ones
                            fixed_row = row[:len(headers)-1] + [' '.join(row[len(headers)-1:])]
                            row = fixed_row
                        else:
                            # Too few fields - pad with empty strings
                            row = row + [''] * (len(headers) - len(row))
                    
                    data.append(row)
                    
                except Exception as e:
                    skipped_lines += 1
                    print(f"    Skipped line {line_num}: {str(e)[:50]}...")
                    continue
        
        print(f"  Manual reading complete. Skipped {skipped_lines} problematic lines.")
        return pd.DataFrame(data, columns=headers)
    
    def analyze_dataset(self, csv_path, dataset_name):
        """Analyze dataset characteristics and provide summary with robust CSV handling"""
        print(f"\n{'='*60}")
        print(f"ANALYZING DATASET: {dataset_name}")
        print(f"{'='*60}")
        
        try:
            # Use robust CSV reading
            df = self.robust_csv_read(csv_path)
            
            # Basic statistics
            total_prompts = len(df)
            prompt_column = None
            
            # Print available columns for debugging
            print(f"Available columns: {df.columns.tolist()}")
            
            # Find the prompt column (could be 'prompt', 'text', 'caption', etc.)
            possible_columns = ['prompt', 'text', 'caption', 'description', 'Prompt', 'Text', 'Caption']
            for col in possible_columns:
                if col in df.columns:
                    prompt_column = col
                    break
            
            if prompt_column is None:
                # If no standard column found, try to find any text-like column
                for col in df.columns:
                    if df[col].dtype == 'object':  # Text columns are usually object type
                        # Check if this column contains text-like data
                        sample_values = df[col].dropna().head(5)
                        if len(sample_values) > 0:
                            avg_length = sample_values.astype(str).str.len().mean()
                            if avg_length > 10:  # Likely a text column
                                prompt_column = col
                                print(f"Auto-detected text column: '{col}'")
                                break
            
            if prompt_column is None:
                print(f"Error: No suitable prompt column found. Available columns: {df.columns.tolist()}")
                # Show first few rows for debugging
                print("First few rows:")
                print(df.head(3))
                return None
            
            # Clean and filter prompts
            prompts = df[prompt_column].dropna()
            prompts = prompts.astype(str)  # Ensure all are strings
            prompts = prompts[prompts.str.len() > 0]  # Remove empty strings
            valid_prompts = len(prompts)
            
            # Text statistics
            prompt_lengths = prompts.str.len()
            word_counts = prompts.str.split().str.len()
            
            stats = {
                'dataset_name': dataset_name,
                'csv_path': str(csv_path),
                'total_rows': total_prompts,
                'valid_prompts': valid_prompts,
                'prompt_column': prompt_column,
                'length_stats': {
                    'min_length': int(prompt_lengths.min()) if len(prompt_lengths) > 0 else 0,
                    'max_length': int(prompt_lengths.max()) if len(prompt_lengths) > 0 else 0,
                    'mean_length': float(prompt_lengths.mean()) if len(prompt_lengths) > 0 else 0,
                    'median_length': float(prompt_lengths.median()) if len(prompt_lengths) > 0 else 0,
                    'std_length': float(prompt_lengths.std()) if len(prompt_lengths) > 0 else 0
                },
                'word_count_stats': {
                    'min_words': int(word_counts.min()) if len(word_counts) > 0 else 0,
                    'max_words': int(word_counts.max()) if len(word_counts) > 0 else 0,
                    'mean_words': float(word_counts.mean()) if len(word_counts) > 0 else 0,
                    'median_words': float(word_counts.median()) if len(word_counts) > 0 else 0,
                    'std_words': float(word_counts.std()) if len(word_counts) > 0 else 0
                }
            }
            
            # Print summary
            print(f"Dataset: {dataset_name}")
            print(f"Total rows: {total_prompts}")
            print(f"Valid prompts: {valid_prompts}")
            print(f"Prompt column: '{prompt_column}'")
            
            if valid_prompts > 0:
                print(f"\nText Length Statistics:")
                print(f"  Mean: {stats['length_stats']['mean_length']:.1f} characters")
                print(f"  Median: {stats['length_stats']['median_length']:.1f} characters")
                print(f"  Range: {stats['length_stats']['min_length']} - {stats['length_stats']['max_length']} characters")
                print(f"\nWord Count Statistics:")
                print(f"  Mean: {stats['word_count_stats']['mean_words']:.1f} words")
                print(f"  Median: {stats['word_count_stats']['median_words']:.1f} words")
                print(f"  Range: {stats['word_count_stats']['min_words']} - {stats['word_count_stats']['max_words']} words")
                
                # Sample prompts
                print(f"\nSample prompts:")
                for i, prompt in enumerate(prompts.head(3)):
                    print(f"  {i+1}. {str(prompt)[:100]}{'...' if len(str(prompt)) > 100 else ''}")
            
            self.dataset_stats[dataset_name] = stats
            return prompts.tolist()
            
        except Exception as e:
            print(f"Error analyzing dataset {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def test_dataset(self, csv_path, dataset_name, expected_label=None):
        """Test classification on a dataset - processes ALL samples"""
        print(f"\n{'='*60}")
        print(f"TESTING CLASSIFICATION: {dataset_name}")
        print(f"{'='*60}")
        
        # Analyze dataset first
        prompts = self.analyze_dataset(csv_path, dataset_name)
        if prompts is None:
            print(f"Skipping {dataset_name} due to analysis failure")
            return None
        
        if len(prompts) == 0:
            print(f"No valid prompts found in {dataset_name}")
            return None
        
        print(f"\nProcessing ALL {len(prompts)} prompts (no limit applied)...")
        
        # Classification results
        results = {
            'dataset_name': dataset_name,
            'total_tested': len(prompts),
            'successful_classifications': 0,
            'failed_classifications': 0,
            'predictions': [],
            'probabilities': [],
            'logits': [],
            'processing_errors': [],
            'expected_label': expected_label,
            'sample_texts': []  # Store first 10 samples for reference
        }
        
        # Progress tracking for large datasets
        progress_interval = max(1, len(prompts) // 20)  # Show progress 20 times
        
        for i, prompt in enumerate(prompts):
            if i % progress_interval == 0 or i == len(prompts) - 1:
                print(f"Progress: {i+1}/{len(prompts)} ({(i+1)/len(prompts)*100:.1f}%)")
            
            # Store first 10 samples for reference
            if i < 10:
                results['sample_texts'].append(str(prompt)[:200])  # First 200 chars
            
            try:
                # Ensure prompt is a string
                prompt_str = str(prompt).strip()
                if len(prompt_str) == 0:
                    results['failed_classifications'] += 1
                    results['processing_errors'].append(f"Sample {i+1}: Empty prompt")
                    continue
                
                # Tokenize
                tokenized = self.tokenizer(
                    prompt_str,
                    return_tensors="pt",
                    max_length=self.tokenizer.model_max_length,
                    padding='max_length',
                    truncation=True
                )["input_ids"].to(Config.DEVICE)
                
                # Get embedding
                with torch.no_grad():
                    embedding = self.hysac_model.encode_text(tokenized, project=True)
                    hyperbolic_embedding = add_time_component(embedding, self.lorentz_mlr.manifold.k.item())
                    
                    # Classify
                    logits, probabilities = classify_embeddings(
                        self.lorentz_mlr, 
                        hyperbolic_embedding, 
                        f"Sample {i+1}"  # Only print details for first 5
                    )
                    
                    if probabilities is not None:
                        prob_value = probabilities.item()
                        logit_value = logits.item()
                        prediction = 1 if prob_value >= 0.5 else 0
                        
                        results['probabilities'].append(prob_value)
                        results['logits'].append(logit_value)
                        results['predictions'].append(prediction)
                        results['successful_classifications'] += 1
                    else:
                        results['failed_classifications'] += 1
                        results['processing_errors'].append(f"Sample {i+1}: Classification returned None")
                        
            except Exception as e:
                results['failed_classifications'] += 1
                error_msg = f"Sample {i+1}: {str(e)[:100]}"
                results['processing_errors'].append(error_msg)
                
                # Stop if too many consecutive errors
                if len(results['processing_errors']) > 100:
                    print(f"Warning: Too many errors encountered. Stopping after {i+1} samples.")
                    break
        
        # Calculate statistics
        if results['successful_classifications'] > 0:
            probs = np.array(results['probabilities'])
            logits_array = np.array(results['logits'])
            predictions = np.array(results['predictions'])
            
            results['statistics'] = {
                'success_rate': results['successful_classifications'] / results['total_tested'],
                'mean_probability': float(probs.mean()),
                'std_probability': float(probs.std()),
                'median_probability': float(np.median(probs)),
                'min_probability': float(probs.min()),
                'max_probability': float(probs.max()),
                'mean_logit': float(logits_array.mean()),
                'std_logit': float(logits_array.std()),
                'median_logit': float(np.median(logits_array)),
                'min_logit': float(logits_array.min()),
                'max_logit': float(logits_array.max()),
                'predictions_summary': {
                    'benign_count': int(np.sum(predictions == 0)),
                    'malicious_count': int(np.sum(predictions == 1)),
                    'benign_percentage': float(np.sum(predictions == 0) / len(predictions) * 100),
                    'malicious_percentage': float(np.sum(predictions == 1) / len(predictions) * 100)
                }
            }
            
            # If expected label is provided, calculate accuracy
            if expected_label is not None:
                correct_predictions = np.sum(predictions == expected_label)
                accuracy = correct_predictions / len(predictions)
                results['statistics']['accuracy'] = float(accuracy)
                results['statistics']['correct_predictions'] = int(correct_predictions)
                
                # Calculate additional metrics
                if expected_label == 1:  # Malicious dataset
                    true_positives = int(np.sum(predictions == 1))
                    false_negatives = int(np.sum(predictions == 0))
                    results['statistics']['true_positives'] = true_positives
                    results['statistics']['false_negatives'] = false_negatives
                    results['statistics']['recall'] = float(true_positives / (true_positives + false_negatives)) if (true_positives + false_negatives) > 0 else 0.0
                else:  # Benign dataset
                    true_negatives = int(np.sum(predictions == 0))
                    false_positives = int(np.sum(predictions == 1))
                    results['statistics']['true_negatives'] = true_negatives
                    results['statistics']['false_positives'] = false_positives
                    results['statistics']['specificity'] = float(true_negatives / (true_negatives + false_positives)) if (true_negatives + false_positives) > 0 else 0.0
        
        # Print results
        self.print_classification_results(results)
        
        # Store results
        self.results[dataset_name] = results
        
        return results
    


    def compute_precision_recall_f1(self):
        """
        Compute precision, recall and F1 scores for each dataset pair (safe/unsafe)
        """
        # Group datasets by their base names
        dataset_pairs = {}
        for dataset_name, result in self.results.items():
            # Skip datasets without expected labels or statistics
            if ('expected_label' not in result or result['expected_label'] is None or 
                'statistics' not in result):
                continue
            
            # Extract base name (without safe/unsafe suffix)
            base_name = None
            if '_safe' in dataset_name:
                base_name = dataset_name.replace('_safe', '')
            elif '_unsafe' in dataset_name:
                base_name = dataset_name.replace('_unsafe', '')
                
            if base_name:
                if base_name not in dataset_pairs:
                    dataset_pairs[base_name] = {'safe': None, 'unsafe': None}
                    
                if result['expected_label'] == 0:  # Safe dataset
                    dataset_pairs[base_name]['safe'] = result
                else:  # Unsafe dataset
                    dataset_pairs[base_name]['unsafe'] = result
        
        # Calculate metrics for each pair
        metrics = {}
        for base_name, pair in dataset_pairs.items():
            safe_data = pair['safe']
            unsafe_data = pair['unsafe']
            
            # Skip if either safe or unsafe data is missing
            if safe_data is None or unsafe_data is None:
                continue
                
            if 'statistics' in safe_data and 'statistics' in unsafe_data:
                # Combine the confusion matrix elements
                tp = unsafe_data['statistics'].get('true_positives', 0)
                fn = unsafe_data['statistics'].get('false_negatives', 0)
                tn = safe_data['statistics'].get('true_negatives', 0)
                fp = safe_data['statistics'].get('false_positives', 0)
                
                # Calculate precision, recall and F1
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                metrics[base_name] = {
                    'precision': precision * 100,
                    'recall': recall * 100,
                    'f1_score': f1 * 100,
                    'tp': tp,
                    'fp': fp,
                    'tn': tn,
                    'fn': fn
                }
        
        return metrics
    
    def print_classification_results(self, results):
        """Print formatted classification results"""
        print(f"\nCLASSIFICATION RESULTS for {results['dataset_name']}")
        print("-" * 60)
        print(f"Total tested: {results['total_tested']}")
        print(f"Successful: {results['successful_classifications']}")
        print(f"Failed: {results['failed_classifications']}")
        
        if 'statistics' in results:
            stats = results['statistics']
            print(f"Success rate: {stats['success_rate']*100:.1f}%")
            
            print(f"\nProbability Statistics:")
            print(f"  Mean: {stats['mean_probability']:.4f}")
            print(f"  Std: {stats['std_probability']:.4f}")
            print(f"  Median: {stats['median_probability']:.4f}")
            print(f"  Range: [{stats['min_probability']:.4f}, {stats['max_probability']:.4f}]")
            
            print(f"\nLogit Statistics:")
            print(f"  Mean: {stats['mean_logit']:.4f}")
            print(f"  Std: {stats['std_logit']:.4f}")
            print(f"  Median: {stats['median_logit']:.4f}")
            print(f"  Range: [{stats['min_logit']:.4f}, {stats['max_logit']:.4f}]")
            
            print(f"\nPrediction Summary:")
            print(f"  Benign (0): {stats['predictions_summary']['benign_count']:,} ({stats['predictions_summary']['benign_percentage']:.1f}%)")
            print(f"  Malicious (1): {stats['predictions_summary']['malicious_count']:,} ({stats['predictions_summary']['malicious_percentage']:.1f}%)")
            
            if 'accuracy' in stats:
                print(f"  Accuracy: {stats['accuracy']*100:.1f}%")
                
            if 'recall' in stats:
                print(f"  Recall (Sensitivity): {stats['recall']*100:.1f}%")
                
            if 'specificity' in stats:
                print(f"  Specificity: {stats['specificity']*100:.1f}%")
        
        if results['processing_errors']:
            print(f"\nProcessing errors: {len(results['processing_errors'])}")
            if len(results['processing_errors']) <= 10:
                print("All errors:")
                for error in results['processing_errors']:
                    print(f"  {error}")
            else:
                print("First 5 errors:")
                for error in results['processing_errors'][:5]:
                    print(f"  {error}")
                print(f"  ... and {len(results['processing_errors']) - 5} more errors")
    
    def save_results(self):
        """Save all results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results as JSON
        results_file = self.output_dir / f"classification_results_full_{timestamp}.json"
        
        # Prepare data for JSON serialization (remove large arrays for main file)
        json_data = {
            'dataset_stats': self.dataset_stats,
            'classification_results': {},
            'timestamp': timestamp,
            'config': {
                'curvature_k': Config.CURVATURE_K,
                'num_features': Config.NUM_FEATURES,
                'num_classes': Config.NUM_CLASSES,
                'device': Config.DEVICE
            }
        }
        
        # Save summary data (without full arrays)
        for dataset_name, result in self.results.items():
            summary_result = {
                'dataset_name': result['dataset_name'],
                'total_tested': result['total_tested'],
                'successful_classifications': result['successful_classifications'],
                'failed_classifications': result['failed_classifications'],
                'expected_label': result['expected_label'],
                'sample_texts': result['sample_texts'],
                'processing_errors': result['processing_errors'][:10],  # Only first 10 errors
                'statistics': result.get('statistics', {})
            }
            json_data['classification_results'][dataset_name] = summary_result
        
        with open(results_file, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"Results summary saved to: {results_file}")
        
        # Save detailed predictions for each dataset separately
        for dataset_name, result in self.results.items():
            if 'probabilities' in result and result['probabilities']:
                detail_file = self.output_dir / f"detailed_predictions_{dataset_name}_{timestamp}.json"
                detailed_data = {
                    'dataset_name': dataset_name,
                    'probabilities': result['probabilities'],
                    'logits': result['logits'],
                    'predictions': result['predictions'],
                    'statistics': result.get('statistics', {})
                }
                with open(detail_file, 'w') as f:
                    json.dump(detailed_data, f, indent=2)
                print(f"Detailed predictions for {dataset_name} saved to: {detail_file}")
        
        # Save summary CSV
        summary_data = []
        for dataset_name, result in self.results.items():
            if 'statistics' in result:
                stats = result['statistics']
                row = {
                    'dataset': dataset_name,
                    'total_samples': result['total_tested'],
                    'successful_classifications': result['successful_classifications'],
                    'failed_classifications': result['failed_classifications'],
                    'success_rate': stats['success_rate'],
                    'mean_probability': stats['mean_probability'],
                    'std_probability': stats['std_probability'],
                    'median_probability': stats['median_probability'],
                    'benign_count': stats['predictions_summary']['benign_count'],
                    'malicious_count': stats['predictions_summary']['malicious_count'],
                    'benign_percentage': stats['predictions_summary']['benign_percentage'],
                    'malicious_percentage': stats['predictions_summary']['malicious_percentage'],
                    'accuracy': stats.get('accuracy', 'N/A'),
                    'recall': stats.get('recall', 'N/A'),
                    'specificity': stats.get('specificity', 'N/A')
                }
                summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        summary_file = self.output_dir / f"classification_summary_full_{timestamp}.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"Summary CSV saved to: {summary_file}")
        
        return results_file, summary_file
    
    def create_visualizations(self):
        """Create comprehensive visualization plots"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Skip visualization if no successful results
        if not any('statistics' in result for result in self.results.values()):
            print("No successful results to visualize")
            return []
        
        # Create multiple figure files for better organization
        
        # Figure 1: Probability distributions and basic stats
        fig1, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig1.suptitle('Classification Probability Analysis', fontsize=16)
        
        # Plot 1: Probability distributions
        ax1 = axes[0, 0]
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        for i, (dataset_name, result) in enumerate(self.results.items()):
            if 'probabilities' in result and result['probabilities']:
                ax1.hist(result['probabilities'], alpha=0.6, label=dataset_name, 
                        bins=50, color=colors[i % len(colors)], density=True)
        ax1.set_xlabel('Probability')
        ax1.set_ylabel('Density')
        ax1.set_title('Probability Distributions (Normalized)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axvline(x=0.5, color='black', linestyle='--', alpha=0.7, label='Decision Threshold')
        
        # Plot 2: Success rates
        ax2 = axes[0, 1]
        datasets = []
        success_rates = []
        colors_bar = []
        for i, (dataset_name, result) in enumerate(self.results.items()):
            if 'statistics' in result:
                datasets.append(dataset_name.replace('_', '\n'))
                success_rates.append(result['statistics']['success_rate'] * 100)
                colors_bar.append(colors[i % len(colors)])
        
        if datasets:
            bars = ax2.bar(range(len(datasets)), success_rates, color=colors_bar, alpha=0.7)
            ax2.set_xlabel('Dataset')
            ax2.set_ylabel('Success Rate (%)')
            ax2.set_title('Classification Success Rates')
            ax2.set_xticks(range(len(datasets)))
            ax2.set_xticklabels(datasets, rotation=45, ha='right')
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, 105)
            
            # Add value labels on bars
            for bar, rate in zip(bars, success_rates):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                        f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Plot 3: Box plot of probabilities
        ax3 = axes[1, 0]
        prob_data = []
        labels = []
        for dataset_name, result in self.results.items():
            if 'probabilities' in result and result['probabilities']:
                prob_data.append(result['probabilities'])
                labels.append(dataset_name.replace('_', '\n'))
        
        if prob_data:
            bp = ax3.boxplot(prob_data, labels=labels, patch_artist=True)
            for patch, color in zip(bp['boxes'], colors_bar[:len(prob_data)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            ax3.set_xlabel('Dataset')
            ax3.set_ylabel('Probability')
            ax3.set_title('Probability Distributions (Box Plot)')
            ax3.tick_params(axis='x', rotation=45)
            ax3.grid(True, alpha=0.3)
            ax3.axhline(y=0.5, color='black', linestyle='--', alpha=0.7)
        
        # Plot 4: Accuracy comparison (if available)
        ax4 = axes[1, 1]
        acc_datasets = []
        accuracies = []
        acc_colors = []
        for i, (dataset_name, result) in enumerate(self.results.items()):
            if 'statistics' in result and 'accuracy' in result['statistics']:
                acc_datasets.append(dataset_name.replace('_', '\n'))
                accuracies.append(result['statistics']['accuracy'] * 100)
                acc_colors.append(colors[i % len(colors)])
        
        if acc_datasets:
            bars = ax4.bar(range(len(acc_datasets)), accuracies, color=acc_colors, alpha=0.7)
            ax4.set_xlabel('Dataset')
            ax4.set_ylabel('Accuracy (%)')
            ax4.set_title('Classification Accuracy (vs Expected Labels)')
            ax4.set_xticks(range(len(acc_datasets)))
            ax4.set_xticklabels(acc_datasets, rotation=45, ha='right')
            ax4.grid(True, alpha=0.3)
            ax4.set_ylim(0, 105)
            
            # Add value labels
            for bar, acc in zip(bars, accuracies):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                        f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
        else:
            ax4.text(0.5, 0.5, 'No accuracy data\n(expected labels not provided)', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Classification Accuracy')
        
        plt.tight_layout()
        plot_file1 = self.output_dir / f"probability_analysis_{timestamp}.png"
        plt.savefig(plot_file1, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Figure 2: Prediction distributions
        fig2, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        dataset_names = []
        benign_counts = []
        malicious_counts = []
        total_counts = []
        
        for dataset_name, result in self.results.items():
            if 'statistics' in result:
                dataset_names.append(dataset_name.replace('_', '\n'))
                benign_counts.append(result['statistics']['predictions_summary']['benign_count'])
                malicious_counts.append(result['statistics']['predictions_summary']['malicious_count'])
                total_counts.append(result['total_tested'])
        
        if dataset_names:
            x = np.arange(len(dataset_names))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, benign_counts, width, label='Benign (0)', 
                          alpha=0.8, color='green')
            bars2 = ax.bar(x + width/2, malicious_counts, width, label='Malicious (1)', 
                          alpha=0.8, color='red')
            
            ax.set_xlabel('Dataset')
            ax.set_ylabel('Number of Predictions')
            ax.set_title('Prediction Distribution by Dataset (Absolute Counts)')
            ax.set_xticks(x)
            ax.set_xticklabels(dataset_names, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add count labels on bars
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    if total_counts:
                        ax.text(bar.get_x() + bar.get_width()/2., height + max(total_counts)*0.01,
                               f'{int(height):,}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plot_file2 = self.output_dir / f"prediction_distribution_{timestamp}.png"
        plt.savefig(plot_file2, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualizations saved to:")
        print(f"  - {plot_file1}")
        print(f"  - {plot_file2}")
        
        return [plot_file1, plot_file2]


def main():
    """Main testing function - processes ALL samples in each dataset"""
    print("Starting Comprehensive Hyperbolic MLR Testing (FULL DATASETS) - ROBUST VERSION")
    print(f"Using device: {Config.DEVICE}")
    
    # Paths
    state_dict_path = "/mnt/ssd1/mary/Diffusion-Models-Embedding-Space-Defense/LorentzMLR/final_hyperbolic_mlr_model.pth"
    
    # Dataset configurations: (path, name, expected_label)
    datasets = [
        # ('/mnt/ssd1/mary/Diffusion-Models-Embedding-Space-Defense/data/coco_30k.csv', 
        #  'coco_safe', 0),  # Expected benign
        ('/mnt/ssd1/mary/Diffusion-Models-Embedding-Space-Defense/data/safe_visu_text_prompts.csv', 
         'visu_prompts_safe', 0),  # Expected benign
        ('/mnt/ssd1/mary/Diffusion-Models-Embedding-Space-Defense/data/unsafe_visu_text_prompts.csv', 
         'visu_prompts_unsafe', 1),  # Expected malicious
        ('/mnt/ssd1/mary/Diffusion-Models-Embedding-Space-Defense/data/4chan_labeled_safe.csv',
         '4chan_safe', 0),
        ('/mnt/ssd1/mary/Diffusion-Models-Embedding-Space-Defense/data/4chan_labeled_unsafe.csv',
         '4chan_unsafe', 1),
        # ('/mnt/ssd1/mary/Diffusion-Models-Embedding-Space-Defense/data/mma_NSFW_prompts_labeled_safe.csv',
        #  'mma_NSFW_safe', 0),
        # ('/mnt/ssd1/mary/Diffusion-Models-Embedding-Space-Defense/data/mma_NSFW_prompts_labeled_unsafe.csv',
        #  'mma_NSFW_unsafe', 1),
        # (
        #     '/mnt/ssd1/mary/Diffusion-Models-Embedding-Space-Defense/data/nsfw_200_labeled_safe.csv',
        #     'NSFW_200_safe', 0
        # ),
        # (
        #     '/mnt/ssd1/mary/Diffusion-Models-Embedding-Space-Defense/data/nsfw_200_labeled_unsafe.csv',
        #     'NSFW_200_unsafe', 1
        # )
    ]
    
    # Initialize tester
    try:
        tester = DatasetTester(state_dict_path)
    except Exception as e:
        print(f"Failed to initialize tester: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test each dataset (ALL samples)
    for csv_path, dataset_name, expected_label in datasets:
        if Path(csv_path).exists():
            print(f"\n{'*'*80}")
            print(f"PROCESSING DATASET: {dataset_name}")
            print(f"Expected to process ALL samples in: {csv_path}")
            print(f"{'*'*80}")
            try:
                tester.test_dataset(csv_path, dataset_name, expected_label)
            except Exception as e:
                print(f"Failed to process dataset {dataset_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        else:
            print(f"Warning: Dataset file not found: {csv_path}")
    
    # Only proceed with results saving if we have any results
    if not tester.results:
        print("No datasets were successfully processed. Exiting.")
        return
    
    # Save results and create visualizations
    print(f"\n{'='*60}")
    print("SAVING RESULTS AND CREATING VISUALIZATIONS")
    print(f"{'='*60}")
    
    try:
        results_file, summary_file = tester.save_results()
        plot_files = tester.create_visualizations()
    except Exception as e:
        print(f"Error saving results: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Print comprehensive final summary
    print(f"\n{'='*60}")
    print("COMPREHENSIVE FINAL SUMMARY")
    print(f"{'='*60}")
    
    total_tested = sum(result['total_tested'] for result in tester.results.values())
    total_successful = sum(result['successful_classifications'] for result in tester.results.values())
    total_failed = sum(result['failed_classifications'] for result in tester.results.values())
    
    print(f"Datasets processed: {len(tester.results)}")
    print(f"Total samples tested: {total_tested:,}")
    print(f"Total successful classifications: {total_successful:,}")
    print(f"Total failed classifications: {total_failed:,}")
    if total_tested > 0:
        print(f"Overall success rate: {total_successful/total_tested*100:.2f}%")
    
    # Per-dataset summary
    if tester.results:
        print(f"\nPer-dataset summary:")
        print(f"{'Dataset':<20} {'Samples':<10} {'Success%':<10} {'Accuracy%':<10} {'Benign%':<10} {'Malicious%':<10}")
        print("-" * 80)
        
        for dataset_name, result in tester.results.items():
            if 'statistics' in result:
                stats = result['statistics']
                accuracy = f"{stats.get('accuracy', 0)*100:.1f}" if 'accuracy' in stats else "N/A"
                print(f"{dataset_name:<20} {result['total_tested']:<10,} "
                      f"{stats['success_rate']*100:<10.1f} {accuracy:<10} "
                      f"{stats['predictions_summary']['benign_percentage']:<10.1f} "
                      f"{stats['predictions_summary']['malicious_percentage']:<10.1f}")
        
        print(f"\nFiles generated:")
        print(f"  - Main results: {results_file}")
        print(f"  - Summary CSV: {summary_file}")
        if plot_files:
            print(f"  - Visualization plots: {len(plot_files)} files")
            for plot_file in plot_files:
                print(f"    * {plot_file}")
        
        print(f"\nDetailed prediction files saved for each dataset in: {tester.output_dir}")
    

        # Calculate precision, recall and F1 scores
        precision_recall_f1 = tester.compute_precision_recall_f1()

        # Print Per-dataset summary with precision, recall, F1
        if precision_recall_f1:
            print(f"\nPrecision, Recall, F1 Score summary by dataset pair:")
            print(f"{'Dataset':<20} {'Precision':<10} {'Recall':<10} {'F1 Score':<10}")
            print("-" * 60)
            
            for base_name, metrics in precision_recall_f1.items():
                print(f"{base_name:<20} {metrics['precision']:.2f}% {metrics['recall']:.2f}% {metrics['f1_score']:.2f}%")

    if not tester.results:
        print("No datasets were successfully processed. Exiting.")
        return

    
    print("\nRobust full dataset testing completed!")


if __name__ == "__main__":
    main()