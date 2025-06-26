"""
Comprehensive analysis script for GAN training metrics
Clean, organized version of the analysis functionality
"""
import os
import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.analysis.metrics_analyzer import MetricsAnalyzer


class ComprehensiveAnalysis:
    """Main class for comprehensive GAN training analysis"""
    
    def __init__(self, metrics_dir: str = "data/metrics"):
        self.metrics_dir = Path(metrics_dir)
        self.analyzer = MetricsAnalyzer()
        self.output_dir = Path("outputs/analysis")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def run_complete_analysis(self) -> None:
        """Run complete analysis for all available metrics"""
        print("Starting comprehensive GAN training analysis...")
        
        # Get all available metrics files
        metrics_files = self._get_metrics_files()
        
        if not metrics_files:
            print(f"No metrics files found in {self.metrics_dir}")
            return
        
        # Generate individual reports
        self._generate_individual_reports(metrics_files)
        
        # Generate comparison plots
        self._generate_comparison_plots(metrics_files)
        
        # Generate summary report
        self._generate_summary_report(metrics_files)
        
        print(f"Analysis complete! Results saved to {self.output_dir}")
    
    def _get_metrics_files(self) -> dict:
        """Get all available metrics files organized by type"""
        files = {
            'fid_files': [],
            'stats_files': []
        }
        
        if not self.metrics_dir.exists():
            return files
        
        for file_path in self.metrics_dir.glob("*.jsonl"):
            filename = file_path.name.lower()
            
            if 'fid' in filename:
                files['fid_files'].append(file_path)
            elif 'stats' in filename:
                files['stats_files'].append(file_path)
        
        return files
    
    def _generate_individual_reports(self, metrics_files: dict) -> None:
        """Generate individual analysis reports for each model"""
        print("Generating individual reports...")
        
        for file_path in metrics_files['fid_files'] + metrics_files['stats_files']:
            model_name = self._extract_model_name(file_path.name)
            print(f"  Analyzing {model_name}...")
            
            report = self.analyzer.generate_training_report(
                str(file_path),
                str(self.output_dir / f"{model_name}_report.json")
            )
            
            self._save_text_report(report, model_name)
    
    def _generate_comparison_plots(self, metrics_files: dict) -> None:
        """Generate comparison plots between different models"""
        print("Generating comparison plots...")
        
        if len(metrics_files['fid_files']) < 2:
            print("  Not enough FID files for comparison")
            return
        
        # Prepare data for plotting
        file_paths = [str(f) for f in metrics_files['fid_files']]
        model_names = [self._extract_model_name(f.name) for f in metrics_files['fid_files']]
        
        # Generate FID comparison plot
        fig = self.analyzer.plot_fid_progression(
            file_paths,
            model_names,
            str(self.output_dir / "fid_comparison.png"),
            smooth=True
        )
        
        print(f"  FID comparison plot saved to {self.output_dir / 'fid_comparison.png'}")
    
    def _generate_summary_report(self, metrics_files: dict) -> None:
        """Generate overall summary report"""
        print("Generating summary report...")
        
        summary = {
            'analyzed_models': [],
            'best_performers': {},
            'recommendations': []
        }
        
        # Analyze each model
        for file_path in metrics_files['fid_files']:
            model_name = self._extract_model_name(file_path.name)
            data = self.analyzer.load_jsonl_data(str(file_path))
            
            if data:
                iterations, fid_scores = self.analyzer.extract_fid_scores(data)
                
                model_summary = {
                    'name': model_name,
                    'final_fid': fid_scores[-1] if fid_scores else None,
                    'best_fid': min(fid_scores) if fid_scores else None,
                    'total_iterations': len(iterations),
                    'convergence_stability': self.analyzer._calculate_stability_score(fid_scores)
                }
                
                summary['analyzed_models'].append(model_summary)
        
        # Determine best performers
        if summary['analyzed_models']:
            best_fid = min(m['best_fid'] for m in summary['analyzed_models'] if m['best_fid'])
            best_model = next(m for m in summary['analyzed_models'] if m['best_fid'] == best_fid)
            summary['best_performers']['best_fid'] = best_model
            
            most_stable = max(m for m in summary['analyzed_models'], key=lambda x: x['convergence_stability'])
            summary['best_performers']['most_stable'] = most_stable
        
        # Generate recommendations
        summary['recommendations'] = self._generate_recommendations(summary['analyzed_models'])
        
        # Save summary
        import json
        with open(self.output_dir / "summary_report.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Save human-readable summary
        self._save_human_readable_summary(summary)
    
    def _extract_model_name(self, filename: str) -> str:
        """Extract clean model name from filename"""
        name = filename.replace('.jsonl', '')
        name = name.replace('_fid50k', '').replace('_stats', '')
        name = name.replace('_metric-fid50k_full', '')
        return name
    
    def _save_text_report(self, report: dict, model_name: str) -> None:
        """Save human-readable text report"""
        if not report:
            return
        
        with open(self.output_dir / f"{model_name}_report.txt", 'w') as f:
            f.write(f"Training Analysis Report: {model_name}\n")
            f.write("=" * 50 + "\n\n")
            
            # Model info
            if 'model_info' in report:
                info = report['model_info']
                f.write("Model Information:\n")
                f.write(f"  Total Iterations: {info.get('total_iterations', 'N/A')}\n")
                f.write(f"  Final FID Score: {info.get('final_fid', 'N/A'):.2f}\n" if info.get('final_fid') else "  Final FID Score: N/A\n")
                f.write(f"  Best FID Score: {info.get('best_fid', 'N/A'):.2f}\n" if info.get('best_fid') else "  Best FID Score: N/A\n")
                f.write(f"  Best FID Iteration: {info.get('best_fid_iteration', 'N/A')}\n")
                f.write("\n")
            
            # Summary
            if 'summary' in report:
                summary = report['summary']
                f.write("Training Summary:\n")
                f.write(f"  Convergence Rate: {summary.get('convergence_rate', 0):.3f}\n")
                f.write(f"  Stability Score: {summary.get('stability_score', 0):.3f}\n")
                f.write(f"  Training Efficiency: {summary.get('training_efficiency', 0):.3f}\n")
    
    def _save_human_readable_summary(self, summary: dict) -> None:
        """Save human-readable overall summary"""
        with open(self.output_dir / "summary_report.txt", 'w') as f:
            f.write("GAN Training Analysis Summary\n")
            f.write("=" * 40 + "\n\n")
            
            f.write(f"Total Models Analyzed: {len(summary['analyzed_models'])}\n\n")
            
            # Best performers
            if summary['best_performers']:
                f.write("Best Performers:\n")
                if 'best_fid' in summary['best_performers']:
                    best = summary['best_performers']['best_fid']
                    f.write(f"  Best FID Score: {best['name']} (FID: {best['best_fid']:.2f})\n")
                
                if 'most_stable' in summary['best_performers']:
                    stable = summary['best_performers']['most_stable']
                    f.write(f"  Most Stable Training: {stable['name']} (Stability: {stable['convergence_stability']:.3f})\n")
                f.write("\n")
            
            # All models
            f.write("All Models:\n")
            for model in summary['analyzed_models']:
                f.write(f"  {model['name']}:\n")
                f.write(f"    Best FID: {model['best_fid']:.2f}\n" if model['best_fid'] else "    Best FID: N/A\n")
                f.write(f"    Final FID: {model['final_fid']:.2f}\n" if model['final_fid'] else "    Final FID: N/A\n")
                f.write(f"    Iterations: {model['total_iterations']}\n")
                f.write(f"    Stability: {model['convergence_stability']:.3f}\n")
                f.write("\n")
            
            # Recommendations
            if summary['recommendations']:
                f.write("Recommendations:\n")
                for i, rec in enumerate(summary['recommendations'], 1):
                    f.write(f"  {i}. {rec}\n")
    
    def _generate_recommendations(self, models: list) -> list:
        """Generate training recommendations based on analysis"""
        recommendations = []
        
        if not models:
            return recommendations
        
        # Find best and worst performers
        valid_models = [m for m in models if m['best_fid'] is not None]
        
        if valid_models:
            best_model = min(valid_models, key=lambda x: x['best_fid'])
            worst_model = max(valid_models, key=lambda x: x['best_fid'])
            
            recommendations.append(f"Best performing model: {best_model['name']} with FID {best_model['best_fid']:.2f}")
            
            if len(valid_models) > 1:
                recommendations.append(f"Consider using {best_model['name']} configuration for future training")
            
            # Stability recommendations
            most_stable = max(valid_models, key=lambda x: x['convergence_stability'])
            if most_stable['convergence_stability'] > 0.8:
                recommendations.append(f"{most_stable['name']} shows excellent training stability")
            
            # Training length recommendations
            avg_iterations = sum(m['total_iterations'] for m in valid_models) / len(valid_models)
            if best_model['total_iterations'] < avg_iterations:
                recommendations.append("Best model achieved good results with fewer iterations - consider early stopping")
        
        return recommendations


def main():
    """Main entry point for analysis script"""
    parser = argparse.ArgumentParser(description="Comprehensive GAN training analysis")
    parser.add_argument(
        "--metrics-dir",
        default="data/metrics",
        help="Directory containing metrics files"
    )
    
    args = parser.parse_args()
    
    analyzer = ComprehensiveAnalysis(args.metrics_dir)
    analyzer.run_complete_analysis()


if __name__ == "__main__":
    main()
