"""
Statistical analysis tools for GAN training metrics
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from typing import List, Dict, Tuple, Any
import os


class MetricsAnalyzer:
    """Analyze and visualize GAN training metrics"""
    
    def __init__(self):
        self.data_cache = {}
    
    def load_jsonl_data(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load and parse JSONL data from file
        
        Args:
            file_path: Path to JSONL file
        
        Returns:
            List of parsed JSON objects
        """
        if file_path in self.data_cache:
            return self.data_cache[file_path]
        
        data = []
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data.append(json.loads(line))
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            return []
        except json.JSONDecodeError as e:
            print(f"JSON decode error in {file_path}: {e}")
            return []
        
        self.data_cache[file_path] = data
        return data
    
    def calculate_moving_average(self, data: List[float], window_size: int = 3) -> List[float]:
        """
        Calculate moving average with specified window size
        
        Args:
            data: List of numeric values
            window_size: Size of the moving average window
        
        Returns:
            List of smoothed values
        """
        if len(data) < window_size:
            return data
        return uniform_filter1d(data, size=window_size, mode='nearest').tolist()
    
    def extract_fid_scores(self, data: List[Dict[str, Any]]) -> Tuple[List[int], List[float]]:
        """
        Extract FID scores and corresponding iterations
        
        Args:
            data: List of metrics data
        
        Returns:
            Tuple of (iterations, fid_scores)
        """
        iterations = []
        fid_scores = []
        
        for entry in data:
            if 'results' in entry and 'fid50k_full' in entry['results']:
                # Extract iteration from snapshot name
                snapshot_name = entry.get('snapshot_pkl', '')
                if 'network-snapshot-' in snapshot_name:
                    iteration_str = snapshot_name.split('network-snapshot-')[1].split('.')[0]
                    try:
                        iteration = int(iteration_str)
                        iterations.append(iteration)
                        fid_scores.append(entry['results']['fid50k_full'])
                    except ValueError:
                        continue
        
        return iterations, fid_scores
    
    def extract_training_stats(self, data: List[Dict[str, Any]]) -> Dict[str, List]:
        """
        Extract various training statistics
        
        Args:
            data: List of training stats data
        
        Returns:
            Dictionary containing extracted metrics
        """
        stats = {
            'kimg': [],
            'fid50k': [],
            'is50k': [],
            'ppl2_wend': [],
            'gpu_memory': [],
            'cpu_memory': [],
            'sec_per_kimg': [],
            'fake_scores': [],
            'real_scores': [],
            'r1_penalty': [],
            'augment_progress': []
        }
        
        for entry in data:
            if 'stats' in entry:
                entry_stats = entry['stats']
                
                # Extract available metrics
                for key in stats.keys():
                    if key in entry_stats:
                        stats[key].append(entry_stats[key])
                    elif key == 'kimg' and 'Progress/kimg' in entry_stats:
                        stats[key].append(entry_stats['Progress/kimg'])
                    elif key == 'sec_per_kimg' and 'Timing/sec_per_kimg' in entry_stats:
                        stats[key].append(entry_stats['Timing/sec_per_kimg'])
        
        return stats
    
    def plot_fid_progression(
        self,
        file_paths: List[str],
        model_names: List[str],
        save_path: str = None,
        smooth: bool = True
    ) -> plt.Figure:
        """
        Plot FID score progression for multiple models
        
        Args:
            file_paths: List of paths to JSONL files
            model_names: List of model names for legend
            save_path: Optional path to save the figure
            smooth: Whether to apply smoothing
        
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
        
        for i, (file_path, model_name) in enumerate(zip(file_paths, model_names)):
            data = self.load_jsonl_data(file_path)
            iterations, fid_scores = self.extract_fid_scores(data)
            
            if not fid_scores:
                continue
            
            if smooth and len(fid_scores) > 5:
                fid_scores = self.calculate_moving_average(fid_scores, window_size=5)
            
            color = colors[i % len(colors)]
            ax.plot(iterations, fid_scores, label=model_name, color=color, linewidth=2)
            ax.scatter(iterations[::5], fid_scores[::5], color=color, alpha=0.6, s=30)
        
        ax.set_xlabel('Training Iterations (kimgs)')
        ax.set_ylabel('FID Score')
        ax.set_title('FID Score Progression During Training')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def generate_training_report(
        self,
        file_path: str,
        output_path: str = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive training report
        
        Args:
            file_path: Path to training stats JSONL file
            output_path: Optional path to save report
        
        Returns:
            Dictionary containing report data
        """
        data = self.load_jsonl_data(file_path)
        
        if not data:
            return {}
        
        # Extract metrics
        iterations, fid_scores = self.extract_fid_scores(data)
        training_stats = self.extract_training_stats(data)
        
        # Calculate statistics
        report = {
            'model_info': {
                'total_iterations': len(data),
                'final_fid': fid_scores[-1] if fid_scores else None,
                'best_fid': min(fid_scores) if fid_scores else None,
                'best_fid_iteration': iterations[fid_scores.index(min(fid_scores))] if fid_scores else None
            },
            'training_stats': training_stats,
            'summary': {
                'convergence_rate': self._calculate_convergence_rate(fid_scores),
                'stability_score': self._calculate_stability_score(fid_scores),
                'training_efficiency': self._calculate_training_efficiency(training_stats)
            }
        }
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
        
        return report
    
    def _calculate_convergence_rate(self, fid_scores: List[float]) -> float:
        """Calculate convergence rate based on FID improvement"""
        if len(fid_scores) < 10:
            return 0.0
        
        initial_fid = np.mean(fid_scores[:5])
        final_fid = np.mean(fid_scores[-5:])
        
        if initial_fid == 0:
            return 0.0
        
        return (initial_fid - final_fid) / initial_fid
    
    def _calculate_stability_score(self, fid_scores: List[float]) -> float:
        """Calculate training stability based on FID variance"""
        if len(fid_scores) < 10:
            return 0.0
        
        # Calculate variance in the last 30% of training
        last_portion = fid_scores[int(len(fid_scores) * 0.7):]
        if len(last_portion) < 5:
            return 0.0
        
        variance = np.var(last_portion)
        mean_fid = np.mean(last_portion)
        
        if mean_fid == 0:
            return 0.0
        
        # Lower coefficient of variation indicates higher stability
        cv = np.sqrt(variance) / mean_fid
        return max(0, 1 - cv)
    
    def _calculate_training_efficiency(self, stats: Dict[str, List]) -> float:
        """Calculate training efficiency metric"""
        if 'sec_per_kimg' not in stats or not stats['sec_per_kimg']:
            return 0.0
        
        avg_time_per_kimg = np.mean(stats['sec_per_kimg'])
        
        # Efficiency is inversely related to time (lower time = higher efficiency)
        if avg_time_per_kimg > 0:
            return 1000.0 / avg_time_per_kimg  # Normalized efficiency score
        
        return 0.0
