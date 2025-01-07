from Advanced_version.main import run_simulation

if __name__ == "__main__":
    real_metrics, synthetic_metrics = run_simulation()
    
    print("\nReal Data Metrics:")
    for metric, value in real_metrics.items():
        print(f"{metric}: {value:.2f}")
        
    print("\nSynthetic Data Metrics:")
    for metric, value in synthetic_metrics.items():
        print(f"{metric}: {value:.2f}")