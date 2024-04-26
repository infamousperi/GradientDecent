def rank_results(results):
    metrics = [{'name': 'train_loss', 'ascending': True}, {'name': 'train_accuracy', 'ascending': False}]

    def extract_last_values(result, metric_names):
        # Retrieve the last value for each metric from the result
        return {metric: result.get(metric, [-1])[-1] for metric in metric_names}

    def sort_key(result):
        # Create a tuple of metrics for sorting
        values = extract_last_values(result, [m['name'] for m in metrics])
        return tuple((values[m['name']] if m['ascending'] else -values[m['name']]) for m in metrics)

    # Sort results based on the specified metrics and their order
    ranked_results = sorted(results, key=sort_key)

    # Print out the ranked results in markdown format
    print("## Ranked Results\n")
    for i, result in enumerate(ranked_results, start=1):
        last_values = extract_last_values(result, [m['name'] for m in metrics])
        print(f"**Rank {i}:** Learning Rate = {result.get('learning_rate', 'N/A')}, ", end="")
        print(f"Hidden Layer Size = {result.get('hidden_layer_size', 'N/A')}, ", end="")
        print(f"Train Loss = {last_values['train_loss']:.3f}, ", end="")
        print(f"Train Accuracy = {last_values['train_accuracy'] * 100:.2f}%\n")
