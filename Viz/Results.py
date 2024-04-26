def rank_results(results):
    metrics = [{'name': 'train_loss', 'ascending': True}, {'name': 'train_accuracy', 'ascending': False}]
    def extract_last_values(result, metrics):
        return {metric: result.get(metric, [-1])[-1] for metric in metrics}

    def sort_key(result):
        # Extract last values based on provided metrics and their desired sort order
        values = extract_last_values(result, [m['name'] for m in metrics])
        return tuple((values[m['name']] if m['ascending'] else -values[m['name']]) for m in metrics)

    ranked_results = sorted(results, key=sort_key)

    formatted_results = ""
    for i, result in enumerate(ranked_results, start=1):
        formatted_results += f"Rank {i}:\n"
        formatted_results += f"  Learning Rate: {result.get('learning_rate', 'N/A')}\n"
        formatted_results += f"  Hidden Layer Size: {result.get('hidden_layer_size', 'N/A')}\n"
        formatted_results += f"  Last Values:\n"
        last_values = extract_last_values(result, [m['name'] for m in metrics])
        for metric in metrics:
            formatted_results += f"    {metric['name']}: {last_values.get(metric['name'], 'N/A')}\n"
        formatted_results += "\n"
    return formatted_results
