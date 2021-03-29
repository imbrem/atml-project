def get_baseline_parameters(model, task_id):
    """Returns the parameters for a given task and model."""
    # Default arguments
    embedding_size = 50
    hidden_size = 50
    root_dir = 'babi_data'
    n_train_to_try = [0]
    learning_rate = 1e-3
    batch_size = 100
    max_iters = 1000
    print_every = 100
    save_every = 2000
    n_targets = 1

    if task_id == 4:
        n_train_to_try = [50, 100, 250, 500, 950]
        batch_size = 20
    elif task_id == 5:
        max_iters = 20000 if model is 'rnn' else 5000
    elif task_id == 16:
        max_iters = 20000 if model is 'rnn' else 5000
        learning_rate = 0.0005
    elif task_id == 18:
        max_iters = 500
        learning_rate = 0.0005
        print_every = 10
        save_every = 1000
    elif task_id == 19:
        max_iters = 10000 if model is 'rnn' else 5000
        batch_size = 20
        n_targets = 2

    parameters = {'model': model,
                  'embedding_size': embedding_size,
                  'hidden_size': hidden_size,
                  'n_targets': n_targets,
                  'learning_rate': learning_rate,
                  'batch_size': batch_size,
                  'max_iters': max_iters,
                  'root_dir': root_dir,
                  'task_id': task_id,
                  'n_train_to_try': n_train_to_try,
                  'print_every': print_every,
                  'save_every': save_every}

    return parameters
